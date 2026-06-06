import pymc as pm
import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
from jax.flatten_util import ravel_pytree
from pymc.sampling.jax import get_jaxified_logp
from jax import tree_util

from blackjax.mcmc.walnuts import walnuts
from blackjax.adaptation.dense_window_adaptation import (
    init_dual_averaging, update_dual_averaging, init_dense_welford, update_dense_welford, get_dense_inverse_mass_matrix
)

def sample_walnuts(model=None, draws=1000, tune=1000, chains=4, random_seed=42):
    model = pm.modelcontext(model)
    print("Auto-assigning WALNUTS sampler...")
    print(f"Compiling PyTensor graph to XLA for {chains} chains, {draws} draws, and {tune} tune steps...")
    
    print("Finding MAP estimate for optimal cold-start...")
    initial_point_dict = pm.find_MAP(model=model, progressbar=False)
    var_names = [var.name for var in model.value_vars]
    q_init, unravel_fn = ravel_pytree(initial_point_dict)
    dim = q_init.shape[0]
    
    jax_logp_list_fn = get_jaxified_logp(model)
    
    def logprob_fn(q_1d):
        pt_dict = unravel_fn(q_1d)
        args = [pt_dict[name] for name in var_names]
        return jax_logp_list_fn(args)
    
    rng = jax.random.PRNGKey(random_seed)
    key_jitter, key_warmup, key_sample = jax.random.split(rng, 3)
    
    q_inits = q_init + jax.random.normal(key_jitter, (chains, dim)) * 0.1
    init_states = jax.vmap(lambda q: walnuts.init(q, logprob_fn))(q_inits)
    
    keys_warmup = jax.random.split(key_warmup, chains)
    keys_sample = jax.random.split(key_sample, chains)
    
    def single_warmup(key, init_state):
        da_s = init_dual_averaging(0.1)
        wel_s = init_dense_welford(dim)
        inv_mass = jnp.eye(dim)
        window_ends = jnp.array([100, 150, 250, 450, 850])
        
        # REFACTOR: Pre-compute the reset state to preserve strict XLA typings
        reset_wel_s = init_dense_welford(dim)
        
        def scan_body(carry, step_idx):
            state, curr_da_s, curr_wel_s, curr_inv_mass, curr_key = carry
            step_key, next_key = jax.random.split(curr_key)
            curr_h = jnp.exp(curr_da_s.log_step_size)
            
            kernel = walnuts(logprob_fn, curr_inv_mass, curr_h).step
            next_state, info = kernel(step_key, state)
            
            next_da_s = update_dual_averaging(curr_da_s, info.unhalved_fraction)
            
            is_slow = (step_idx >= 75) & (step_idx < 850)
            next_wel_s = tree_util.tree_map(
                lambda old, new: jnp.where(is_slow, new, old), 
                curr_wel_s, 
                update_dense_welford(curr_wel_s, next_state.position)
            )
            
            is_update = jnp.any(step_idx == window_ends)
            next_inv_mass = jnp.where(is_update, get_dense_inverse_mass_matrix(next_wel_s), curr_inv_mass)
            
            # REFACTOR: Safely map back to the strictly-typed initial state
            next_wel_s = tree_util.tree_map(
                lambda current, reset: jnp.where(is_update, reset, current), 
                next_wel_s, 
                reset_wel_s
            )
            
            return (next_state, next_da_s, next_wel_s, next_inv_mass, next_key), None

        fin_carry, _ = jax.lax.scan(scan_body, (init_state, da_s, wel_s, inv_mass, key), jnp.arange(tune))
        return {
            "final_state": fin_carry[0], 
            "optimal_h": jnp.exp(fin_carry[1].log_step_size_avg), 
            "inverse_mass_matrix": fin_carry[3]
        }

    def single_sample(key, state, inv_mass, opt_h):
        kernel = walnuts(logprob_fn, inv_mass, opt_h).step
        
        def scan_body(curr_state, curr_key):
            step_key, next_key = jax.random.split(curr_key)
            next_state, _ = kernel(step_key, curr_state)
            return next_state, next_state.position
            
        return jax.lax.scan(scan_body, state, jax.random.split(key, draws))[1]

    num_devices = jax.local_device_count()
    print(f"Detected {num_devices} local JAX devices.")
    
    if num_devices > 1 and chains % num_devices == 0:
        print(f"Deploying Multi-GPU strategy: {chains // num_devices} chains per device.")
        keys_w_reshaped = keys_warmup.reshape(num_devices, chains // num_devices, -1)
        keys_s_reshaped = keys_sample.reshape(num_devices, chains // num_devices, -1)
        
        init_s_reshaped = jax.tree_util.tree_map(
            lambda x: x.reshape(num_devices, chains // num_devices, *x.shape[1:]), 
            init_states
        )

        warmup_res = jax.pmap(jax.vmap(single_warmup))(keys_w_reshaped, init_s_reshaped)
        samples = jax.pmap(jax.vmap(single_sample))(
            keys_s_reshaped,
            warmup_res["final_state"],
            warmup_res["inverse_mass_matrix"],
            warmup_res["optimal_h"]
        )
        samples = samples.reshape(chains, draws, -1)
    else:
        print("Deploying Single-GPU strategy (VMAP).")
        warmup_res = jax.vmap(single_warmup)(keys_warmup, init_states)
        samples = jax.vmap(single_sample)(
            keys_sample,
            warmup_res["final_state"],
            warmup_res["inverse_mass_matrix"],
            warmup_res["optimal_h"]
        )
    
    samples.block_until_ready()
    print("Sampling complete. Formatting InferenceData...")
    
    unraveled_trace = jax.vmap(jax.vmap(unravel_fn))(samples)
    trace_dict = {name: np.array(unraveled_trace[name]) for name in var_names}
    return az.from_dict(posterior=trace_dict)