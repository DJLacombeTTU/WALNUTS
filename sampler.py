import pymc as pm
import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
from jax.flatten_util import ravel_pytree
from pymc.sampling.jax import get_jaxified_logp

# Import the heavy lifting from the backend
from walnuts_backend import new_integrator_state, dense_warmup, dense_sample

def sample_walnuts(model=None, draws=1000, tune=1000, chains=4, random_seed=42):
    """
    A drop-in replacement for pm.sample() using the Dense WALNUTS JAX engine.
    Natively vectorizes multiple chains using jax.vmap.
    """
    model = pm.modelcontext(model)
    print(f"Auto-assigning WALNUTS sampler...")
    print(f"Compiling PyTensor graph to XLA for {chains} chains, {draws} draws, and {tune} tune steps...")
    
    # 1. Flatten the model
    initial_point_dict = model.initial_point()
    var_names = [var.name for var in model.value_vars]
    q_init, unravel_fn = ravel_pytree(initial_point_dict)
    dim = q_init.shape[0]
    
    # 2. Compile to JAX
    jax_logp_list_fn = get_jaxified_logp(model)
    
    def logprob_fn(q_1d):
        pt_dict = unravel_fn(q_1d)
        args = [pt_dict[name] for name in var_names]
        return jax_logp_list_fn(args)
    
    val_and_grad_fn = jax.jit(jax.value_and_grad(logprob_fn))
    
    # 3. Initialize States with Jitter across Chains
    rng = jax.random.PRNGKey(random_seed)
    key_jitter, key_warmup, key_sample = jax.random.split(rng, 3)
    
    # Broadcast initial position and add slight normal jitter for chain diversity
    q_inits = q_init + jax.random.normal(key_jitter, (chains, dim)) * 0.1
    
    # Vmap the gradient evaluation to get initial states for all chains simultaneously
    lp, grad = jax.vmap(val_and_grad_fn)(q_inits)
    init_states = new_integrator_state(q_inits, jnp.zeros_like(q_inits), lp, grad)
    
    keys_warmup = jax.random.split(key_warmup, chains)
    keys_sample = jax.random.split(key_sample, chains)
    
    # 4. Create Vmap Closures for the Backend Functions
    def single_warmup(key, state):
        return dense_warmup(key, state, val_and_grad_fn, num_warmup_steps=tune)

    def single_sample(key, state, inv_mass, opt_h):
        return dense_sample(key, state, val_and_grad_fn, inv_mass, opt_h, delta=0.05, num_draws=draws)

    # 5. Execute Warmup (Vectorized across chains)
    print(f"Executing Dense Warmup vectorized across {chains} chains...")
    warmup_res = jax.vmap(single_warmup)(keys_warmup, init_states)
    
    # 6. Execute Sampling (Vectorized across chains)
    print(f"Sampling Posterior vectorized across {chains} chains...")
    samples = jax.vmap(single_sample)(
        keys_sample,
        warmup_res["final_state"],
        warmup_res["inverse_mass_matrix"],
        warmup_res["optimal_h"]
    )
    
    # Block to ensure the Alienware's hardware is finished before formatting
    samples.block_until_ready()
    print("Sampling complete. Formatting InferenceData...")
    
    # 7. Double-vmap the unravel function: (chains, draws, dim) -> dict of (chains, draws, ...)
    unraveled_trace = jax.vmap(jax.vmap(unravel_fn))(samples)
    trace_dict = {name: np.array(unraveled_trace[name]) for name in var_names}
    
    # 8. Convert directly to ArviZ InferenceData
    idata = az.from_dict(posterior=trace_dict)
    
    return idata