import pymc as pm
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from pymc.sampling.jax import get_jaxified_logp

from integrators import new_integrator_state
from adaptation import walnuts_warmup
from sample import walnuts_sample

def sample_with_walnuts(model, draws=1000, tune=1000, delta=0.05, random_seed=42):
    print("Extracting PyTensor graph and compiling to JAX...")
    
    # 1. Get the initial parameter dictionary and the exact order PyMC expects them
    initial_point_dict = model.initial_point()
    var_names = [var.name for var in model.value_vars]
    
    # 2. Use JAX's rock-solid built-in tree flattener
    # q_init is a 1D array. unravel_fn is a function that converts a 1D array back to a dict.
    q_init, unravel_fn = ravel_pytree(initial_point_dict)
    
    # 3. Get the PyMC JAX-compiled logp function
    jax_logp_list_fn = get_jaxified_logp(model)
    
    # 4. Create the 1D wrapper function for our WALNUTS kernel
    def logprob_fn(q_1d):
        # Reconstruct the parameter dictionary
        pt_dict = unravel_fn(q_1d)
        # Pass arguments in the exact order PyMC expects
        args = [pt_dict[name] for name in var_names]
        return jax_logp_list_fn(args)
    
    # 5. Automatically derive the gradient using JAX
    val_and_grad_fn = jax.jit(jax.value_and_grad(logprob_fn))
    
    # Initialize WALNUTS State
    logprob_init, grad_init = val_and_grad_fn(q_init)
    init_state = new_integrator_state(
        position=q_init,
        momentum=jnp.zeros_like(q_init),
        logprob=logprob_init,
        logprob_grad=grad_init
    )
    
    rng_key = jax.random.PRNGKey(random_seed)
    key_warmup, key_sample = jax.random.split(rng_key)
    
    # 6. Run Warmup
    print(f"Running {tune} Warmup Steps (Dual Averaging + Welford)...")
    warmup_results = walnuts_warmup(
        rng_key=key_warmup,
        initial_state=init_state,
        logprob_grad_fn=val_and_grad_fn,
        num_warmup_steps=tune,
        initial_macro_step=0.1,
        max_energy_error=delta
    )
    
    # 7. Run Sampling
    print(f"Running {draws} Posterior Draws...")
    posterior_trace = walnuts_sample(
        rng_key=key_sample,
        initial_state=warmup_results["final_state"],
        logprob_grad_fn=val_and_grad_fn,
        inverse_mass_matrix=warmup_results["inverse_mass_matrix"],
        optimal_h=warmup_results["optimal_h"],
        delta=delta,
        num_draws=draws
    )
    
    print("Sampling complete. Unmapping traces to PyMC format...")
    
    # 8. Unmap the 1D trace back to the dictionary format
    # We use jax.vmap to instantly unflatten the entire 2D array matrix into a dictionary!
    final_draws_1d = posterior_trace["position"]
    unraveled_trace_dict = jax.vmap(unravel_fn)(final_draws_1d)
    
    # Convert JAX arrays to standard NumPy arrays for the user
    trace_dict = {name: np.array(unraveled_trace_dict[name]) for name in var_names}
            
    return trace_dict