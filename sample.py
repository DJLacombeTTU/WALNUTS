import jax
import jax.numpy as jnp
from kernel import walnuts_step

@jax.jit(static_argnames=['logprob_grad_fn', 'num_draws'])
def walnuts_sample(
    rng_key,
    initial_state,
    logprob_grad_fn,
    inverse_mass_matrix,
    optimal_h,
    delta,
    num_draws=1000
):
    """Executes the final posterior sampling phase."""
    
    def scan_body(state, key):
        step_key, next_key = jax.random.split(key)
        
        next_state, info = walnuts_step(
            rng_key=step_key,
            current_state=state,
            logprob_grad_fn=logprob_grad_fn,
            inverse_mass_matrix=inverse_mass_matrix,
            macro_step_size=optimal_h,
            max_energy_error=delta
        )
        
        # We save the position and the tree depth for diagnostics
        trace = {
            "position": next_state["position"],
            "tree_depth": info["tree_depth"],
            "diverging": info["diverging"]
        }
        
        return next_state, trace

    keys = jax.random.split(rng_key, num_draws)
    final_state, posterior_trace = jax.lax.scan(scan_body, initial_state, keys)
    
    return posterior_trace