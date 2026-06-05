import jax
import jax.numpy as jnp
from kernel import walnuts_step
from jax import tree_util
# Ensure you import walnuts_step from kernel.py

# =========================================================
# Dual Averaging (Step Size Tuning)
# =========================================================
def init_dual_averaging(initial_macro_step):
    return {
        "log_step_size": jnp.log(initial_macro_step),
        "log_step_size_avg": jnp.log(initial_macro_step),
        "iteration": 1,
        "error_sum": 0.0,
        "mu": jnp.log(10 * initial_macro_step),
    }

def update_dual_averaging(state, unhalved_fraction, target_gamma=0.8):
    """Targets an 80% probability that no micro-halving is needed."""
    error = target_gamma - unhalved_fraction
    iteration = state["iteration"]
    error_sum = state["error_sum"] + error
    
    # Primal-Dual updates
    log_step_size = state["mu"] - (jnp.sqrt(iteration) / 0.05) * (error_sum / (iteration + 10.0))
    eta = iteration ** -0.75
    log_step_size_avg = eta * log_step_size + (1 - eta) * state["log_step_size_avg"]
    
    return {
        "log_step_size": log_step_size,
        "log_step_size_avg": log_step_size_avg,
        "iteration": iteration + 1,
        "error_sum": error_sum,
        "mu": state["mu"],
    }

# =========================================================
# Welford Algorithm (Mass Matrix Tuning)
# =========================================================
def init_dense_welford(num_dimensions):
    return {
        "count": 0,
        "mean": jnp.zeros(num_dimensions),
        "m2": jnp.zeros((num_dimensions, num_dimensions)), # Now a 2D Covariance Matrix
    }

def update_dense_welford(state, position):
    count = state["count"] + 1
    delta = position - state["mean"]
    mean = state["mean"] + delta / count
    delta2 = position - mean
    
    # Outer product captures the cross-correlations
    m2 = state["m2"] + jnp.outer(delta, delta2)
    
    return {"count": count, "mean": mean, "m2": m2}

def get_dense_inverse_mass_matrix(welford_state, regularization=1e-3):
    variance = welford_state["m2"] / jnp.maximum(welford_state["count"] - 1.0, 1.0)
    
    # Add small regularization to the diagonal to ensure positive-definiteness
    reg_matrix = jnp.eye(variance.shape[0]) * regularization
    return variance + reg_matrix

# =========================================================
# The JIT-Compiled Warmup Loop
# =========================================================
@jax.jit(static_argnames=['logprob_grad_fn', 'num_warmup_steps'])
def walnuts_warmup(
    rng_key, 
    initial_state, 
    logprob_grad_fn, 
    num_warmup_steps=1000,
    initial_macro_step=0.1,
    max_energy_error=0.05
):
    num_dimensions = initial_state["position"].shape[0]
    
    da_state = init_dual_averaging(initial_macro_step)
    welford_state = init_dense_welford(num_dimensions)
    inv_mass_matrix = jnp.eye(num_dimensions)
    
    # Stan-style expanding windows for iterative preconditioning
    window_ends = jnp.array([100, 150, 250, 450, 850])
    
    def scan_body(carry, step_idx):
        state, da_state, welford_state, current_inv_mass, key = carry
        step_key, key = jax.random.split(key)
        
        current_h = jnp.exp(da_state["log_step_size"])
        
        next_state, info = walnuts_step(
            rng_key=step_key,
            current_state=state,
            logprob_grad_fn=logprob_grad_fn,
            inverse_mass_matrix=current_inv_mass,
            macro_step_size=current_h,
            max_energy_error=max_energy_error
        )
        
        da_state = update_dual_averaging(da_state, info["unhalved_fraction"], target_gamma=0.8)
        
        # Welford accumulates only during the slow window phase
        is_slow_window = (step_idx >= 75) & (step_idx < 850)
        welford_state = tree_util.tree_map(
            lambda old, new: jnp.where(is_slow_window, new, old),
            welford_state,
            update_dense_welford(welford_state, next_state["position"])
        )
        
        # Update mass matrix iteratively to prevent locking in random-walk variance
        is_update_step = jnp.any(step_idx == window_ends)
        current_inv_mass = jnp.where(is_update_step, get_dense_inverse_mass_matrix(welford_state), current_inv_mass)
        
        # Flush the Welford state so the next window learns from the improved geometry
        welford_state = tree_util.tree_map(
            lambda x: jnp.where(is_update_step, 0.0, x), 
            welford_state
        )
        
        trace = {
            "position": next_state["position"],
            "macro_step_size": current_h,
            "unhalved_fraction": info["unhalved_fraction"],
            "tree_depth": info["tree_depth"]
        }
        
        return (next_state, da_state, welford_state, current_inv_mass, key), trace

    init_carry = (initial_state, da_state, welford_state, inv_mass_matrix, rng_key)
    final_carry, warmup_trace = jax.lax.scan(scan_body, init_carry, jnp.arange(num_warmup_steps))
    
    final_state, final_da_state, _, final_inv_mass, _ = final_carry
    
    result = {
        "final_state": final_state,
        "optimal_h": jnp.exp(final_da_state["log_step_size_avg"]),
        "inverse_mass_matrix": final_inv_mass,
        "trace": warmup_trace
    }
    
    return result