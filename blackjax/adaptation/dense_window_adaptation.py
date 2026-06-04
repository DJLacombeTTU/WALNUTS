from typing import NamedTuple
import jax
import jax.numpy as jnp

class DualAveragingState(NamedTuple):
    log_h: float
    log_h_avg: float
    iter: int
    err_sum: float
    mu: float

def init_da(init_h: float) -> DualAveragingState:
    return DualAveragingState(
        log_h=jnp.log(init_h),
        log_h_avg=jnp.log(init_h),
        iter=1,
        err_sum=0.0,
        mu=jnp.log(10 * init_h)
    )

def update_da(
    state: DualAveragingState, 
    unh_frac: float, 
    target_accept: float = 0.90,  # <-- Hierarchical Standard Default
    min_h: float = 1e-4,          # <-- Lower bound (prevents freezing)
    max_h: float = 1.0            # <-- Upper bound (prevents leapfrog instability)
) -> DualAveragingState:
    """
    Updates the Dual Averaging state while strictly enforcing 
    physical leapfrog stability bounds.
    """
    err = target_accept - unh_frac
    it = state.iter
    err_sum = state.err_sum + err
    
    # Calculate unbounded step size
    log_h_unbounded = state.mu - (jnp.sqrt(it) / 0.05) * (err_sum / (it + 10.0))
    
    # Calculate unbounded running average
    eta = it ** -0.75
    log_h_avg_unbounded = eta * log_h_unbounded + (1 - eta) * state.log_h_avg
    
    # THE PHYSICS BOUNDS: Enforce stability at the engine level
    log_h_clipped = jnp.clip(log_h_unbounded, jnp.log(min_h), jnp.log(max_h))
    log_h_avg_clipped = jnp.clip(log_h_avg_unbounded, jnp.log(min_h), jnp.log(max_h))
    
    return DualAveragingState(
        log_h=log_h_clipped,
        log_h_avg=log_h_avg_clipped,
        iter=it + 1,
        err_sum=err_sum,
        mu=state.mu
    )