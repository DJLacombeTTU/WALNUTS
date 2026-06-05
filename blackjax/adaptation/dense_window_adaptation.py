from typing import NamedTuple
import jax
import jax.numpy as jnp

class DualAveragingState(NamedTuple):
    log_step_size: jax.Array | float
    log_step_size_avg: jax.Array | float
    iteration: jax.Array | int
    error_sum: jax.Array | float
    mu: jax.Array | float

def init_dual_averaging(initial_step_size: float) -> DualAveragingState:
    return DualAveragingState(
        log_step_size=jnp.log(initial_step_size),
        log_step_size_avg=jnp.log(initial_step_size),
        iteration=1,
        error_sum=0.0,
        mu=jnp.log(10 * initial_step_size)
    )

def update_dual_averaging(
    state: DualAveragingState, 
    unhalved_fraction: float, 
    target_accept: float = 0.90,  # <-- Hierarchical Standard Default
    min_step_size: float = 1e-4,  # <-- Lower bound (prevents freezing)
    max_step_size: float = 1.0    # <-- Upper bound (prevents leapfrog instability)
) -> DualAveragingState:
    """
    Updates the Dual Averaging state while strictly enforcing 
    physical leapfrog stability bounds.
    """
    error = target_accept - unhalved_fraction
    iteration = state.iteration
    error_sum = state.error_sum + error
    
    # Calculate unbounded step size
    log_step_size_unbounded = state.mu - (jnp.sqrt(iteration) / 0.05) * (error_sum / (iteration + 10.0))
    
    # Calculate unbounded running average
    eta = iteration ** -0.75
    log_step_size_avg_unbounded = eta * log_step_size_unbounded + (1 - eta) * state.log_step_size_avg
    
    # THE PHYSICS BOUNDS: Enforce stability at the engine level
    log_step_size_clipped = jnp.clip(log_step_size_unbounded, jnp.log(min_step_size), jnp.log(max_step_size))
    log_step_size_avg_clipped = jnp.clip(log_step_size_avg_unbounded, jnp.log(min_step_size), jnp.log(max_step_size))
    
    return DualAveragingState(
        log_step_size=log_step_size_clipped,
        log_step_size_avg=log_step_size_avg_clipped,
        iteration=iteration + 1,
        error_sum=error_sum,
        mu=state.mu
    )

class DenseWelfordState(NamedTuple):
    count: jax.Array | int
    mean: jax.Array
    m2: jax.Array

def init_dense_welford(dim: int) -> DenseWelfordState:
    return DenseWelfordState(
        count=0,
        mean=jnp.zeros(dim),
        m2=jnp.zeros((dim, dim))
    )

def update_dense_welford(state: DenseWelfordState, position: jax.Array) -> DenseWelfordState:
    count = state.count + 1
    delta = position - state.mean
    mean = state.mean + delta / count
    # Outer product for off-diagonal covariance tracking
    m2 = state.m2 + jnp.outer(delta, position - mean)
    return DenseWelfordState(count=count, mean=mean, m2=m2)

def get_dense_inverse_mass_matrix(state: DenseWelfordState) -> jax.Array:
    variance = state.m2 / jnp.maximum(state.count - 1.0, 1.0)
    return variance + jnp.eye(variance.shape[0]) * 1e-3
