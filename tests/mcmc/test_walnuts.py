import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp

from blackjax.mcmc.walnuts import walnuts

def test_walnuts():
    # 1. Target Distribution
    # 2D standard Normal log-density function
    def logdensity_fn(x):
        return -0.5 * jnp.sum(x ** 2)

    # 2. Initialization Test
    position = jnp.array([1.0, -1.0])
    sampler = walnuts(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=jnp.eye(2),
        step_size=0.1
    )
    initial_state = sampler.init(position)

    # Verify state shapes and finite logdensity
    assert initial_state.position.shape == (2,), f"Expected shape (2,), got {initial_state.position.shape}"
    assert initial_state.logdensity_grad.shape == (2,), f"Expected shape (2,), got {initial_state.logdensity_grad.shape}"
    assert jnp.isfinite(initial_state.logdensity), "Log-density is not finite"

    # 4. Transition Test
    # Execute a single step with a PRNG key
    rng_key = jax.random.PRNGKey(42)
    next_state, info = sampler.step(rng_key, initial_state)

    # 5. Output Validation
    # Next state should maintain tensor shapes
    assert next_state.position.shape == (2,), f"Expected next position shape (2,), got {next_state.position.shape}"
    assert next_state.logdensity_grad.shape == (2,), f"Expected next grad shape (2,), got {next_state.logdensity_grad.shape}"
    assert jnp.isfinite(next_state.logdensity), "Next log-density is not finite"

    # Validate diagnostic information exists
    assert hasattr(info, 'tree_depth'), "Info is missing 'tree_depth'"
    assert hasattr(info, 'unhalved_fraction'), "Info is missing 'unhalved_fraction'"
    
    print("WALNUTS unit tests completed successfully.")

if __name__ == "__main__":
    test_walnuts()
