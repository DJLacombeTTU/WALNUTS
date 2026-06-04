import jax
import jax.numpy as jnp

from blackjax.mcmc.walnuts import init, build_kernel

def test_walnuts():
    # 1. Target Distribution
    # 2D standard Normal log-density function
    def logdensity_fn(x):
        return -0.5 * jnp.sum(x ** 2)

    # 2. Initialization Test
    position = jnp.array([1.0, -1.0])
    initial_state = init(position, logdensity_fn)

    # Verify state shapes and finite logdensity
    assert initial_state.position.shape == (2,), f"Expected shape (2,), got {initial_state.position.shape}"
    assert initial_state.logdensity_grad.shape == (2,), f"Expected shape (2,), got {initial_state.logdensity_grad.shape}"
    assert jnp.isfinite(initial_state.logdensity), "Log-density is not finite"

    # 3. Kernel Test
    # Instantiate the transition kernel with identity inverse mass matrix and static step size 0.1
    inverse_mass_matrix = jnp.eye(2)
    step_size = 0.1
    kernel = build_kernel(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=step_size
    )

    # 4. Transition Test
    # Execute a single step with a PRNG key
    rng_key = jax.random.PRNGKey(42)
    next_state, info = kernel(rng_key, initial_state)

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
