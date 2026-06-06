import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp

from blackjax.mcmc.walnuts import walnuts


def test_walnuts_benchmark_funnel():
    """
    Lean, CI-friendly benchmark test for WALNUTS using Neal's Funnel.
    Scaled down to N = 5 dimensions to prevent choking low-power runners.
    """
    dim = 5

    def funnel_logdensity(x):
        # Neal's Funnel
        v = x[0]
        log_density_v = -0.5 * (v**2) / 9.0
        log_density_x = -0.5 * jnp.sum((x[1:] ** 2) / jnp.exp(v))
        return log_density_v + log_density_x

    inverse_mass_matrix = jnp.eye(dim)
    step_size = 0.1

    sampler = walnuts(
        logdensity_fn=funnel_logdensity,
        inverse_mass_matrix=inverse_mass_matrix,
        step_size=step_size,
        max_energy_error=1000.0,
        max_tree_depth=5,
        max_halvings=5,
    )

    initial_position = jnp.ones(dim) * 0.1
    initial_state = sampler.init(initial_position)

    # Tiny counts for warmup and production to keep runtime minimal
    warmup_steps = 10
    production_steps = 5

    rng_key = jax.random.PRNGKey(0)

    def step_fn(state, key):
        next_state, info = sampler.step(key, state)
        return next_state, info

    # Warmup
    keys_warmup = jax.random.split(rng_key, warmup_steps)
    final_warmup_state, _ = jax.lax.scan(step_fn, initial_state, keys_warmup)

    # Production
    keys_prod = jax.random.split(keys_warmup[-1], production_steps)
    final_prod_state, infos = jax.lax.scan(
        step_fn, final_warmup_state, keys_prod
    )

    assert final_prod_state.position.shape == (dim,)
    assert jnp.isfinite(final_prod_state.logdensity)

    print("WALNUTS benchmark (Funnel) completed successfully.")


if __name__ == "__main__":
    test_walnuts_benchmark_funnel()
