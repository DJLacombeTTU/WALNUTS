import sys
import os

# Dynamically route Python to the root WALNUTS directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import jax
import jax.numpy as jnp
import numpy as np

# Import the functional kernel and the adaptation modules
from blackjax.mcmc.walnuts import init, build_kernel
from blackjax.adaptation.dense_window_adaptation import init_da, update_da

def non_centered_funnel_logpdf(q):
    """
    FULLY Non-Centered 10-Dimensional Neal's Funnel.
    y_raw and x_raw are both evaluated as N(0,1).
    The Identity Mass Matrix is now mathematically perfect for this isotropic space.
    """
    y_raw = q[0]
    x_raw = q[1:]
    
    # PERFECT ISOTROPIC BOWL
    log_p_y_raw = -0.5 * jnp.square(y_raw)
    log_p_x_raw = jnp.sum(-0.5 * jnp.square(x_raw))
    
    return log_p_y_raw + log_p_x_raw

def run_adapted_vmap_funnel():
    print("Initializing Vectorized Fully Non-Centered Funnel Test...")
    dim = 10
    tune = 3000
    draws = 5000
    chains = 4
    
    rng_key = jax.random.PRNGKey(42)
    key_jitter, key_warmup, key_sample = jax.random.split(rng_key, 3)
    
    # 1. Jitter Initialization across 4 chains
    base_position = jnp.ones(dim)
    q_inits = base_position + jax.random.normal(key_jitter, (chains, dim)) * 0.1
    inverse_mass_matrix = jnp.eye(dim)
    
    # Vectorize the initialization
    init_states = jax.vmap(lambda q: init(q, non_centered_funnel_logpdf))(q_inits)
    
    # ---------------------------------------------------------
    # 2. VECTORIZED WARMUP
    # ---------------------------------------------------------
    def single_chain_warmup(init_state, key):
        da_state = init_da(init_h=0.1)
        
        def warmup_step(carry, k):
            state, da_s = carry
            curr_h = jnp.exp(da_s.log_h)
            kernel = build_kernel(non_centered_funnel_logpdf, inverse_mass_matrix, curr_h)
            next_state, info = kernel(k, state)
            
            # The engine now handles step size bounding internally.
            # target_accept=0.90 is passed to enforce the hierarchical standard.
            next_da_s = update_da(da_s, info.unhalved_fraction, target_accept=0.90)
            
            return (next_state, next_da_s), None

        keys = jax.random.split(key, tune)
        (final_state, final_da_state), _ = jax.lax.scan(warmup_step, (init_state, da_state), keys)
        return final_state, jnp.exp(final_da_state.log_h_avg)

    print(f"Executing {tune} warmup steps across {chains} vectorized chains...")
    keys_w = jax.random.split(key_warmup, chains)
    warmup_states, optimal_hs = jax.vmap(single_chain_warmup)(init_states, keys_w)
    
    print(f"Optimal macro step sizes discovered: {np.array(optimal_hs)}")

    # ---------------------------------------------------------
    # 3. VECTORIZED SAMPLING
    # ---------------------------------------------------------
    def single_chain_sample(state, optimal_h, key):
        sampling_kernel = build_kernel(non_centered_funnel_logpdf, inverse_mass_matrix, optimal_h)
        
        def sample_step(s, k):
            next_s, info = sampling_kernel(k, s)
            return next_s, (next_s.position, info.tree_depth)

        keys = jax.random.split(key, draws)
        final_state, (positions, tree_depths) = jax.lax.scan(sample_step, state, keys)
        return positions, tree_depths

    print(f"Sampling {draws} draws across {chains} vectorized chains...")
    keys_s = jax.random.split(key_sample, chains)
    positions, tree_depths = jax.vmap(single_chain_sample)(warmup_states, optimal_hs, keys_s)
    
    positions.block_until_ready()
    
    # ---------------------------------------------------------
    # 4. EVALUATION (Pooling the Chains & Deterministic Transformation)
    # ---------------------------------------------------------
    # Flatten the (chains, draws, dim) arrays to (chains * draws, dim)
    pooled_positions = positions.reshape(chains * draws, dim)
    pooled_depths = tree_depths.reshape(chains * draws)
    
    y_raw_samples = pooled_positions[:, 0]
    x_raw_samples = pooled_positions[:, 1:]
    
    # Deterministic reconstruction of both parameter scales
    y_samples = y_raw_samples * 3.0  # Scale y back to N(0, 3^2)
    x_samples = x_raw_samples * jnp.exp(y_samples[:, None] / 2.0)
    x0_samples = x_samples[:, 0]
    
    print("\n--- Fully Non-Centered Geometry Recovery (8,000 Total Draws) ---")
    print(f"Mean of y (True: 0.0): {np.mean(y_samples):.3f}")
    print(f"Variance of y (True: 9.0): {np.var(y_samples):.3f}")
    print(f"Mean of x_0 (True: 0.0): {np.mean(x0_samples):.3f}")
    
    print("\n--- WALNUTS Engine Diagnostics ---")
    print(f"Average Tree Depth: {np.mean(pooled_depths):.2f}")

if __name__ == "__main__":
    run_adapted_vmap_funnel()