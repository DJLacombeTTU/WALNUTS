import jax
import jax.numpy as jnp
from jax import tree_util
from integrators import micro_routine, compute_hamiltonian

def check_u_turn(momentum_left, momentum_right, momentum_sum, inverse_mass_matrix):
    rho_left = jnp.dot(momentum_left, inverse_mass_matrix * momentum_sum)
    rho_right = jnp.dot(momentum_right, inverse_mass_matrix * momentum_sum)
    return (rho_left < 0) | (rho_right < 0)

def extend_trajectory(
    rng_key, initial_state, direction, num_macro_steps, logprob_grad_fn,
    inverse_mass_matrix, macro_step_size, max_energy_error, initial_energy
):
    def loop_condition(loop_state):
        step_idx, _, _, _, _, is_diverging, is_turning, _, _ = loop_state
        # Stop building the sub-tree immediately if a U-turn or divergence occurs
        return (step_idx < num_macro_steps) & jnp.logical_not(is_diverging)

    def step_body(loop_state):
        (step_idx, current_state, proposed_state, current_log_weight, 
         momentum_sum, is_diverging, is_turning, unhalved_count, key) = loop_state
        
        step_l, next_state, actual_error = micro_routine(
            initial_state=current_state,
            logprob_grad_fn=logprob_grad_fn,
            inverse_mass_matrix=inverse_mass_matrix,
            macro_step_size=macro_step_size * direction,
            max_energy_error=max_energy_error
        )
        
        diverged = actual_error > max_energy_error + 1000.0
        new_is_diverging = jnp.logical_or(is_diverging, diverged)
        
        was_unhalved = jnp.where(step_l == 1, 1.0, 0.0)
        new_unhalved_count = unhalved_count + was_unhalved
        
        next_energy = compute_hamiltonian(next_state, inverse_mass_matrix)
        
        # --- FIX 1: Pure Log-Space Weighting ---
        log_state_weight = initial_energy - next_energy
        new_log_weight = jnp.logaddexp(current_log_weight, log_state_weight)
        accept_prob = jnp.exp(log_state_weight - new_log_weight)
        
        key, subkey = jax.random.split(key)
        keep_new_state = jax.random.uniform(subkey) < accept_prob
        
        new_proposed_state = tree_util.tree_map(
            lambda old, new: jnp.where(keep_new_state, new, old), 
            proposed_state, next_state
        )
        
        new_momentum_sum = momentum_sum + next_state["momentum"]
        
        # --- FIX 2: Sub-tree U-Turn Detection ---
        new_is_turning = check_u_turn(initial_state["momentum"], next_state["momentum"], new_momentum_sum, inverse_mass_matrix)
        
        return (step_idx + 1, next_state, new_proposed_state, new_log_weight, 
                new_momentum_sum, new_is_diverging, new_is_turning, new_unhalved_count, key)

    init_loop_state = (
        0, initial_state, initial_state, -jnp.inf, initial_state["momentum"], 
        False, False, 0.0, rng_key
    )

    final_loop_state = jax.lax.while_loop(loop_condition, step_body, init_loop_state)
    
    # We now unpack 9 variables from the while_loop
    (_, final_boundary_state, final_proposed_state, total_log_weight, 
     total_momentum_sum, diverged, turned, final_unhalved_count, _) = final_loop_state
    
    # Return the 'turned' flag so the global tree knows to stop expanding
    return final_boundary_state, final_proposed_state, total_log_weight, total_momentum_sum, diverged, final_unhalved_count, turned