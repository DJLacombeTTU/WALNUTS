import jax
import jax.numpy as jnp
from jax import tree_util
from integrators import new_integrator_state, compute_hamiltonian
from trajectory import extend_trajectory, check_u_turn

def sample_momentum(rng_key, inverse_mass_matrix):
    """
    Samples p ~ N(0, M). 
    Since we have M^-1, we find L = cholesky(M^-1) and solve L^T * p = z
    """
    z = jax.random.normal(rng_key, shape=(inverse_mass_matrix.shape[0],))
    
    # Lower Cholesky decomposition of the inverse mass matrix
    L = jax.scipy.linalg.cholesky(inverse_mass_matrix, lower=True)
    
    # Solve for p using JAX's optimized triangular solver
    momentum = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
    
    return momentum

@jax.jit(static_argnames=['logprob_grad_fn', 'max_tree_depth'])
def walnuts_step(
    rng_key, current_state, logprob_grad_fn, inverse_mass_matrix,
    macro_step_size, max_energy_error, max_tree_depth=10
):
    key_momentum, key_tree = jax.random.split(rng_key)
    initial_momentum = sample_momentum(key_momentum, inverse_mass_matrix)
    
    state = new_integrator_state(
        current_state["position"], initial_momentum,
        current_state["logprob"], current_state["logprob_grad"]
    )
    
    initial_energy = compute_hamiltonian(state, inverse_mass_matrix)
    
    def tree_condition(loop_state):
        depth, _, _, _, _, _, is_turning, is_diverging, _, _ = loop_state
        return (depth < max_tree_depth) & jnp.logical_not(is_turning) & jnp.logical_not(is_diverging)

    def tree_body(loop_state):
        (depth, left_state, right_state, proposed_state, 
         log_total_weight, momentum_sum, is_turning, is_diverging, 
         unhalved_count, key) = loop_state
        
        key, direction_key, accept_key = jax.random.split(key, 3)
        direction = jnp.where(jax.random.uniform(direction_key) < 0.5, -1, 1)
        L = 2 ** depth
        
        def build_left(_):
            return extend_trajectory(
                accept_key, left_state, -1, L, logprob_grad_fn, 
                inverse_mass_matrix, macro_step_size, max_energy_error, initial_energy
            )
            
        def build_right(_):
            return extend_trajectory(
                accept_key, right_state, 1, L, logprob_grad_fn, 
                inverse_mass_matrix, macro_step_size, max_energy_error, initial_energy
            )
            
        # Catch the new 7th variable: subtree_turned
        (boundary_state, new_proposed_state, log_subtree_weight, 
         subtree_momentum, diverged, subtree_unhalved, subtree_turned) = jax.lax.cond(
            direction == -1, build_left, build_right, operand=None
        )
        
        new_left_state = tree_util.tree_map(lambda old, new: jnp.where(direction == -1, new, old), left_state, boundary_state)
        new_right_state = tree_util.tree_map(lambda old, new: jnp.where(direction == 1, new, old), right_state, boundary_state)
        
        new_log_total_weight = jnp.logaddexp(log_total_weight, log_subtree_weight)
        accept_prob = jnp.exp(log_subtree_weight - new_log_total_weight)
        
        keep_new_proposal = jax.random.uniform(accept_key) < accept_prob
        final_proposed_state = tree_util.tree_map(
            lambda old, new: jnp.where(keep_new_proposal, new, old), 
            proposed_state, new_proposed_state
        )
        
        new_momentum_sum = momentum_sum + subtree_momentum
        
        # --- FIX 3: Global U-Turn evaluates Sub-Tree U-Turns ---
        turning = check_u_turn(new_left_state["momentum"], new_right_state["momentum"], new_momentum_sum, inverse_mass_matrix)
        turning = turning | subtree_turned
        
        diverging = jnp.logical_or(is_diverging, diverged)
        new_unhalved_count = unhalved_count + subtree_unhalved
        
        return (depth + 1, new_left_state, new_right_state, final_proposed_state, 
                new_log_total_weight, new_momentum_sum, turning, diverging, 
                new_unhalved_count, key)

    init_loop_state = (0, state, state, state, 0.0, state["momentum"], False, False, 0.0, key_tree)
    final_loop_state = jax.lax.while_loop(tree_condition, tree_body, init_loop_state)
    (final_depth, _, _, final_state, _, _, is_turning, is_diverging, final_unhalved_count, _) = final_loop_state
    
    output_state = tree_util.tree_map(lambda old, new: jnp.where(is_diverging, old, new), state, final_state)
    total_macro_steps = (2 ** final_depth) - 1.0
    unhalved_fraction = jnp.where(total_macro_steps > 0, final_unhalved_count / total_macro_steps, 0.0)
    
    info = {
        "tree_depth": final_depth,
        "diverging": is_diverging,
        "turning": is_turning,
        "unhalved_fraction": unhalved_fraction
    }
    
    return output_state, info