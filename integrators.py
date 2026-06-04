import jax
import jax.numpy as jnp

def new_integrator_state(position, momentum, logprob, logprob_grad):
    """Holds the state of the integrator at a given point."""
    return {
        "position": position,
        "momentum": momentum,
        "logprob": logprob,
        "logprob_grad": logprob_grad,
    }

def dense_leapfrog_step(state, step_size, logprob_grad_fn, inverse_mass_matrix):
    position = state["position"]
    momentum = state["momentum"]
    
    momentum_half = momentum + 0.5 * step_size * state["logprob_grad"]
    
    # Use jnp.dot to multiply the 2D inverse mass matrix by the 1D momentum
    position_next = position + step_size * jnp.dot(inverse_mass_matrix, momentum_half)
    
    logprob_next, grad_next = logprob_grad_fn(position_next)
    momentum_next = momentum_half + 0.5 * step_size * grad_next
    
    return new_integrator_state(position_next, momentum_next, logprob_next, grad_next)

def compute_dense_hamiltonian(state, inverse_mass_matrix):
    # K = 0.5 * p^T * M^-1 * p
    M_inv_p = jnp.dot(inverse_mass_matrix, state["momentum"])
    kinetic_energy = 0.5 * jnp.dot(state["momentum"], M_inv_p)
    
    potential_energy = -state["logprob"]
    return potential_energy + kinetic_energy

@jax.jit(static_argnames=['logprob_grad_fn', 'max_halvings'])
def micro_routine(initial_state, logprob_grad_fn, inverse_mass_matrix, macro_step_size, max_energy_error, max_halvings=10):
    """
    Finds the smallest power-of-two (l = 2^i) such that 'l' micro-steps 
    keep the energy error below delta.
    """
    initial_energy = compute_hamiltonian(initial_state, inverse_mass_matrix)
    
    def condition_fn(loop_state):
        i, max_error, _, _ = loop_state
        # Continue if error is too large AND we haven't hit the halving limit
        return (max_error > max_energy_error) & (i < max_halvings)

    def body_fn(loop_state):
        i, _, _, _ = loop_state
        
        # Calculate dyadic schedule for this iteration
        l = 2 ** i
        micro_step_size = macro_step_size / l
        
        def micro_step_fn(step_idx, inner_state):
            current_int_state, current_max_h, current_min_h = inner_state
            
            next_int_state = leapfrog_step(
                current_int_state, micro_step_size, logprob_grad_fn, inverse_mass_matrix
            )
            
            current_energy = compute_hamiltonian(next_int_state, inverse_mass_matrix)
            
            new_max_h = jnp.maximum(current_max_h, current_energy)
            new_min_h = jnp.minimum(current_min_h, current_energy)
            
            return next_int_state, new_max_h, new_min_h

        # Run the micro-trajectory
        final_int_state, max_h, min_h = jax.lax.fori_loop(
            0, l, micro_step_fn, (initial_state, initial_energy, initial_energy)
        )
        
        current_max_error = max_h - min_h
        
        return i + 1, current_max_error, l, final_int_state

    # Initialize loop: i=0 (l=1), max_error=infinity (to force first run)
    init_loop_state = (0, jnp.inf, 1, initial_state)
    
    final_i, final_error, optimal_l, final_state = jax.lax.while_loop(
        condition_fn, body_fn, init_loop_state
    )
    
    return optimal_l, final_state, final_error