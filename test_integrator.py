import jax
import jax.numpy as jnp

# =====================================================================
# 1. Core Integrator Components (From Phase 1a)
# =====================================================================

def new_integrator_state(position, momentum, logprob, logprob_grad):
    return {
        "position": position,
        "momentum": momentum,
        "logprob": logprob,
        "logprob_grad": logprob_grad,
    }

def leapfrog_step(state, step_size, logprob_grad_fn, inverse_mass_matrix):
    position = state["position"]
    momentum = state["momentum"]
    
    # Adding the log-probability gradient is equivalent to subtracting the potential gradient
    momentum_half = momentum + 0.5 * step_size * state["logprob_grad"]
    position_next = position + step_size * (inverse_mass_matrix * momentum_half)
    
    logprob_next, grad_next = logprob_grad_fn(position_next)
    momentum_next = momentum_half + 0.5 * step_size * grad_next
    
    return new_integrator_state(position_next, momentum_next, logprob_next, grad_next)

def compute_hamiltonian(state, inverse_mass_matrix):
    kinetic_energy = 0.5 * jnp.dot(state["momentum"], inverse_mass_matrix * state["momentum"])
    # Potential Energy U(q) = -log p(q)
    potential_energy = -state["logprob"]
    return potential_energy + kinetic_energy

@jax.jit(static_argnames=['logprob_grad_fn', 'max_halvings'])
def micro_routine(initial_state, logprob_grad_fn, inverse_mass_matrix, macro_step_size, max_energy_error, max_halvings=10):
    initial_energy = compute_hamiltonian(initial_state, inverse_mass_matrix)
    
    def condition_fn(loop_state):
        i, max_error, _, _ = loop_state
        return (max_error > max_energy_error) & (i < max_halvings)

    def body_fn(loop_state):
        i, _, _, _ = loop_state
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

        final_int_state, max_h, min_h = jax.lax.fori_loop(
            0, l, micro_step_fn, (initial_state, initial_energy, initial_energy)
        )
        current_max_error = max_h - min_h
        
        return i + 1, current_max_error, l, final_int_state

    init_loop_state = (0, jnp.inf, 1, initial_state)
    final_i, final_error, optimal_l, final_state = jax.lax.while_loop(condition_fn, body_fn, init_loop_state)
    
    return optimal_l, final_state, final_error

# =====================================================================
# 2. Dummy Test Environment
# =====================================================================

def standard_gaussian_target(position):
    """
    Dummy target: 2D Standard Normal Distribution.
    log p(q) = -0.5 * q^T q
    d(log p)/dq = -q
    """
    logprob = -0.5 * jnp.dot(position, position)
    grad = -position
    return logprob, grad

def run_test():
    print("Initializing WALNUTS test state...")
    
    q_init = jnp.array([1.0, 1.0])
    p_init = jnp.array([0.5, -0.5])
    inv_mass_matrix = jnp.array([1.0, 1.0]) 
    
    # Get initial Log-Prob and Gradient
    logprob_init, grad_init = standard_gaussian_target(q_init)
    init_state = new_integrator_state(q_init, p_init, logprob_init, grad_init)
    
    macro_step = 1.5      
    delta = 0.05          

    print(f"Macro step size (h): {macro_step}")
    print(f"Energy error threshold (delta): {delta}\n")

    print("Running JAX JIT compilation and executing micro_routine...")
    optimal_l, final_state, actual_error = micro_routine(
        initial_state=init_state,
        logprob_grad_fn=standard_gaussian_target,
        inverse_mass_matrix=inv_mass_matrix,
        macro_step_size=macro_step,
        max_energy_error=delta
    )
    
    print("--- Test Results ---")
    print(f"Optimal step-size reduction factor (l): {optimal_l}")
    print(f"Number of micro-steps executed: {optimal_l}")
    print(f"Actual micro-step size used: {macro_step / optimal_l:.4f}")
    print(f"Final Max Energy Error: {actual_error:.6f} (Must be <= {delta})")
    print(f"Final Position: {final_state['position']}")
    print(f"Final Momentum: {final_state['momentum']}")

if __name__ == "__main__":
    run_test()