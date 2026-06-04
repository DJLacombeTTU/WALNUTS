import jax
import jax.numpy as jnp
import time

# Import the modules built in Phases 1 & 2
from integrators import new_integrator_state
from adaptation import walnuts_warmup

def standard_gaussian_target(position):
    """
    Dummy target: 2D Standard Normal Distribution.
    log p(q) = -0.5 * q^T q
    d(log p)/dq = -q
    """
    logprob = -0.5 * jnp.dot(position, position)
    grad = -position
    return logprob, grad

def run_warmup_test():
    print("Initializing full WALNUTS Warmup...")
    
    # 1. Setup initial conditions
    # We intentionally start far from the mode (0,0) to force the algorithm
    # to navigate back to the typical set while simultaneously adapting.
    q_init = jnp.array([5.0, -5.0])
    p_init = jnp.array([0.0, 0.0])
    
    logprob_init, grad_init = standard_gaussian_target(q_init)
    init_state = new_integrator_state(q_init, p_init, logprob_init, grad_init)
    
    # 2. Warmup Parameters
    num_warmup_steps = 1000
    initial_macro_step = 0.5  # Dual Averaging will override this quickly
    initial_delta = 0.05      # Strict energy error threshold
    
    rng_key = jax.random.PRNGKey(42)

    print(f"Executing {num_warmup_steps} MCMC warmup steps.")
    print("Compiling XLA graph (this takes a moment)...")
    
    start_time = time.time()
    
    # 3. Run the JIT-compiled warmup loop
    result = walnuts_warmup(
        rng_key=rng_key,
        initial_state=init_state,
        logprob_grad_fn=standard_gaussian_target,
        num_warmup_steps=num_warmup_steps,
        initial_macro_step=initial_macro_step,
        max_energy_error=initial_delta
    )
    
    # Block until JAX finishes executing on the hardware
    result["optimal_h"].block_until_ready()
    end_time = time.time()
    
    print(f"\nWarmup completed in {end_time - start_time:.4f} seconds!")
    print("-" * 50)
    print("### WARMUP RESULTS ###")
    
    # The optimal macro-step size h found by Dual Averaging
    print(f"Optimal Macro Step Size (h): {result['optimal_h']:.4f}")
    
    # The variance of the posterior learned by Welford's algorithm
    print(f"Learned Inverse Mass Matrix: {result['inverse_mass_matrix']}")
    
    # Verify the chain moved from [5.0, -5.0] into the typical set (near 0)
    print(f"Final Position (Draw 1000):  {result['final_state']['position']}")
    
    # 4. Trace Analysis
    trace = result["trace"]
    
    # The fraction of steps that required no micro-halving (Target is 0.8)
    avg_unhalved = jnp.mean(trace["unhalved_fraction"])
    
    # Print a snapshot of how Dual Averaging changed the step size
    print("\n### DUAL AVERAGING TRACE ###")
    print(f"Target Gamma (no-halving rate): 0.80")
    print(f"Achieved Average Gamma:       {avg_unhalved:.4f}")
    
    print("\nMacro Step Size (h) evolution:")
    print(f"  Step 100:  {trace['macro_step_size'][100]:.4f}")
    print(f"  Step 500:  {trace['macro_step_size'][500]:.4f}")
    print(f"  Step 999:  {trace['macro_step_size'][999]:.4f}")

if __name__ == "__main__":
    run_warmup_test()