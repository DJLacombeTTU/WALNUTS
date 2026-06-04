import jax
import jax.numpy as jnp
import numpy as np
import time

# Import your WALNUTS modules
from integrators import new_integrator_state
from adaptation import walnuts_warmup
from sample import walnuts_sample

# =====================================================================
# 1. Define the Target Distribution (Pure JAX)
# =====================================================================
def target_logprob(position):
    """
    Target: 2D Highly Correlated Gaussian
    Mean = [0, 0]
    Covariance = [[1.0, 0.9], 
                  [0.9, 1.0]]
    """
    # Inverse Covariance (Precision Matrix) for the math:
    # ~ [[ 5.26, -4.73],
    #    [-4.73,  5.26]]
    precision = jnp.array([[1.0, 0.0], 
                           [0.0, 1.0]])
    
    logp = -0.5 * jnp.dot(position, jnp.dot(precision, position))
    return logp

# Let JAX automatically derive the gradient of your log-probability!
val_and_grad_fn = jax.jit(jax.value_and_grad(target_logprob))

# =====================================================================
# 2. Main Execution Block
# =====================================================================
def run_pure_jax_test():
    print("--- PURE JAX WALNUTS TEST ---")
    print("Target: Highly Correlated 2D Gaussian (rho = 0.9)\n")
    
    # Parameters
    tune_steps = 1000
    draw_steps = 2000
    delta = 0.05
    rng_key = jax.random.PRNGKey(123)
    
    # Initialize far from the mode to test warmup robustness
    q_init = jnp.array([-3.0, 3.0])
    logprob_init, grad_init = val_and_grad_fn(q_init)
    
    init_state = new_integrator_state(
        position=q_init,
        momentum=jnp.zeros_like(q_init),
        logprob=logprob_init,
        logprob_grad=grad_init
    )
    
    key_warmup, key_sample = jax.random.split(rng_key)
    
    # ---------------------------------------------------------
    # Phase 1: WARMUP
    # ---------------------------------------------------------
    print(f"1. Compiling and running {tune_steps} Warmup steps...")
    t0 = time.time()
    
    warmup_results = walnuts_warmup(
        rng_key=key_warmup,
        initial_state=init_state,
        logprob_grad_fn=val_and_grad_fn,
        num_warmup_steps=tune_steps,
        initial_macro_step=0.1,
        max_energy_error=delta
    )
    
    # Block until GPU finishes
    optimal_h = warmup_results["optimal_h"]
    inv_mass = warmup_results["inverse_mass_matrix"]
    optimal_h.block_until_ready()
    t1 = time.time()
    
    print(f"   Warmup Time: {t1 - t0:.4f} seconds")
    print(f"   Tuned Macro-Step (h): {optimal_h:.4f}")
    print(f"   Learned Inverse Mass (Diagonal): {inv_mass}\n")
    
    # ---------------------------------------------------------
    # Phase 2: SAMPLING
    # ---------------------------------------------------------
    print(f"2. Compiling and running {draw_steps} Sampling steps...")
    t2 = time.time()
    
    posterior_trace = walnuts_sample(
        rng_key=key_sample,
        initial_state=warmup_results["final_state"],
        logprob_grad_fn=val_and_grad_fn,
        inverse_mass_matrix=inv_mass,
        optimal_h=optimal_h,
        delta=delta,
        num_draws=draw_steps
    )
    
    # Extract positions and block until GPU finishes
    samples = posterior_trace["position"]
    samples.block_until_ready()
    t3 = time.time()
    
    print(f"   Sampling Time: {t3 - t2:.4f} seconds")
    
    # Calculate draws per second
    dps = draw_steps / (t3 - t2)
    print(f"   Throughput: {dps:.0f} draws/second\n")
    
    # ---------------------------------------------------------
    # Phase 3: EMPIRICAL VERIFICATION
    # ---------------------------------------------------------
    print("3. Empirical Results (Recovering the Posterior)")
    
    # Convert to standard NumPy for analysis
    samples_np = np.array(samples)
    
    emp_mean = np.mean(samples_np, axis=0)
    emp_cov = np.cov(samples_np, rowvar=False)
    
    print(f"   True Mean: [0.0, 0.0]")
    print(f"   Est. Mean: [{emp_mean[0]:.4f}, {emp_mean[1]:.4f}]\n")
    
    print("   True Covariance:\n   [[1.0, 0.9]\n    [0.9, 1.0]]")
    print(f"   Est. Covariance:\n   [[{emp_cov[0,0]:.4f}, {emp_cov[0,1]:.4f}]\n    [{emp_cov[1,0]:.4f}, {emp_cov[1,1]:.4f}]]")

if __name__ == "__main__":
    run_pure_jax_test()