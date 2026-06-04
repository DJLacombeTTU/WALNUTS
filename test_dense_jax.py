import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np
import time

# ==========================================
# 1. DENSE INTEGRATORS & PHYSICS
# ==========================================
def new_integrator_state(position, momentum, logprob, logprob_grad):
    return {"position": position, "momentum": momentum, "logprob": logprob, "logprob_grad": logprob_grad}

def dense_leapfrog(state, step_size, logprob_grad_fn, inv_mass_matrix):
    momentum_half = state["momentum"] + 0.5 * step_size * state["logprob_grad"]
    # Rotate the position proposal using the dense inverse mass matrix
    position_next = state["position"] + step_size * jnp.dot(inv_mass_matrix, momentum_half)
    logprob_next, grad_next = logprob_grad_fn(position_next)
    momentum_next = momentum_half + 0.5 * step_size * grad_next
    return new_integrator_state(position_next, momentum_next, logprob_next, grad_next)

def compute_dense_hamiltonian(state, inv_mass_matrix):
    # K = 0.5 * p^T * M^-1 * p
    kinetic_energy = 0.5 * jnp.dot(state["momentum"], jnp.dot(inv_mass_matrix, state["momentum"]))
    return -state["logprob"] + kinetic_energy

@jax.jit(static_argnames=['logprob_grad_fn', 'max_halvings'])
def dense_micro_routine(initial_state, logprob_grad_fn, inv_mass_matrix, macro_step_size, max_energy_error, max_halvings=10):
    initial_energy = compute_dense_hamiltonian(initial_state, inv_mass_matrix)
    
    def condition_fn(loop_state):
        i, max_error, _, _ = loop_state
        return (max_error > max_energy_error) & (i < max_halvings)
        
    def body_fn(loop_state):
        i, _, _, _ = loop_state
        l = 2 ** i
        micro_step_size = macro_step_size / l
        
        def micro_step_fn(step_idx, inner_state):
            curr_state, curr_max_h, curr_min_h = inner_state
            next_state = dense_leapfrog(curr_state, micro_step_size, logprob_grad_fn, inv_mass_matrix)
            curr_energy = compute_dense_hamiltonian(next_state, inv_mass_matrix)
            return next_state, jnp.maximum(curr_max_h, curr_energy), jnp.minimum(curr_min_h, curr_energy)
            
        final_state, max_h, min_h = jax.lax.fori_loop(0, l, micro_step_fn, (initial_state, initial_energy, initial_energy))
        return i + 1, max_h - min_h, l, final_state
        
    init_loop_state = (0, jnp.inf, 1, initial_state)
    _, final_error, optimal_l, final_state = jax.lax.while_loop(condition_fn, body_fn, init_loop_state)
    return optimal_l, final_state, final_error

# ==========================================
# 2. TRAJECTORY & TRANSITION KERNEL
# ==========================================
def check_dense_u_turn(momentum_left, momentum_right, momentum_sum, inv_mass_matrix):
    rho_left = jnp.dot(momentum_left, jnp.dot(inv_mass_matrix, momentum_sum))
    rho_right = jnp.dot(momentum_right, jnp.dot(inv_mass_matrix, momentum_sum))
    return (rho_left < 0) | (rho_right < 0)

def extend_dense_trajectory(rng_key, init_state, direction, num_macro_steps, logprob_grad_fn, inv_mass_matrix, macro_step_size, max_energy_error, init_energy):
    def loop_condition(loop_state):
        step_idx, _, _, _, _, is_div, _, _, _ = loop_state
        return (step_idx < num_macro_steps) & jnp.logical_not(is_div)
        
    def step_body(loop_state):
        step_idx, curr_state, prop_state, curr_log_w, mom_sum, is_div, is_turn, unhalved_c, key = loop_state
        step_l, next_state, actual_error = dense_micro_routine(curr_state, logprob_grad_fn, inv_mass_matrix, macro_step_size * direction, max_energy_error)
        
        diverged = actual_error > max_energy_error + 1000.0
        new_is_div = is_div | diverged
        was_unhalved = jnp.where(step_l == 1, 1.0, 0.0)
        
        next_energy = compute_dense_hamiltonian(next_state, inv_mass_matrix)
        log_state_w = init_energy - next_energy
        new_log_w = jnp.logaddexp(curr_log_w, log_state_w)
        
        key, subkey = jax.random.split(key)
        keep = jax.random.uniform(subkey) < jnp.exp(log_state_w - new_log_w)
        new_prop_state = tree_util.tree_map(lambda old, new: jnp.where(keep, new, old), prop_state, next_state)
        
        new_mom_sum = mom_sum + next_state["momentum"]
        new_is_turn = check_dense_u_turn(init_state["momentum"], next_state["momentum"], new_mom_sum, inv_mass_matrix)
        
        return (step_idx + 1, next_state, new_prop_state, new_log_w, new_mom_sum, new_is_div, new_is_turn, unhalved_c + was_unhalved, key)
        
    init_loop = (0, init_state, init_state, -jnp.inf, init_state["momentum"], False, False, 0.0, rng_key)
    final_loop = jax.lax.while_loop(loop_condition, step_body, init_loop)
    _, final_bound, final_prop, tot_log_w, tot_mom_sum, div, turned, fin_unhalved, _ = final_loop
    return final_bound, final_prop, tot_log_w, tot_mom_sum, div, fin_unhalved, turned

def sample_dense_momentum(rng_key, inv_mass_matrix):
    """Samples momentum dynamically using Cholesky decomposition"""
    z = jax.random.normal(rng_key, shape=(inv_mass_matrix.shape[0],))
    L = jax.scipy.linalg.cholesky(inv_mass_matrix, lower=True)
    return jax.scipy.linalg.solve_triangular(L.T, z, lower=False)

@jax.jit(static_argnames=['logprob_grad_fn', 'max_tree_depth'])
def dense_walnuts_step(rng_key, curr_state, logprob_grad_fn, inv_mass_matrix, macro_step_size, max_energy_error, max_tree_depth=10):
    key_mom, key_tree = jax.random.split(rng_key)
    init_mom = sample_dense_momentum(key_mom, inv_mass_matrix)
    state = new_integrator_state(curr_state["position"], init_mom, curr_state["logprob"], curr_state["logprob_grad"])
    init_energy = compute_dense_hamiltonian(state, inv_mass_matrix)
    
    def tree_condition(loop_state):
        depth, _, _, _, _, _, is_turn, is_div, _, _ = loop_state
        return (depth < max_tree_depth) & jnp.logical_not(is_turn) & jnp.logical_not(is_div)
        
    def tree_body(loop_state):
        depth, left_s, right_s, prop_s, log_tot_w, mom_sum, is_turn, is_div, unh_c, key = loop_state
        key, dir_key, acc_key = jax.random.split(key, 3)
        direction = jnp.where(jax.random.uniform(dir_key) < 0.5, -1, 1)
        L = 2 ** depth
        
        def build_left(_): return extend_dense_trajectory(acc_key, left_s, -1, L, logprob_grad_fn, inv_mass_matrix, macro_step_size, max_energy_error, init_energy)
        def build_right(_): return extend_dense_trajectory(acc_key, right_s, 1, L, logprob_grad_fn, inv_mass_matrix, macro_step_size, max_energy_error, init_energy)
        
        bound_s, new_prop_s, log_sub_w, sub_mom, div, sub_unh, sub_turn = jax.lax.cond(direction == -1, build_left, build_right, operand=None)
        
        new_left_s = tree_util.tree_map(lambda old, new: jnp.where(direction == -1, new, old), left_s, bound_s)
        new_right_s = tree_util.tree_map(lambda old, new: jnp.where(direction == 1, new, old), right_s, bound_s)
        
        new_log_tot_w = jnp.logaddexp(log_tot_w, log_sub_w)
        keep = jax.random.uniform(acc_key) < jnp.exp(log_sub_w - new_log_tot_w)
        final_prop_s = tree_util.tree_map(lambda old, new: jnp.where(keep, new, old), prop_s, new_prop_s)
        
        new_mom_sum = mom_sum + sub_mom
        turning = check_dense_u_turn(new_left_s["momentum"], new_right_s["momentum"], new_mom_sum, inv_mass_matrix) | sub_turn
        return (depth + 1, new_left_s, new_right_s, final_prop_s, new_log_tot_w, new_mom_sum, turning, is_div | div, unh_c + sub_unh, key)
    
    init_loop = (0, state, state, state, 0.0, state["momentum"], False, False, 0.0, key_tree)
    fin_depth, _, _, fin_state, _, _, is_turn, is_div, fin_unh, _ = jax.lax.while_loop(tree_condition, tree_body, init_loop)
    
    out_state = tree_util.tree_map(lambda old, new: jnp.where(is_div, old, new), state, fin_state)
    tot_macro = (2 ** fin_depth) - 1.0
    unh_frac = jnp.where(tot_macro > 0, fin_unh / tot_macro, 0.0)
    return out_state, {"tree_depth": fin_depth, "diverging": is_div, "turning": is_turn, "unhalved_fraction": unh_frac}

# ==========================================
# 3. WARMUP & ADAPTATION
# ==========================================
def init_da(init_h): return {"log_h": jnp.log(init_h), "log_h_avg": jnp.log(init_h), "iter": 1, "err_sum": 0.0, "mu": jnp.log(10 * init_h)}
def update_da(state, unh_frac):
    err = 0.8 - unh_frac
    it = state["iter"]
    err_sum = state["err_sum"] + err
    log_h = state["mu"] - (jnp.sqrt(it) / 0.05) * (err_sum / (it + 10.0))
    eta = it ** -0.75
    return {"log_h": log_h, "log_h_avg": eta * log_h + (1 - eta) * state["log_h_avg"], "iter": it + 1, "err_sum": err_sum, "mu": state["mu"]}

def init_dense_welford(dim): return {"count": 0, "mean": jnp.zeros(dim), "m2": jnp.zeros((dim, dim))}
def update_dense_welford(state, pos):
    c = state["count"] + 1
    delta = pos - state["mean"]
    mean = state["mean"] + delta / c
    # Outer product for off-diagonal covariance tracking
    m2 = state["m2"] + jnp.outer(delta, pos - mean)
    return {"count": c, "mean": mean, "m2": m2}
def get_dense_inv_mass(state):
    v = state["m2"] / jnp.maximum(state["count"] - 1.0, 1.0)
    return v + jnp.eye(v.shape[0]) * 1e-3

@jax.jit(static_argnames=['logprob_grad_fn', 'num_warmup_steps'])
def dense_warmup(rng_key, init_state, logprob_grad_fn, num_warmup_steps=1000, init_h=0.1, delta=0.05):
    dim = init_state["position"].shape[0]
    da_s = init_da(init_h)
    wel_s = init_dense_welford(dim)
    inv_mass = jnp.eye(dim)
    window_ends = jnp.array([100, 150, 250, 450, 850])
    
    def scan_body(carry, step_idx):
        state, da_s, wel_s, curr_inv_mass, key = carry
        step_key, key = jax.random.split(key)
        curr_h = jnp.exp(da_s["log_h"])
        
        next_state, info = dense_walnuts_step(step_key, state, logprob_grad_fn, curr_inv_mass, curr_h, delta)
        da_s = update_da(da_s, info["unhalved_fraction"])
        
        is_slow = (step_idx >= 75) & (step_idx < 850)
        wel_s = tree_util.tree_map(lambda old, new: jnp.where(is_slow, new, old), wel_s, update_dense_welford(wel_s, next_state["position"]))
        
        is_update = jnp.any(step_idx == window_ends)
        curr_inv_mass = jnp.where(is_update, get_dense_inv_mass(wel_s), curr_inv_mass)
        wel_s = tree_util.tree_map(lambda x: jnp.where(is_update, 0.0, x), wel_s)
        return (next_state, da_s, wel_s, curr_inv_mass, key), None

    fin_carry, _ = jax.lax.scan(scan_body, (init_state, da_s, wel_s, inv_mass, rng_key), jnp.arange(num_warmup_steps))
    return {"final_state": fin_carry[0], "optimal_h": jnp.exp(fin_carry[1]["log_h_avg"]), "inverse_mass_matrix": fin_carry[3]}

@jax.jit(static_argnames=['logprob_grad_fn', 'num_draws'])
def dense_sample(rng_key, init_state, logprob_grad_fn, inv_mass, opt_h, delta, num_draws=2000):
    def scan_body(state, key):
        step_key, next_key = jax.random.split(key)
        next_state, _ = dense_walnuts_step(step_key, state, logprob_grad_fn, inv_mass, opt_h, delta)
        return next_state, next_state["position"]
    return jax.lax.scan(scan_body, init_state, jax.random.split(rng_key, num_draws))[1]

# ==========================================
# 4. EXECUTION
# ==========================================
def target_logprob(position):
    precision = jnp.array([[5.26315789, -4.73684211], [-4.73684211,  5.26315789]])
    return -0.5 * jnp.dot(position, jnp.dot(precision, position))
val_and_grad_fn = jax.jit(jax.value_and_grad(target_logprob))

def run():
    print("--- DENSE MASS MATRIX WALNUTS TEST ---")
    q_init = jnp.array([-3.0, 3.0])
    lp, grad = val_and_grad_fn(q_init)
    init_state = new_integrator_state(q_init, jnp.zeros_like(q_init), lp, grad)
    
    rng = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(rng)
    
    t0 = time.time()
    res = dense_warmup(k1, init_state, val_and_grad_fn)
    opt_h = res["optimal_h"]
    inv_mass = res["inverse_mass_matrix"]
    opt_h.block_until_ready()
    print(f"Warmup Time: {time.time()-t0:.2f}s | Opt h: {opt_h:.4f}")
    
    # You will see the off-diagonals dynamically learned during warmup!
    print(f"Learned DENSE Inverse Mass Matrix:\n{np.array(inv_mass)}\n")
    
    t1 = time.time()
    samples = dense_sample(k2, res["final_state"], val_and_grad_fn, inv_mass, opt_h, 0.05)
    samples.block_until_ready()
    t2 = time.time()
    print(f"Sampling Time: {t2-t1:.2f}s | Throughput: {2000/(t2-t1):.0f} draws/sec\n")
    
    samples_np = np.array(samples)
    print("Est. Mean:", np.mean(samples_np, axis=0))
    
    # The true covariance off-diagonals (0.9) will now be perfectly recovered
    print("Est. Covariance:\n", np.cov(samples_np, rowvar=False))

if __name__ == "__main__":
    run()