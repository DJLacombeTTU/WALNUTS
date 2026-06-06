from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import tree_util


class WalnutsState(NamedTuple):
    position: jax.Array
    logdensity: jax.Array
    logdensity_grad: jax.Array


class WalnutsInfo(NamedTuple):
    tree_depth: int
    diverging: bool
    turning: bool
    unhalved_fraction: jax.Array | float


def init(position: jax.Array, logdensity_fn: Callable) -> WalnutsState:
    logdensity, logdensity_grad = jax.value_and_grad(logdensity_fn)(position)
    return WalnutsState(position, logdensity, logdensity_grad)


def build_kernel(
    logdensity_fn: Callable,
    inverse_mass_matrix: jax.Array,
    step_size: float,
    max_energy_error: float = 1000.0,
    max_tree_depth: int = 10,
    max_halvings: int = 10,
) -> Callable:
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)

    # Compute invariant matrices inside the closure scope
    L_inv_mass = jax.scipy.linalg.cholesky(inverse_mass_matrix, lower=True)

    def dense_leapfrog(state: WalnutsState, momentum: jax.Array, step_sz: float):
        momentum_half = momentum + 0.5 * step_sz * state.logdensity_grad
        position_next = state.position + step_sz * jnp.dot(
            inverse_mass_matrix, momentum_half
        )
        logdensity_next, grad_next = logdensity_and_grad_fn(position_next)
        momentum_next = momentum_half + 0.5 * step_sz * grad_next
        return (
            WalnutsState(position_next, logdensity_next, grad_next),
            momentum_next,
        )

    def compute_dense_hamiltonian(state: WalnutsState, momentum: jax.Array):
        kinetic_energy = 0.5 * jnp.dot(momentum, jnp.dot(inverse_mass_matrix, momentum))
        return -state.logdensity + kinetic_energy

    def check_dense_u_turn(momentum_left, momentum_right, momentum_sum):
        rho_left = jnp.dot(momentum_left, jnp.dot(inverse_mass_matrix, momentum_sum))
        rho_right = jnp.dot(momentum_right, jnp.dot(inverse_mass_matrix, momentum_sum))
        return (rho_left < 0) | (rho_right < 0)

    def dense_micro_routine(
        initial_state: WalnutsState,
        initial_momentum: jax.Array,
        macro_step_size: float,
    ):
        initial_energy = compute_dense_hamiltonian(initial_state, initial_momentum)

        def condition_fn(loop_state):
            i, max_error, _, _, _ = loop_state
            return (max_error > max_energy_error) & (i < max_halvings)

        def body_fn(loop_state):
            i, _, _, _, _ = loop_state
            num_micro_steps = 2**i
            micro_step_size = macro_step_size / num_micro_steps

            def micro_step_fn(step_idx, inner_state):
                curr_state, curr_mom, curr_max_h, curr_min_h = inner_state
                next_state, next_mom = dense_leapfrog(
                    curr_state, curr_mom, micro_step_size
                )
                curr_energy = compute_dense_hamiltonian(next_state, next_mom)
                return (
                    next_state,
                    next_mom,
                    jnp.maximum(curr_max_h, curr_energy),
                    jnp.minimum(curr_min_h, curr_energy),
                )

            final_state, final_mom, max_h, min_h = jax.lax.fori_loop(
                0,
                num_micro_steps,
                micro_step_fn,
                (
                    initial_state,
                    initial_momentum,
                    initial_energy,
                    initial_energy,
                ),
            )
            return (
                i + 1,
                max_h - min_h,
                num_micro_steps,
                final_state,
                final_mom,
            )

        init_loop_state = (0, jnp.inf, 1, initial_state, initial_momentum)
        _, final_error, optimal_l, final_state, final_mom = jax.lax.while_loop(
            condition_fn, body_fn, init_loop_state
        )
        return optimal_l, final_state, final_mom, final_error

    def extend_dense_trajectory(
        rng_key,
        init_state: WalnutsState,
        init_mom: jax.Array,
        direction,
        num_macro_steps,
        init_energy,
    ):
        def loop_condition(loop_state):
            return (loop_state[0] < num_macro_steps) & jnp.logical_not(loop_state[7])

        def step_body(loop_state):
            (
                step_idx,
                curr_state,
                curr_mom,
                prop_state,
                prop_mom,
                curr_log_w,
                mom_sum,
                is_div,
                is_turn,
                unhalved_c,
                key,
            ) = loop_state
            step_l, next_state, next_mom, actual_error = dense_micro_routine(
                curr_state, curr_mom, step_size * direction
            )

            diverged = actual_error > max_energy_error + 1000.0
            new_is_div = is_div | diverged
            was_unhalved = jnp.where(step_l == 1, 1.0, 0.0)

            next_energy = compute_dense_hamiltonian(next_state, next_mom)
            log_state_w = init_energy - next_energy
            new_log_w = jnp.logaddexp(curr_log_w, log_state_w)

            key, subkey = jax.random.split(key)
            keep = jax.random.uniform(subkey) < jnp.exp(log_state_w - new_log_w)

            new_prop_state = tree_util.tree_map(
                lambda old, new: jnp.where(keep, new, old),
                prop_state,
                next_state,
            )
            new_prop_mom = jnp.where(keep, next_mom, prop_mom)

            new_mom_sum = mom_sum + next_mom
            new_is_turn = check_dense_u_turn(init_mom, next_mom, new_mom_sum)

            return (
                step_idx + 1,
                next_state,
                next_mom,
                new_prop_state,
                new_prop_mom,
                new_log_w,
                new_mom_sum,
                new_is_div,
                new_is_turn,
                unhalved_c + was_unhalved,
                key,
            )

        init_loop = (
            0,
            init_state,
            init_mom,
            init_state,
            init_mom,
            -jnp.inf,
            init_mom,
            False,
            False,
            0.0,
            rng_key,
        )
        final_loop = jax.lax.while_loop(loop_condition, step_body, init_loop)
        (
            _,
            final_bound_s,
            final_bound_m,
            final_prop_s,
            final_prop_m,
            tot_log_w,
            tot_mom_sum,
            div,
            turned,
            fin_unhalved,
            _,
        ) = final_loop
        return (
            final_bound_s,
            final_bound_m,
            final_prop_s,
            final_prop_m,
            tot_log_w,
            tot_mom_sum,
            div,
            fin_unhalved,
            turned,
        )

    def sample_dense_momentum(rng_key):
        z = jax.random.normal(rng_key, shape=(inverse_mass_matrix.shape[0],))
        return jax.scipy.linalg.solve_triangular(L_inv_mass.T, z, lower=False)

    def one_step(
        state: WalnutsState, rng_key: jax.Array
    ) -> tuple[WalnutsState, WalnutsInfo]:
        key_mom, key_tree = jax.random.split(rng_key)
        init_mom = sample_dense_momentum(key_mom)
        init_energy = compute_dense_hamiltonian(state, init_mom)

        def tree_condition(loop_state):
            return (
                (loop_state[0] < max_tree_depth)
                & jnp.logical_not(loop_state[9])
                & jnp.logical_not(loop_state[10])
            )

        def tree_body(loop_state):
            (
                depth,
                left_s,
                left_m,
                right_s,
                right_m,
                prop_s,
                prop_m,
                log_tot_w,
                mom_sum,
                is_turn,
                is_div,
                unh_c,
                key,
            ) = loop_state
            key, dir_key, acc_key = jax.random.split(key, 3)
            direction = jnp.where(jax.random.uniform(dir_key) < 0.5, -1, 1)
            L = 2**depth

            def build_left(_):
                return extend_dense_trajectory(
                    acc_key, left_s, left_m, -1, L, init_energy
                )

            def build_right(_):
                return extend_dense_trajectory(
                    acc_key, right_s, right_m, 1, L, init_energy
                )

            (
                bound_s,
                bound_m,
                new_prop_s,
                new_prop_m,
                log_sub_w,
                sub_mom,
                div,
                sub_unh,
                sub_turn,
            ) = jax.lax.cond(direction == -1, build_left, build_right, operand=None)

            new_left_s = tree_util.tree_map(
                lambda old, new: jnp.where(direction == -1, new, old),
                left_s,
                bound_s,
            )
            new_left_m = jnp.where(direction == -1, bound_m, left_m)
            new_right_s = tree_util.tree_map(
                lambda old, new: jnp.where(direction == 1, new, old),
                right_s,
                bound_s,
            )
            new_right_m = jnp.where(direction == 1, bound_m, right_m)

            new_log_tot_w = jnp.logaddexp(log_tot_w, log_sub_w)
            keep = jax.random.uniform(acc_key) < jnp.exp(log_sub_w - new_log_tot_w)

            final_prop_s = tree_util.tree_map(
                lambda old, new: jnp.where(keep, new, old), prop_s, new_prop_s
            )
            final_prop_m = jnp.where(keep, new_prop_m, prop_m)

            new_mom_sum = mom_sum + sub_mom
            turning = (
                check_dense_u_turn(new_left_m, new_right_m, new_mom_sum) | sub_turn
            )
            return (
                depth + 1,
                new_left_s,
                new_left_m,
                new_right_s,
                new_right_m,
                final_prop_s,
                final_prop_m,
                new_log_tot_w,
                new_mom_sum,
                turning,
                is_div | div,
                unh_c + sub_unh,
                key,
            )

        init_loop = (
            0,
            state,
            init_mom,
            state,
            init_mom,
            state,
            init_mom,
            0.0,
            init_mom,
            False,
            False,
            0.0,
            key_tree,
        )
        (
            fin_depth,
            _,
            _,
            _,
            _,
            fin_state,
            _,
            _,
            _,
            is_turn,
            is_div,
            fin_unh,
            _,
        ) = jax.lax.while_loop(tree_condition, tree_body, init_loop)

        out_state = tree_util.tree_map(
            lambda old, new: jnp.where(is_div, old, new), state, fin_state
        )
        tot_macro = (2**fin_depth) - 1.0
        unh_frac = jnp.where(tot_macro > 0, fin_unh / tot_macro, 0.0)

        info = WalnutsInfo(
            tree_depth=fin_depth,
            diverging=is_div,
            turning=is_turn,
            unhalved_fraction=unh_frac,
        )
        return out_state, info

    return one_step


class SamplingAlgorithm(NamedTuple):
    init: Callable
    step: Callable


class walnuts:
    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore
        cls,
        logdensity_fn: Callable,
        inverse_mass_matrix: jax.Array,
        step_size: float,
        *,
        max_energy_error: float = 1000.0,
        max_tree_depth: int = 10,
        max_halvings: int = 10,
    ) -> SamplingAlgorithm:
        def init_fn(position: jax.Array):
            return cls.init(position, logdensity_fn)

        kernel = cls.build_kernel(
            logdensity_fn,
            inverse_mass_matrix,
            step_size,
            max_energy_error=max_energy_error,
            max_tree_depth=max_tree_depth,
            max_halvings=max_halvings,
        )

        def step_fn(rng_key: jax.Array, state: WalnutsState):
            return kernel(state, rng_key)

        return SamplingAlgorithm(init_fn, step_fn)
