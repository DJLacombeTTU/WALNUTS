from typing import NamedTuple
import jax.numpy as jnp

class WALNUTSState(NamedTuple):
    position: jnp.ndarray
    logdensity: float
    logdensity_grad: jnp.ndarray
    # Note: Momentum is sampled transiently inside the step, 
    # so it does not persist in the main Markov state.

class WALNUTSInfo(NamedTuple):
    tree_depth: int
    diverging: bool
    turning: bool
    unhalved_fraction: float