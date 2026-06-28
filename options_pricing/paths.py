"""Path generation utilities: GBM path simulation and quasi-random normals."""

import math

import numpy as np
from scipy.stats import norm, qmc


def gbm_paths(S0, T, r, sigma, n_paths, n_steps, q=0.0, antithetic=False, seed=None):
    """Simulate geometric Brownian motion paths.

    Returns an array of shape (n_paths, n_steps + 1) including the initial spot.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)
    else:
        z = rng.standard_normal((n_paths, n_steps))

    log_increments = drift + vol * z
    log_paths = np.concatenate(
        [np.full((z.shape[0], 1), math.log(S0)), log_increments], axis=1
    ).cumsum(axis=1)
    return np.exp(log_paths)


def sobol_normals(n_paths, dim=1, seed=None):
    """Generate standard-normal draws from a scrambled Sobol sequence.

    n_paths is rounded down to the nearest power of two, as Sobol sequences
    achieve their low-discrepancy property on such sample sizes.
    """
    m = int(math.log2(n_paths))
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    u = sampler.random_base2(m)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return norm.ppf(u)
