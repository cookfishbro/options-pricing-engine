"""Quasi-Monte Carlo pricing of European options using a Sobol sequence.

Replacing pseudo-random draws with a low-discrepancy sequence improves the
convergence rate of the integration error from O(n^{-1/2}) toward O(n^{-1})
(up to log factors) for low-dimensional, smooth integrands such as the
single-step European payoff.
"""

import math

import numpy as np

from .paths import sobol_normals


def qmc_european_price(S0, K, T, r, sigma, option_type="call", q=0.0, n_paths=65536, seed=None):
    """Price a European option with Sobol quasi-random terminal draws.

    Returns the price. (A low-discrepancy sequence has no i.i.d. standard
    error; accuracy is assessed by convergence experiments instead.)
    """
    z = sobol_normals(n_paths, dim=1, seed=seed).ravel()
    drift = (r - q - 0.5 * sigma ** 2) * T
    ST = S0 * np.exp(drift + sigma * math.sqrt(T) * z)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return math.exp(-r * T) * payoff.mean()
