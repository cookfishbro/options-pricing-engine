"""Merton (1976) jump-diffusion model.

Dynamics under the risk-neutral measure:

    dS_t / S_t = (r - q - lambda*k) dt + sigma dW_t + (J - 1) dN_t

where N_t is a Poisson process with intensity lambda, jump sizes J are
log-normal with ln J ~ N(muJ, sigmaJ^2), and k = E[J-1] = exp(muJ + sigmaJ^2/2) - 1
is the compensator that keeps the discounted price a martingale.

Provides Merton's closed-form series (a Poisson-weighted sum of Black-Scholes
prices) and a Monte Carlo cross-check.
"""

import math

import numpy as np

from .black_scholes import bs_price


def merton_price(
    S, K, T, r, sigma, lam, muJ, sigmaJ, option_type="call", q=0.0, n_terms=50,
):
    """Closed-form Merton price as a Poisson-weighted sum of BS prices."""
    k = math.exp(muJ + 0.5 * sigmaJ ** 2) - 1.0
    lam_prime = lam * (1.0 + k)  # jump-adjusted intensity (see note below)
    price = 0.0
    for n in range(n_terms):
        # Weight uses lam_prime so the per-term BS rate r_n stays consistent with
        # discounting at r; equivalently this absorbs the e^{(r_n - r)T} factor.
        poisson_w = math.exp(-lam_prime * T) * (lam_prime * T) ** n / math.factorial(n)
        sigma_n = math.sqrt(sigma ** 2 + n * sigmaJ ** 2 / T)
        r_n = r - lam * k + n * (muJ + 0.5 * sigmaJ ** 2) / T
        price += poisson_w * bs_price(S, K, T, r_n, sigma_n, option_type, q)
    return price


def merton_terminal_samples(
    S, T, r, sigma, lam, muJ, sigmaJ, q=0.0, n_paths=200_000, seed=None,
):
    """Simulate terminal prices S_T under the Merton model. Returns an array of
    length n_paths."""
    rng = np.random.default_rng(seed)
    k = math.exp(muJ + 0.5 * sigmaJ ** 2) - 1.0

    n_jumps = rng.poisson(lam * T, size=n_paths)
    z = rng.standard_normal(n_paths)
    # Conditional on n jumps, the total jump in log-price is N(n*muJ, n*sigmaJ^2).
    jump_component = n_jumps * muJ + sigmaJ * np.sqrt(n_jumps) * rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma ** 2 - lam * k) * T
    log_ST = math.log(S) + drift + sigma * math.sqrt(T) * z + jump_component
    return np.exp(log_ST)


def merton_mc_price(
    S, K, T, r, sigma, lam, muJ, sigmaJ, option_type="call", q=0.0,
    n_paths=200_000, seed=None,
):
    """Monte Carlo price by simulating the terminal value directly. Returns
    (price, standard_error)."""
    ST = merton_terminal_samples(S, T, r, sigma, lam, muJ, sigmaJ, q, n_paths, seed)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = math.exp(-r * T) * payoff
    return discounted.mean(), discounted.std(ddof=1) / math.sqrt(n_paths)
