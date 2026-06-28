"""Heston (1993) stochastic volatility model.

Dynamics under the risk-neutral measure:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_t^v
    d<W^S, W^v>_t = rho dt

Parameters:
    v0     initial variance
    kappa  mean-reversion speed of variance
    theta  long-run variance
    sigma  volatility of variance ("vol of vol")
    rho    correlation between the two Brownian motions

Pricing is provided two ways: a semi-analytic Fourier integral and a
Monte Carlo simulation (full-truncation Euler), which serve as mutual
cross-checks.
"""

import math

import numpy as np
from scipy.integrate import quad


def _char_func(u, S0, T, r, q, v0, kappa, theta, sigma, rho):
    """Characteristic function of log(S_T) using the Albrecher et al. (2007)
    "good" formulation, which is stable across long maturities (avoids the
    branch-cut discontinuity known as the "Little Heston Trap")."""
    x0 = math.log(S0)
    xi = kappa - sigma * rho * 1j * u
    d = np.sqrt(xi ** 2 + sigma ** 2 * (1j * u + u ** 2))
    g = (xi - d) / (xi + d)
    exp_dt = np.exp(-d * T)

    C = (r - q) * 1j * u * T + (kappa * theta / sigma ** 2) * (
        (xi - d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = ((xi - d) / sigma ** 2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return np.exp(C + D * v0 + 1j * u * x0)


def heston_price(S0, K, T, r, v0, kappa, theta, sigma, rho, option_type="call", q=0.0):
    """Semi-analytic Heston price via the P1/P2 Fourier decomposition.

        Call = S0 e^{-qT} P1 - K e^{-rT} P2
    """
    args = (S0, T, r, q, v0, kappa, theta, sigma, rho)
    ln_K = math.log(K)

    # phi(-i) = E[S_T] = S0 exp((r-q)T) normalises the first probability.
    denom_p1 = _char_func(-1j, *args)

    def integrand_p1(u):
        cf = _char_func(u - 1j, *args) / denom_p1
        return (np.exp(-1j * u * ln_K) * cf / (1j * u)).real

    def integrand_p2(u):
        cf = _char_func(u, *args)
        return (np.exp(-1j * u * ln_K) * cf / (1j * u)).real

    P1 = 0.5 + (1.0 / math.pi) * quad(integrand_p1, 0, 200, limit=200)[0]
    P2 = 0.5 + (1.0 / math.pi) * quad(integrand_p2, 0, 200, limit=200)[0]

    call = S0 * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2
    if option_type == "call":
        return call
    elif option_type == "put":
        return call - S0 * math.exp(-q * T) + K * math.exp(-r * T)  # put-call parity
    raise ValueError("option_type must be 'call' or 'put'")


def heston_mc_price(
    S0, K, T, r, v0, kappa, theta, sigma, rho, option_type="call", q=0.0,
    n_paths=100_000, n_steps=200, seed=None,
):
    """Monte Carlo price under the full-truncation Euler scheme (Lord et al.,
    2010). Returns (price, standard_error)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    log_s = np.full(n_paths, math.log(S0))
    v = np.full(n_paths, v0)
    chol = np.array([[1.0, 0.0], [rho, math.sqrt(1.0 - rho ** 2)]])

    for _ in range(n_steps):
        z = rng.standard_normal((n_paths, 2)) @ chol.T
        z_s, z_v = z[:, 0], z[:, 1]
        v_pos = np.maximum(v, 0.0)
        log_s += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z_s
        v += kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * sqrt_dt * z_v

    ST = np.exp(log_s)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = math.exp(-r * T) * payoff
    return discounted.mean(), discounted.std(ddof=1) / math.sqrt(n_paths)
