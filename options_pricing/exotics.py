"""Path-dependent (exotic) options priced by Monte Carlo under GBM.

These payoffs generally have no simple closed form, which is the canonical
motivation for simulation. Where a closed form *does* exist (the discrete
geometric-average Asian) it is used both to validate the simulator and as a
control variate to reduce the variance of the arithmetic Asian estimator.
"""

import math

import numpy as np
from scipy.stats import norm

from .paths import gbm_paths


def geometric_asian_closed_form(S0, K, T, r, sigma, n_fixings, option_type="call", q=0.0):
    """Exact price of a discrete geometric-average Asian option.

    The geometric average of log-normals is itself log-normal, giving a
    Black-Scholes-style formula.
    """
    t = np.arange(1, n_fixings + 1) * (T / n_fixings)
    mu_g = math.log(S0) + (r - q - 0.5 * sigma ** 2) * t.mean()
    # Var of the average of correlated Brownian terms: cov(W_ti, W_tj) = min(ti, tj).
    cov = np.minimum.outer(t, t)
    var_g = sigma ** 2 * cov.sum() / n_fixings ** 2
    sig_g = math.sqrt(var_g)

    d1 = (mu_g - math.log(K) + var_g) / sig_g
    d2 = d1 - sig_g
    if option_type == "call":
        return math.exp(-r * T) * (math.exp(mu_g + 0.5 * var_g) * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        return math.exp(-r * T) * (K * norm.cdf(-d2) - math.exp(mu_g + 0.5 * var_g) * norm.cdf(-d1))
    raise ValueError("option_type must be 'call' or 'put'")


def asian_mc_price(
    S0, K, T, r, sigma, n_fixings=50, option_type="call", q=0.0,
    n_paths=100_000, control_variate=True, seed=None,
):
    """Arithmetic-average Asian option by Monte Carlo, with an optional
    geometric-Asian control variate. Returns (price, standard_error)."""
    paths = gbm_paths(S0, T, r, sigma, n_paths, n_fixings, q=q, antithetic=True, seed=seed)
    monitored = paths[:, 1:]  # exclude S0

    arith_avg = monitored.mean(axis=1)
    if option_type == "call":
        payoff = np.maximum(arith_avg - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - arith_avg, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    disc_payoff = math.exp(-r * T) * payoff

    if not control_variate:
        return disc_payoff.mean(), disc_payoff.std(ddof=1) / math.sqrt(n_paths)

    geo_avg = np.exp(np.log(monitored).mean(axis=1))
    if option_type == "call":
        geo_payoff = np.maximum(geo_avg - K, 0.0)
    else:
        geo_payoff = np.maximum(K - geo_avg, 0.0)
    disc_geo = math.exp(-r * T) * geo_payoff
    geo_exact = geometric_asian_closed_form(S0, K, T, r, sigma, n_fixings, option_type, q)

    b = np.cov(disc_payoff, disc_geo)[0, 1] / np.var(disc_geo)
    adjusted = disc_payoff - b * (disc_geo - geo_exact)
    return adjusted.mean(), adjusted.std(ddof=1) / math.sqrt(n_paths)


def barrier_mc_price(
    S0, K, H, T, r, sigma, barrier_type="down-and-out", option_type="call",
    q=0.0, n_paths=100_000, n_steps=200, seed=None,
):
    """Knock-in / knock-out barrier option by Monte Carlo (discrete monitoring).

    barrier_type is one of: down-and-out, down-and-in, up-and-out, up-and-in.
    Returns (price, standard_error).
    """
    paths = gbm_paths(S0, T, r, sigma, n_paths, n_steps, q=q, antithetic=True, seed=seed)
    ST = paths[:, -1]

    if barrier_type.startswith("down"):
        breached = paths.min(axis=1) <= H
    elif barrier_type.startswith("up"):
        breached = paths.max(axis=1) >= H
    else:
        raise ValueError("barrier_type must start with 'down' or 'up'")

    if option_type == "call":
        vanilla = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        vanilla = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if barrier_type.endswith("out"):
        payoff = np.where(breached, 0.0, vanilla)
    else:  # knock-in
        payoff = np.where(breached, vanilla, 0.0)

    disc = math.exp(-r * T) * payoff
    return disc.mean(), disc.std(ddof=1) / math.sqrt(n_paths)


def lookback_mc_price(
    S0, T, r, sigma, option_type="call", q=0.0,
    n_paths=100_000, n_steps=200, seed=None,
):
    """Floating-strike lookback option by Monte Carlo.

    Call pays S_T - min(S); put pays max(S) - S_T. Returns (price, std_error).
    """
    paths = gbm_paths(S0, T, r, sigma, n_paths, n_steps, q=q, antithetic=True, seed=seed)
    ST = paths[:, -1]
    if option_type == "call":
        payoff = ST - paths.min(axis=1)
    elif option_type == "put":
        payoff = paths.max(axis=1) - ST
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    disc = math.exp(-r * T) * payoff
    return disc.mean(), disc.std(ddof=1) / math.sqrt(n_paths)
