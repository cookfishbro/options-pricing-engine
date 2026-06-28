"""Calibration of the Heston and Merton models to market option quotes.

Given observed option prices across strikes/maturities, find the model
parameters that minimise the (optionally weighted) squared pricing error. This
is the practical workflow a desk uses to mark exotics consistently with vanilla
market quotes.

Weights are typically set to the reciprocal of each option's Black-Scholes vega,
which converts a dollar-price residual into an approximate implied-volatility
residual and prevents cheap out-of-the-money options (which define the skew)
from being swamped by expensive at-the-money ones.
"""

import numpy as np
from scipy.optimize import least_squares

from .heston import heston_price
from .merton_jump import merton_price


def _unpack(market_quotes, r, q):
    """Return per-quote arrays. Quotes may be (K, T, price, option_type) or
    (K, T, price, option_type, r, q); in the latter case the per-quote r and q
    override the scalar defaults (used for surfaces with a maturity-dependent
    forward)."""
    Ks = np.array([t[0] for t in market_quotes])
    Ts = np.array([t[1] for t in market_quotes])
    prices = np.array([t[2] for t in market_quotes])
    types = [t[3] for t in market_quotes]
    rs = np.array([t[4] if len(t) > 4 else r for t in market_quotes])
    qs = np.array([t[5] if len(t) > 5 else q for t in market_quotes])
    return Ks, Ts, prices, types, rs, qs


def calibrate_heston(market_quotes, S0, r=0.0, q=0.0, weights=None,
                     initial_guess=None, verbose=False):
    """Calibrate Heston parameters (v0, kappa, theta, sigma, rho) to quotes.

    market_quotes: iterable of (K, T, price, option_type) or
    (K, T, price, option_type, r, q) tuples.
    weights: optional per-quote weights (e.g. 1/vega). Defaults to equal weights.
    Returns a dict of calibrated parameters and the weighted RMSE.
    """
    if initial_guess is None:
        initial_guess = [0.04, 2.0, 0.04, 0.5, -0.5]  # v0, kappa, theta, sigma, rho
    lower = [1e-4, 1e-2, 1e-4, 1e-2, -0.999]
    upper = [1.0, 20.0, 1.0, 5.0, 0.999]

    Ks, Ts, prices, types, rs, qs = _unpack(market_quotes, r, q)
    w = np.ones_like(prices) if weights is None else np.asarray(weights)

    def residuals(params):
        v0, kappa, theta, sigma, rho = params
        model = np.array([
            heston_price(S0, K, T, ri, v0, kappa, theta, sigma, rho, ot, qi)
            for K, T, ot, ri, qi in zip(Ks, Ts, types, rs, qs)
        ])
        return (model - prices) * w

    result = least_squares(residuals, initial_guess, bounds=(lower, upper),
                           method="trf", verbose=2 if verbose else 0)
    v0, kappa, theta, sigma, rho = (float(x) for x in result.x)
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    return {
        "v0": v0, "kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho,
        "rmse": rmse, "feller_satisfied": bool(2 * kappa * theta > sigma ** 2),
    }


def calibrate_merton(market_quotes, S0, r=0.0, q=0.0, weights=None,
                     initial_guess=None, verbose=False):
    """Calibrate Merton parameters (sigma, lam, muJ, sigmaJ) to quotes.

    Same interface as calibrate_heston. Returns a dict of parameters and RMSE.
    """
    if initial_guess is None:
        initial_guess = [0.15, 0.5, -0.1, 0.15]  # sigma, lam, muJ, sigmaJ
    lower = [1e-3, 1e-3, -1.0, 1e-3]
    upper = [1.0, 10.0, 1.0, 1.0]

    Ks, Ts, prices, types, rs, qs = _unpack(market_quotes, r, q)
    w = np.ones_like(prices) if weights is None else np.asarray(weights)

    def residuals(params):
        sigma, lam, muJ, sigmaJ = params
        model = np.array([
            merton_price(S0, K, T, ri, sigma, lam, muJ, sigmaJ, ot, qi)
            for K, T, ot, ri, qi in zip(Ks, Ts, types, rs, qs)
        ])
        return (model - prices) * w

    result = least_squares(residuals, initial_guess, bounds=(lower, upper),
                           method="trf", verbose=2 if verbose else 0)
    sigma, lam, muJ, sigmaJ = (float(x) for x in result.x)
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    return {
        "sigma": sigma, "lam": lam, "muJ": muJ, "sigmaJ": sigmaJ, "rmse": rmse,
    }
