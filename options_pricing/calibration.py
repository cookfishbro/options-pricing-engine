"""Calibration of the Heston model to a market volatility smile.

Given observed option prices (or implied vols) across strikes/maturities, find
the Heston parameters (v0, kappa, theta, sigma, rho) that minimise the squared
pricing error. This is the practical workflow a desk uses to mark exotics
consistently with vanilla market quotes.
"""

import numpy as np
from scipy.optimize import least_squares

from .heston import heston_price


def calibrate_heston(market_quotes, S0, r, q=0.0, initial_guess=None, verbose=False):
    """Calibrate Heston parameters to a set of market quotes.

    market_quotes: iterable of (K, T, price, option_type) tuples.
    Returns a dict of calibrated parameters and the final RMSE.

    The Feller condition (2 kappa theta > sigma^2) is encouraged via bounds but
    not strictly enforced, mirroring common market practice.
    """
    if initial_guess is None:
        initial_guess = [0.04, 2.0, 0.04, 0.5, -0.5]  # v0, kappa, theta, sigma, rho

    lower = [1e-4, 1e-2, 1e-4, 1e-2, -0.999]
    upper = [1.0, 20.0, 1.0, 5.0, 0.999]

    Ks = np.array([q_[0] for q_ in market_quotes])
    Ts = np.array([q_[1] for q_ in market_quotes])
    prices = np.array([q_[2] for q_ in market_quotes])
    types = [q_[3] for q_ in market_quotes]

    def residuals(params):
        v0, kappa, theta, sigma, rho = params
        model = np.array([
            heston_price(S0, K, T, r, v0, kappa, theta, sigma, rho, ot, q)
            for K, T, ot in zip(Ks, Ts, types)
        ])
        return model - prices

    result = least_squares(
        residuals, initial_guess, bounds=(lower, upper),
        method="trf", verbose=2 if verbose else 0,
    )

    v0, kappa, theta, sigma, rho = result.x
    rmse = np.sqrt(np.mean(result.fun ** 2))
    return {
        "v0": v0, "kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho,
        "rmse": rmse, "feller_satisfied": 2 * kappa * theta > sigma ** 2,
    }
