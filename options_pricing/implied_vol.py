"""Implied volatility solver: Newton-Raphson with a bisection fallback."""

from scipy.optimize import brentq

from .black_scholes import bs_greeks, bs_price


def implied_vol(price, S, K, T, r, option_type="call", q=0.0, tol=1e-8, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        model_price = bs_price(S, K, T, r, sigma, option_type, q)
        diff = model_price - price
        if abs(diff) < tol * max(1.0, price):
            return sigma
        vega = bs_greeks(S, K, T, r, sigma, option_type, q)["vega"] * 100
        if vega < 1e-10:
            break
        sigma -= diff / vega
        if sigma <= 0:
            break

    # Newton-Raphson diverged (bad initial guess / near-zero vega) -> fall back to brentq.
    def objective(s):
        return bs_price(S, K, T, r, s, option_type, q) - price

    return brentq(objective, 1e-6, 5.0, xtol=tol)
