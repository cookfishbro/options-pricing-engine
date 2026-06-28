"""Implied-volatility smile/surface construction.

Given any pricing model, compute prices across strikes and maturities, then
invert each price through Black-Scholes to obtain the implied volatility. A
constant-volatility model produces a flat surface; stochastic-volatility and
jump models reproduce the smile/skew observed in real markets.
"""

import numpy as np

from .implied_vol import implied_vol


def implied_vol_smile(price_func, S0, strikes, T, r, q=0.0, option_type="call"):
    """Return implied vols for a row of strikes at a single maturity.

    price_func(K) must return the option price for strike K.
    """
    ivs = []
    for K in strikes:
        price = price_func(K)
        try:
            ivs.append(implied_vol(price, S0, K, T, r, option_type, q))
        except (ValueError, RuntimeError):
            ivs.append(np.nan)
    return np.array(ivs)


def implied_vol_surface(price_func, S0, strikes, maturities, r, q=0.0, option_type="call"):
    """Return a (len(maturities), len(strikes)) grid of implied vols.

    price_func(K, T) must return the option price for strike K and maturity T.
    """
    surface = np.empty((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        surface[i] = implied_vol_smile(
            lambda K, T=T: price_func(K, T), S0, strikes, T, r, q, option_type
        )
    return surface
