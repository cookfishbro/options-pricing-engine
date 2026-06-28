import math

import numpy as np

from options_pricing import (
    bs_price, heston_price, heston_mc_price, heston_terminal_samples,
)

S, K, T, r = 100, 100, 1.0, 0.05
PARAMS = dict(v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7)


def test_reduces_to_black_scholes_as_vol_of_vol_vanishes():
    # With v0 = theta and negligible vol-of-vol, variance is ~constant at 0.04,
    # so the Heston price must match BS with sigma = sqrt(0.04) = 0.20.
    bs = bs_price(S, K, T, r, 0.2, "call")
    hest = heston_price(S, K, T, r, v0=0.04, kappa=2.0, theta=0.04,
                        sigma=1e-4, rho=0.0, option_type="call")
    assert abs(bs - hest) < 1e-3


def test_put_call_parity():
    call = heston_price(S, K, T, r, option_type="call", **PARAMS)
    put = heston_price(S, K, T, r, option_type="put", **PARAMS)
    assert abs((call - put) - (S - K * math.exp(-r * T))) < 1e-6


def test_fourier_matches_monte_carlo():
    fourier = heston_price(S, K, T, r, option_type="call", **PARAMS)
    mc, se = heston_mc_price(S, K, T, r, option_type="call",
                             n_paths=200_000, n_steps=200, seed=1, **PARAMS)
    # Within ~3 SE plus a small allowance for Euler discretisation bias.
    assert abs(fourier - mc) < 3 * se + 0.05


def test_terminal_samples_satisfy_martingale_property():
    # Discounted expected terminal price must equal the forward S0*exp(-qT).
    ST = heston_terminal_samples(S, T, r, n_paths=200_000, n_steps=200, seed=1, **PARAMS)
    discounted_mean = math.exp(-r * T) * ST.mean()
    assert abs(discounted_mean - S) / S < 0.01


def test_negative_correlation_creates_negative_skew():
    # Equity markets: rho < 0 makes OTM puts (low strikes) relatively expensive,
    # so the implied-vol of a low strike exceeds that of a high strike.
    from options_pricing import implied_vol
    low_k = heston_price(S, 80, T, r, option_type="put", **PARAMS)
    high_k = heston_price(S, 120, T, r, option_type="put", **PARAMS)
    iv_low = implied_vol(low_k, S, 80, T, r, "put")
    iv_high = implied_vol(high_k, S, 120, T, r, "put")
    assert iv_low > iv_high
