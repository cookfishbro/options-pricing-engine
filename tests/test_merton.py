import math

from options_pricing import (
    bs_price, merton_price, merton_mc_price, merton_terminal_samples,
)

S, K, T, r = 100, 100, 1.0, 0.05
JUMP = dict(sigma=0.2, lam=0.5, muJ=-0.1, sigmaJ=0.15)


def test_reduces_to_black_scholes_without_jumps():
    bs = bs_price(S, K, T, r, 0.2, "call")
    merton = merton_price(S, K, T, r, sigma=0.2, lam=0.0, muJ=0.0, sigmaJ=0.0, option_type="call")
    assert abs(bs - merton) < 1e-8


def test_closed_form_matches_monte_carlo():
    closed = merton_price(S, K, T, r, option_type="call", **JUMP)
    mc, se = merton_mc_price(S, K, T, r, option_type="call", n_paths=500_000, seed=2, **JUMP)
    assert abs(closed - mc) < 3 * se


def test_terminal_samples_satisfy_martingale_property():
    ST = merton_terminal_samples(S, T, r, n_paths=500_000, seed=4, **JUMP)
    discounted_mean = math.exp(-r * T) * ST.mean()
    assert abs(discounted_mean - S) / S < 0.01


def test_negative_jumps_raise_put_value_vs_black_scholes():
    # Downward jumps (muJ < 0) add left-tail risk, so a put is worth more than
    # under pure diffusion with the same diffusive vol.
    bs_put = bs_price(S, K, T, r, 0.2, "put")
    merton_put = merton_price(S, K, T, r, option_type="put", **JUMP)
    assert merton_put > bs_put
