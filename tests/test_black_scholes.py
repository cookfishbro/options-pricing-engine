import math

from options_pricing import bs_price, bs_greeks

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_put_call_parity():
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    parity_lhs = call - put
    parity_rhs = S - K * math.exp(-r * T)
    assert abs(parity_lhs - parity_rhs) < 1e-10


def test_call_price_known_value():
    # Reference value for S=K=100, T=1, r=5%, sigma=20% (standard textbook case).
    call = bs_price(S, K, T, r, sigma, "call")
    assert abs(call - 10.4506) < 1e-3


def test_deep_itm_call_converges_to_intrinsic():
    call = bs_price(S=1000, K=100, T=1, r=r, sigma=sigma, option_type="call")
    assert abs(call - (1000 - 100 * math.exp(-r * 1))) < 0.5


def test_delta_bounds():
    g_call = bs_greeks(S, K, T, r, sigma, "call")
    g_put = bs_greeks(S, K, T, r, sigma, "put")
    assert 0 < g_call["delta"] < 1
    assert -1 < g_put["delta"] < 0


def test_gamma_same_for_call_and_put():
    g_call = bs_greeks(S, K, T, r, sigma, "call")
    g_put = bs_greeks(S, K, T, r, sigma, "put")
    assert abs(g_call["gamma"] - g_put["gamma"]) < 1e-10
