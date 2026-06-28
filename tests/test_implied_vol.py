from options_pricing import bs_price, implied_vol

S, K, T, r = 100, 100, 1.0, 0.05


def test_recovers_known_vol():
    true_sigma = 0.27
    price = bs_price(S, K, T, r, true_sigma, "call")
    iv = implied_vol(price, S, K, T, r, "call")
    assert abs(iv - true_sigma) < 1e-6


def test_recovers_known_vol_for_put():
    true_sigma = 0.15
    price = bs_price(S, K, T, r, true_sigma, "put")
    iv = implied_vol(price, S, K, T, r, "put")
    assert abs(iv - true_sigma) < 1e-6


def test_handles_low_vega_otm_via_fallback():
    true_sigma = 0.18
    price = bs_price(S=100, K=140, T=0.5, r=r, sigma=true_sigma, option_type="call")
    iv = implied_vol(price, S=100, K=140, T=0.5, r=r, option_type="call")
    assert abs(iv - true_sigma) < 1e-4
