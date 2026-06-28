from options_pricing import mc_price, bs_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_mc_matches_black_scholes_within_confidence_interval():
    bs = bs_price(S, K, T, r, sigma, "call")
    price, se = mc_price(S, K, T, r, sigma, "call", n_paths=200_000, seed=42)
    assert abs(price - bs) < 3 * se


def test_variance_reduction_lowers_standard_error():
    _, se_plain = mc_price(
        S, K, T, r, sigma, "call", n_paths=50_000,
        antithetic=False, control_variate=False, seed=1,
    )
    _, se_reduced = mc_price(
        S, K, T, r, sigma, "call", n_paths=50_000,
        antithetic=True, control_variate=True, seed=1,
    )
    assert se_reduced < se_plain
