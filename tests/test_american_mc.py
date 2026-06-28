from options_pricing import binomial_price, longstaff_schwartz_price, bs_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_lsm_matches_binomial_for_american_put():
    lsm, se = longstaff_schwartz_price(S, K, T, r, sigma, "put",
                                       n_paths=100_000, n_steps=50, seed=1)
    tree = binomial_price(S, K, T, r, sigma, n_steps=1000, option_type="put", american=True)
    assert abs(lsm - tree) < 3 * se + 0.02


def test_american_put_at_least_european():
    lsm, _ = longstaff_schwartz_price(S, K, T, r, sigma, "put",
                                      n_paths=100_000, n_steps=50, seed=2)
    european = bs_price(S, K, T, r, sigma, "put")
    assert lsm >= european - 0.05
