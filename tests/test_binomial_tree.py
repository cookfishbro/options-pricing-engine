from options_pricing import binomial_price, bs_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_european_converges_to_black_scholes():
    bs = bs_price(S, K, T, r, sigma, "call")
    tree = binomial_price(S, K, T, r, sigma, n_steps=2000, option_type="call", american=False)
    assert abs(bs - tree) < 0.01


def test_american_put_at_least_as_valuable_as_european():
    european = binomial_price(S, K, T, r, sigma, n_steps=500, option_type="put", american=False)
    american = binomial_price(S, K, T, r, sigma, n_steps=500, option_type="put", american=True)
    assert american >= european - 1e-9


def test_american_call_no_dividend_equals_european():
    # With q=0, early exercise of a call is never optimal, so American == European.
    european = binomial_price(S, K, T, r, sigma, n_steps=500, option_type="call", american=False)
    american = binomial_price(S, K, T, r, sigma, n_steps=500, option_type="call", american=True)
    assert abs(european - american) < 1e-6
