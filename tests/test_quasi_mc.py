import numpy as np

from options_pricing import bs_price, qmc_european_price, mc_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def test_qmc_converges_to_black_scholes():
    bs = bs_price(S, K, T, r, sigma, "call")
    qmc = qmc_european_price(S, K, T, r, sigma, "call", n_paths=65536, seed=1)
    assert abs(qmc - bs) < 0.01


def test_qmc_beats_pseudo_random_at_equal_budget():
    # Averaged over several seeds, Sobol QMC should have smaller mean error than
    # plain pseudo-random Monte Carlo for this smooth 1-D payoff.
    bs = bs_price(S, K, T, r, sigma, "call")
    n = 4096
    qmc_err, mc_err = [], []
    for seed in range(8):
        qmc_err.append(abs(qmc_european_price(S, K, T, r, sigma, "call", n_paths=n, seed=seed) - bs))
        price, _ = mc_price(S, K, T, r, sigma, "call", n_paths=n,
                            antithetic=False, control_variate=False, seed=seed)
        mc_err.append(abs(price - bs))
    assert np.mean(qmc_err) < np.mean(mc_err)
