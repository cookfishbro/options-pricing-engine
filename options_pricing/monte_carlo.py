"""Monte Carlo pricer for European options with optional variance reduction."""

import math

import numpy as np


def mc_price(
    S, K, T, r, sigma, option_type="call", n_paths=100_000,
    antithetic=True, control_variate=True, q=0.0, seed=None,
):
    """Returns (price, standard_error).

    antithetic: pairs Z with -Z to cut variance from sampling noise.
    control_variate: uses discounted terminal spot (known mean S) as a
    control variable, which is highly correlated with the option payoff.
    """
    rng = np.random.default_rng(seed)

    n = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal(n)
    if antithetic:
        Z = np.concatenate([Z, -Z])

    drift = (r - q - 0.5 * sigma ** 2) * T
    ST = S * np.exp(drift + sigma * math.sqrt(T) * Z)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoff = math.exp(-r * T) * payoff

    if control_variate:
        # E[exp(-rT) * ST] = S * exp(-qT) under the risk-neutral measure,
        # so (discounted ST - S*exp(-qT)) is a zero-mean control variable.
        control = math.exp(-r * T) * ST - S * math.exp(-q * T)
        b = np.cov(discounted_payoff, control)[0, 1] / np.var(control)
        adjusted = discounted_payoff - b * control
    else:
        adjusted = discounted_payoff

    price = adjusted.mean()
    se = adjusted.std(ddof=1) / math.sqrt(len(adjusted))
    return price, se
