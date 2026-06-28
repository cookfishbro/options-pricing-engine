"""Cox-Ross-Rubinstein binomial tree pricer for European and American options."""

import math

import numpy as np


def binomial_price(S, K, T, r, sigma, n_steps=200, option_type="call", american=False, q=0.0):
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    discount = math.exp(-r * dt)

    j = np.arange(n_steps + 1)
    spot_at_maturity = S * u ** j * d ** (n_steps - j)
    if option_type == "call":
        values = np.maximum(spot_at_maturity - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - spot_at_maturity, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    for step in range(n_steps - 1, -1, -1):
        values = discount * (p * values[1:step + 2] + (1 - p) * values[0:step + 1])
        if american:
            j = np.arange(step + 1)
            spot = S * u ** j * d ** (step - j)
            exercise = np.maximum(spot - K, 0.0) if option_type == "call" else np.maximum(K - spot, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])
