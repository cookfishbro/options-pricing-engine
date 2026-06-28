"""American option pricing by Least-Squares Monte Carlo (Longstaff & Schwartz, 2001).

At each exercise date the continuation value is estimated by regressing
discounted future cash flows on a polynomial basis of the spot price, using
only in-the-money paths. The optimal early-exercise policy is then applied
backward through time.
"""

import math

import numpy as np

from .paths import gbm_paths


def longstaff_schwartz_price(
    S0, K, T, r, sigma, option_type="put", q=0.0,
    n_paths=100_000, n_steps=50, basis_degree=3, seed=None,
):
    """Returns (price, standard_error) for an American option via LSM."""
    paths = gbm_paths(S0, T, r, sigma, n_paths, n_steps, q=q, antithetic=True, seed=seed)
    dt = T / n_steps
    discount = math.exp(-r * dt)

    if option_type == "put":
        intrinsic = lambda s: np.maximum(K - s, 0.0)
    elif option_type == "call":
        intrinsic = lambda s: np.maximum(s - K, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Cash flow if held to the final step, then roll backward.
    cash_flow = intrinsic(paths[:, -1])

    for t in range(n_steps - 1, 0, -1):
        cash_flow *= discount
        spot = paths[:, t]
        exercise = intrinsic(spot)
        itm = exercise > 0
        if itm.sum() == 0:
            continue

        # Regress continuation (discounted future CF) on a polynomial of spot.
        coeffs = np.polyfit(spot[itm], cash_flow[itm], basis_degree)
        continuation = np.polyval(coeffs, spot[itm])

        do_exercise = exercise[itm] > continuation
        idx = np.where(itm)[0][do_exercise]
        cash_flow[idx] = exercise[itm][do_exercise]

    price_paths = cash_flow * discount
    return price_paths.mean(), price_paths.std(ddof=1) / math.sqrt(n_paths)
