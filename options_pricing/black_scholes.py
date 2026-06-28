"""Black-Scholes-Merton pricing and Greeks for European options."""

import math

from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """European option price under Black-Scholes-Merton.

    S: spot price, K: strike, T: time to maturity (years),
    r: risk-free rate, sigma: volatility, q: continuous dividend yield.
    """
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Returns delta, gamma, vega (per 1% vol), theta (per day), rho (per 1% rate)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = math.exp(-q * T) * norm.cdf(d1)
        theta_annual = (
            -S * math.exp(-q * T) * pdf_d1 * sigma / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
            + q * S * math.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type == "put":
        delta = -math.exp(-q * T) * norm.cdf(-d1)
        theta_annual = (
            -S * math.exp(-q * T) * pdf_d1 * sigma / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
            - q * S * math.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = math.exp(-q * T) * pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * pdf_d1 * math.sqrt(T) / 100

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_annual / 365,
        "rho": rho,
    }
