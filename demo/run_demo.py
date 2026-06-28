"""Benchmark numerical-method figures, computed on the REAL SPY contract.

Reads the reference contract and calibrated parameters produced by
``real_data_demo.py`` (results/real_summary.json), so every figure here is based
on genuine market data rather than toy inputs.

Run:  python demo/real_data_demo.py   (first, to produce the summary)
      python demo/run_demo.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options_pricing import bs_price, bs_greeks, binomial_price, mc_price, implied_vol

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_reference():
    path = os.path.join(RESULTS_DIR, "real_summary.json")
    if not os.path.exists(path):
        raise SystemExit("Run `python demo/real_data_demo.py` first "
                         "(it writes results/real_summary.json).")
    with open(path) as f:
        s = json.load(f)
    return s["meta"], s["cross_validation"]["contract"]


META, CONTRACT = load_reference()
S, K = CONTRACT["S0"], CONTRACT["K"]
T, r, q, sigma = CONTRACT["T"], CONTRACT["r"], CONTRACT["q"], CONTRACT["atm_iv"]
LABEL = f"{META['ticker']} {META['asof']}, K={K:.0f}, T={T:.2f}, IV={sigma*100:.1f}%"


def binomial_convergence_plot():
    bs = bs_price(S, K, T, r, sigma, "call", q)
    steps = np.arange(5, 305, 5)
    errors = [abs(binomial_price(S, K, T, r, sigma, n, "call", q=q) - bs) for n in steps]

    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, errors)
    plt.xlabel("Number of steps")
    plt.ylabel("Absolute error vs Black-Scholes")
    plt.title(f"Binomial Tree Convergence ({LABEL})")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "binomial_convergence.png"), dpi=150)
    plt.close()


def monte_carlo_variance_reduction_plot():
    bs = bs_price(S, K, T, r, sigma, "call", q)
    path_counts = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

    plain_errors, reduced_errors = [], []
    for n in path_counts:
        price_plain, _ = mc_price(S, K, T, r, sigma, "call", n_paths=n, q=q,
                                  antithetic=False, control_variate=False, seed=7)
        price_reduced, _ = mc_price(S, K, T, r, sigma, "call", n_paths=n, q=q,
                                    antithetic=True, control_variate=True, seed=7)
        plain_errors.append(abs(price_plain - bs))
        reduced_errors.append(abs(price_reduced - bs))

    plt.figure(figsize=(7, 4.5))
    plt.plot(path_counts, plain_errors, marker="o", label="Plain Monte Carlo")
    plt.plot(path_counts, reduced_errors, marker="o", label="Antithetic + Control Variate")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of simulated paths")
    plt.ylabel("Absolute error vs Black-Scholes")
    plt.title(f"Monte Carlo Variance Reduction ({LABEL})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mc_variance_reduction.png"), dpi=150)
    plt.close()


def print_summary_table():
    bs = bs_price(S, K, T, r, sigma, "call", q)
    tree = binomial_price(S, K, T, r, sigma, 500, "call", q=q)
    mc, se = mc_price(S, K, T, r, sigma, "call", n_paths=200_000, q=q, seed=42)
    greeks = bs_greeks(S, K, T, r, sigma, "call", q)

    print(f"Reference contract: {LABEL}")
    print(f"Inputs: S={S}, K={K}, T={T:.4f}, r={r}, q={q}, sigma={sigma}\n")
    print(f"{'Method':<30}{'Price':>10}")
    print(f"{'Black-Scholes':<30}{bs:>10.4f}")
    print(f"{'Binomial Tree (500 steps)':<30}{tree:>10.4f}")
    print(f"{'Monte Carlo (200k paths)':<30}{mc:>10.4f}  (SE = {se:.5f})")
    print("\nGreeks (Black-Scholes):")
    for name, value in greeks.items():
        print(f"  {name:<8}{value:>10.5f}")


if __name__ == "__main__":
    print_summary_table()
    binomial_convergence_plot()
    monte_carlo_variance_reduction_plot()
    print(f"\nPlots saved to {RESULTS_DIR}")
