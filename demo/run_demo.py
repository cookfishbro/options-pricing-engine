"""Generates convergence plots and a pricing summary table, saved to results/."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options_pricing import bs_price, bs_greeks, binomial_price, mc_price, implied_vol

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2


def binomial_convergence_plot():
    bs = bs_price(S, K, T, r, sigma, "call")
    steps = np.arange(5, 305, 5)
    errors = [abs(binomial_price(S, K, T, r, sigma, n_steps=n, option_type="call") - bs) for n in steps]

    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, errors)
    plt.xlabel("Number of steps")
    plt.ylabel("Absolute error vs Black-Scholes")
    plt.title("Binomial Tree Convergence (European Call)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "binomial_convergence.png"), dpi=150)
    plt.close()


def monte_carlo_variance_reduction_plot():
    bs = bs_price(S, K, T, r, sigma, "call")
    path_counts = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

    plain_errors, reduced_errors = [], []
    for n in path_counts:
        price_plain, _ = mc_price(S, K, T, r, sigma, "call", n_paths=n,
                                   antithetic=False, control_variate=False, seed=7)
        price_reduced, _ = mc_price(S, K, T, r, sigma, "call", n_paths=n,
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
    plt.title("Monte Carlo Variance Reduction (European Call)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mc_variance_reduction.png"), dpi=150)
    plt.close()


def print_summary_table():
    bs = bs_price(S, K, T, r, sigma, "call")
    tree = binomial_price(S, K, T, r, sigma, n_steps=500, option_type="call")
    mc, se = mc_price(S, K, T, r, sigma, "call", n_paths=200_000, seed=42)
    greeks = bs_greeks(S, K, T, r, sigma, "call")
    iv = implied_vol(bs, S, K, T, r, "call")

    print(f"Inputs: S={S}, K={K}, T={T}, r={r}, sigma={sigma}\n")
    print(f"{'Method':<30}{'Price':>10}")
    print(f"{'Black-Scholes':<30}{bs:>10.4f}")
    print(f"{'Binomial Tree (500 steps)':<30}{tree:>10.4f}")
    print(f"{'Monte Carlo (200k paths)':<30}{mc:>10.4f}  (SE = {se:.5f})")
    print()
    print("Greeks (Black-Scholes):")
    for name, value in greeks.items():
        print(f"  {name:<8}{value:>10.5f}")
    print(f"\nImplied vol recovered from BS price: {iv:.6f} (input sigma = {sigma})")


if __name__ == "__main__":
    print_summary_table()
    binomial_convergence_plot()
    monte_carlo_variance_reduction_plot()
    print(f"\nPlots saved to {RESULTS_DIR}")
