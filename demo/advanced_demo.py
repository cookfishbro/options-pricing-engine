"""Generates the advanced-model figures used in the paper, saved to results/.

Figures produced:
  vol_smile.png            BS (flat) vs Heston vs Merton implied-vol smiles
  heston_surface.png       Heston implied-vol surface across strikes/maturities
  qmc_convergence.png      Sobol QMC vs pseudo-random MC convergence
  calibration_fit.png      Calibrated Heston smile vs synthetic market quotes
  return_distributions.png Terminal log-return densities by model (smile mechanism)
  greeks_profiles.png      Delta, Gamma, Vega vs spot
  exercise_boundary.png    American-put optimal early-exercise boundary
  smile_sensitivity.png    Heston smile vs correlation and vol-of-vol
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options_pricing import (
    bs_price, bs_greeks, heston_price, merton_price, implied_vol, implied_vol_smile,
    qmc_european_price, mc_price, calibrate_heston,
    heston_terminal_samples, merton_terminal_samples, american_exercise_boundary,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

S0, r = 100.0, 0.05


def vol_smile_plot():
    strikes = np.linspace(70, 130, 25)
    T = 1.0

    heston_p = dict(v0=0.04, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7)
    merton_p = dict(sigma=0.18, lam=0.7, muJ=-0.15, sigmaJ=0.2)

    iv_bs = implied_vol_smile(lambda K: bs_price(S0, K, T, r, 0.2, "call"),
                              S0, strikes, T, r, option_type="call")
    iv_heston = implied_vol_smile(lambda K: heston_price(S0, K, T, r, option_type="call", **heston_p),
                                  S0, strikes, T, r, option_type="call")
    iv_merton = implied_vol_smile(lambda K: merton_price(S0, K, T, r, option_type="call", **merton_p),
                                  S0, strikes, T, r, option_type="call")

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(strikes, iv_bs * 100, "--", label="Black-Scholes (constant vol)")
    plt.plot(strikes, iv_heston * 100, marker="o", ms=3, label="Heston (stochastic vol)")
    plt.plot(strikes, iv_merton * 100, marker="s", ms=3, label="Merton (jump-diffusion)")
    plt.axvline(S0, color="grey", lw=0.8, alpha=0.6)
    plt.xlabel("Strike")
    plt.ylabel("Black-Scholes implied volatility (%)")
    plt.title("Implied Volatility Smile by Model (T = 1 year)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "vol_smile.png"), dpi=150)
    plt.close()


def heston_surface_plot():
    strikes = np.linspace(75, 125, 20)
    maturities = np.linspace(0.1, 2.0, 20)
    heston_p = dict(v0=0.04, kappa=2.0, theta=0.05, sigma=0.6, rho=-0.7)

    K_grid, T_grid = np.meshgrid(strikes, maturities)
    iv = np.empty_like(K_grid)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            price = heston_price(S0, K, T, r, option_type="call", **heston_p)
            try:
                iv[i, j] = implied_vol(price, S0, K, T, r, "call") * 100
            except (ValueError, RuntimeError):
                iv[i, j] = np.nan

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K_grid, T_grid, iv, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied vol (%)")
    ax.set_title("Heston Implied Volatility Surface")
    ax.view_init(elev=22, azim=-130)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "heston_surface.png"), dpi=150)
    plt.close()


def qmc_convergence_plot():
    K, T, sigma = 100.0, 1.0, 0.2
    bs = bs_price(S0, K, T, r, sigma, "call")
    exponents = range(6, 16)
    sizes = [2 ** m for m in exponents]

    qmc_err, mc_err = [], []
    for n in sizes:
        q_errs, m_errs = [], []
        for seed in range(20):
            q_errs.append(abs(qmc_european_price(S0, K, T, r, sigma, "call", n_paths=n, seed=seed) - bs))
            price, _ = mc_price(S0, K, T, r, sigma, "call", n_paths=n,
                                antithetic=False, control_variate=False, seed=seed)
            m_errs.append(abs(price - bs))
        qmc_err.append(np.mean(q_errs))
        mc_err.append(np.mean(m_errs))

    sizes_arr = np.array(sizes)
    plt.figure(figsize=(7.5, 4.8))
    plt.loglog(sizes_arr, mc_err, marker="o", label="Pseudo-random MC")
    plt.loglog(sizes_arr, qmc_err, marker="s", label="Sobol QMC")
    plt.loglog(sizes_arr, mc_err[0] * (sizes_arr / sizes_arr[0]) ** -0.5, "k--",
               alpha=0.5, label=r"$O(n^{-1/2})$ reference")
    plt.loglog(sizes_arr, qmc_err[0] * (sizes_arr / sizes_arr[0]) ** -1.0, "k:",
               alpha=0.5, label=r"$O(n^{-1})$ reference")
    plt.xlabel("Number of samples")
    plt.ylabel("Mean absolute error vs Black-Scholes")
    plt.title("Convergence: Quasi-Monte Carlo vs Pseudo-random")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "qmc_convergence.png"), dpi=150)
    plt.close()


def calibration_fit_plot():
    true = dict(v0=0.045, kappa=1.5, theta=0.05, sigma=0.5, rho=-0.6)
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    maturities = [0.25, 0.5, 1.0]

    quotes, rng = [], np.random.default_rng(0)
    for T in maturities:
        for K in strikes:
            price = heston_price(S0, K, T, r, option_type="call", **true)
            price *= (1 + rng.normal(0, 0.002))  # small synthetic quote noise
            quotes.append((K, T, price, "call"))

    result = calibrate_heston(quotes, S0, r)
    calib = {k: result[k] for k in ("v0", "kappa", "theta", "sigma", "rho")}

    fig, axes = plt.subplots(1, len(maturities), figsize=(13, 4.2), sharey=True)
    for ax, T in zip(axes, maturities):
        market_iv, model_iv = [], []
        for K in strikes:
            mq = next(p for (kk, tt, p, _) in quotes if kk == K and tt == T)
            market_iv.append(implied_vol(mq, S0, K, T, r, "call") * 100)
            mp = heston_price(S0, K, T, r, option_type="call", **calib)
            model_iv.append(implied_vol(mp, S0, K, T, r, "call") * 100)
        ax.plot(strikes, market_iv, "o", label="Market (synthetic)")
        ax.plot(strikes, model_iv, "-", label="Calibrated Heston")
        ax.set_title(f"T = {T} yr")
        ax.set_xlabel("Strike")
    axes[0].set_ylabel("Implied vol (%)")
    axes[0].legend()
    fig.suptitle(f"Heston Calibration Fit (RMSE = {result['rmse']:.4f}, "
                 f"Feller {'OK' if result['feller_satisfied'] else 'violated'})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "calibration_fit.png"), dpi=150)
    plt.close()

    print("Calibration result:")
    for k, v in result.items():
        print(f"  {k:18}{v}")
    print("True parameters:", true)


def return_distribution_plot():
    """Terminal log-return densities under each model: the mechanism behind the
    smile. Heston is left-skewed (rho<0); Merton has a sharp peak and fat tails."""
    T = 1.0
    heston_p = dict(v0=0.04, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7)
    merton_p = dict(sigma=0.18, lam=0.7, muJ=-0.15, sigmaJ=0.2)

    n = 400_000
    bs_sigma = 0.2
    z = np.random.default_rng(0).standard_normal(n)
    bs_ret = (r - 0.5 * bs_sigma ** 2) * T + bs_sigma * np.sqrt(T) * z

    heston_ST = heston_terminal_samples(S0, T, r, n_paths=n, n_steps=200, seed=1, **heston_p)
    heston_ret = np.log(heston_ST / S0)
    merton_ST = merton_terminal_samples(S0, T, r, n_paths=n, seed=2, **merton_p)
    merton_ret = np.log(merton_ST / S0)

    grid = np.linspace(-1.0, 0.7, 500)
    plt.figure(figsize=(8, 4.8))
    for ret, label in [(bs_ret, "Black-Scholes (Gaussian)"),
                       (heston_ret, "Heston (stochastic vol)"),
                       (merton_ret, "Merton (jump-diffusion)")]:
        kde = gaussian_kde(ret)
        plt.plot(grid, kde(grid), label=label)
    plt.xlabel(r"Terminal log-return $\ln(S_T/S_0)$")
    plt.ylabel("Density")
    plt.title("Risk-Neutral Return Distributions (T = 1 year)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "return_distributions.png"), dpi=150)
    plt.close()


def greeks_profile_plot():
    spots = np.linspace(60, 140, 100)
    K, T, sigma = 100.0, 1.0, 0.2

    call = [bs_greeks(s, K, T, r, sigma, "call") for s in spots]
    put = [bs_greeks(s, K, T, r, sigma, "put") for s in spots]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(spots, [g["delta"] for g in call], label="Call")
    axes[0].plot(spots, [g["delta"] for g in put], label="Put")
    axes[0].set_title("Delta")
    axes[0].legend()
    axes[1].plot(spots, [g["gamma"] for g in call])
    axes[1].set_title("Gamma (call = put)")
    axes[2].plot(spots, [g["vega"] for g in call])
    axes[2].set_title("Vega (call = put)")
    for ax in axes:
        ax.axvline(K, color="grey", lw=0.8, alpha=0.6)
        ax.set_xlabel("Spot price")
    fig.suptitle("Black-Scholes Greeks vs Spot (K = 100, T = 1, sigma = 20%)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "greeks_profiles.png"), dpi=150)
    plt.close()


def exercise_boundary_plot():
    K, T, sigma = 100.0, 1.0, 0.2
    times, boundary = american_exercise_boundary(S0, K, T, r, sigma, n_steps=600, option_type="put")

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(times, boundary, label="Exercise boundary $S^*(t)$")
    plt.axhline(K, color="grey", ls="--", lw=1, label="Strike $K$")
    plt.fill_between(times, 0, boundary, alpha=0.15, label="Exercise region")
    plt.ylim(boundary[~np.isnan(boundary)].min() - 5, K + 5)
    plt.xlabel("Time $t$ (years)")
    plt.ylabel("Spot price")
    plt.title("American Put: Optimal Early-Exercise Boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exercise_boundary.png"), dpi=150)
    plt.close()


def smile_sensitivity_plot():
    strikes = np.linspace(75, 125, 25)
    T = 1.0
    base = dict(v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.5)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for rho in [-0.9, -0.5, 0.0, 0.5]:
        p = {**base, "rho": rho}
        iv = implied_vol_smile(lambda K: heston_price(S0, K, T, r, option_type="call", **p),
                               S0, strikes, T, r, option_type="call")
        axes[0].plot(strikes, iv * 100, marker=".", ms=4, label=fr"$\rho={rho}$")
    axes[0].set_title(r"Effect of correlation $\rho$ (skew)")
    axes[0].set_xlabel("Strike")
    axes[0].set_ylabel("Implied vol (%)")
    axes[0].legend()

    for xi in [0.1, 0.3, 0.6, 0.9]:
        p = {**base, "sigma": xi}
        iv = implied_vol_smile(lambda K: heston_price(S0, K, T, r, option_type="call", **p),
                               S0, strikes, T, r, option_type="call")
        axes[1].plot(strikes, iv * 100, marker=".", ms=4, label=fr"$\xi={xi}$")
    axes[1].set_title(r"Effect of vol-of-vol $\xi$ (convexity)")
    axes[1].set_xlabel("Strike")
    axes[1].legend()

    fig.suptitle("Heston Smile Sensitivity to Parameters")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "smile_sensitivity.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Generating volatility smile...")
    vol_smile_plot()
    print("Generating Heston surface...")
    heston_surface_plot()
    print("Generating QMC convergence...")
    qmc_convergence_plot()
    print("Generating return distributions...")
    return_distribution_plot()
    print("Generating Greeks profiles...")
    greeks_profile_plot()
    print("Generating exercise boundary...")
    exercise_boundary_plot()
    print("Generating smile sensitivity...")
    smile_sensitivity_plot()
    print("Generating calibration fit...")
    calibration_fit_plot()
    print(f"\nFigures saved to {RESULTS_DIR}")
