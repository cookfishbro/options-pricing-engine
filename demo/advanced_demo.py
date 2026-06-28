"""Advanced-model figures, parameterised by the REAL calibrated models.

Reads the calibrated Heston/Merton parameters and reference SPY contract from
results/real_summary.json (produced by real_data_demo.py), so the model-behaviour
figures here depict the models as actually fitted to market data.

Figures produced:
  heston_surface.png       Calibrated-Heston implied-vol surface
  qmc_convergence.png      Sobol QMC vs pseudo-random MC convergence
  return_distributions.png Terminal log-return densities (calibrated models)
  greeks_profiles.png      Delta, Gamma, Vega vs spot for the real contract
  exercise_boundary.png    American-put optimal early-exercise boundary
  smile_sensitivity.png    Heston smile vs correlation and vol-of-vol

Run:  python demo/real_data_demo.py   (first)
      python demo/advanced_demo.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options_pricing import (
    bs_price, bs_greeks, heston_price, merton_price, implied_vol, implied_vol_smile,
    qmc_european_price, mc_price, heston_terminal_samples, merton_terminal_samples,
    american_exercise_boundary,
)

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
    return s["meta"], s["heston"], s["merton"], s["cross_validation"]["contract"]


META, HESTON, MERTON, CONTRACT = load_reference()
S0 = CONTRACT["S0"]
K_REF, T_REF = CONTRACT["K"], CONTRACT["T"]
R, Q, SIGMA_ATM = CONTRACT["r"], CONTRACT["q"], CONTRACT["atm_iv"]
HP = {k: HESTON[k] for k in ("v0", "kappa", "theta", "sigma", "rho")}
MP = {k: MERTON[k] for k in ("sigma", "lam", "muJ", "sigmaJ")}
TICK = META["ticker"]


def heston_surface_plot():
    strikes = np.linspace(0.80 * S0, 1.20 * S0, 22)
    maturities = np.linspace(0.08, 1.0, 20)
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    iv = np.empty_like(K_grid)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            ot = "call" if K >= S0 else "put"
            price = heston_price(S0, K, T, R, option_type=ot, q=Q, **HP)
            try:
                iv[i, j] = implied_vol(price, S0, K, T, R, ot, Q) * 100
            except (ValueError, RuntimeError):
                iv[i, j] = np.nan

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K_grid / S0, T_grid, iv, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Moneyness (K / S)")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied vol (%)")
    ax.set_title(f"Calibrated-Heston Implied Volatility Surface ({TICK})")
    ax.view_init(elev=22, azim=-130)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "heston_surface.png"), dpi=150)
    plt.close()


def qmc_convergence_plot():
    bs = bs_price(S0, K_REF, T_REF, R, SIGMA_ATM, "call", Q)
    sizes = [2 ** m for m in range(6, 16)]
    qmc_err, mc_err = [], []
    for n in sizes:
        q_errs, m_errs = [], []
        for seed in range(20):
            q_errs.append(abs(qmc_european_price(S0, K_REF, T_REF, R, SIGMA_ATM, "call", q=Q, n_paths=n, seed=seed) - bs))
            price, _ = mc_price(S0, K_REF, T_REF, R, SIGMA_ATM, "call", n_paths=n, q=Q,
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
    plt.title(f"Convergence: Quasi-Monte Carlo vs Pseudo-random ({TICK} ATM call)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "qmc_convergence.png"), dpi=150)
    plt.close()


def return_distribution_plot():
    T = T_REF
    n = 400_000
    z = np.random.default_rng(0).standard_normal(n)
    bs_ret = (R - Q - 0.5 * SIGMA_ATM ** 2) * T + SIGMA_ATM * np.sqrt(T) * z
    heston_ret = np.log(heston_terminal_samples(S0, T, R, q=Q, n_paths=n, n_steps=200, seed=1, **HP) / S0)
    merton_ret = np.log(merton_terminal_samples(S0, T, R, q=Q, n_paths=n, seed=2, **MP) / S0)

    lo = min(bs_ret.min(), heston_ret.min(), merton_ret.min())
    hi = max(bs_ret.max(), heston_ret.max(), merton_ret.max())
    grid = np.linspace(lo, hi, 500)
    plt.figure(figsize=(8, 4.8))
    for ret, label in [(bs_ret, "Black-Scholes (Gaussian)"),
                       (heston_ret, "Calibrated Heston"),
                       (merton_ret, "Calibrated Merton")]:
        plt.plot(grid, gaussian_kde(ret)(grid), label=label)
    plt.xlabel(r"Terminal log-return $\ln(S_T/S_0)$")
    plt.ylabel("Density")
    plt.title(f"Risk-Neutral Return Distributions ({TICK}, T = {T:.2f} yr)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "return_distributions.png"), dpi=150)
    plt.close()


def greeks_profile_plot():
    spots = np.linspace(0.6 * S0, 1.4 * S0, 100)
    call = [bs_greeks(s, K_REF, T_REF, R, SIGMA_ATM, "call", Q) for s in spots]
    put = [bs_greeks(s, K_REF, T_REF, R, SIGMA_ATM, "put", Q) for s in spots]

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
        ax.axvline(K_REF, color="grey", lw=0.8, alpha=0.6)
        ax.set_xlabel("Spot price")
    fig.suptitle(f"Black-Scholes Greeks vs Spot ({TICK}: K={K_REF:.0f}, "
                 f"T={T_REF:.2f}, IV={SIGMA_ATM*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "greeks_profiles.png"), dpi=150)
    plt.close()


def exercise_boundary_plot():
    times, boundary = american_exercise_boundary(S0, K_REF, T_REF, R, SIGMA_ATM,
                                                  n_steps=600, option_type="put", q=Q)
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(times, boundary, label="Exercise boundary $S^*(t)$")
    plt.axhline(K_REF, color="grey", ls="--", lw=1, label="Strike $K$")
    plt.fill_between(times, 0, boundary, alpha=0.15, label="Exercise region")
    plt.ylim(np.nanmin(boundary) - 0.03 * S0, K_REF + 0.03 * S0)
    plt.xlabel("Time $t$ (years)")
    plt.ylabel("Spot price")
    plt.title(f"American Put: Optimal Early-Exercise Boundary ({TICK})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exercise_boundary.png"), dpi=150)
    plt.close()


def smile_sensitivity_plot():
    strikes = np.linspace(0.80 * S0, 1.20 * S0, 25)
    T = T_REF
    base = dict(HP)

    # Heston prices satisfy put-call parity, so pricing every strike as a call
    # and inverting as a call recovers the correct implied-vol smile.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for rho in [-0.9, -0.6, -0.3, 0.0]:
        p = {**base, "rho": rho}
        iv = implied_vol_smile(lambda K: heston_price(S0, K, T, R, option_type="call", q=Q, **p),
                               S0, strikes, T, R, option_type="call")
        axes[0].plot(strikes / S0, iv * 100, marker=".", ms=4, label=fr"$\rho={rho}$")
    axes[0].set_title(r"Effect of correlation $\rho$ (skew)")
    axes[0].set_xlabel("Moneyness (K / S)")
    axes[0].set_ylabel("Implied vol (%)")
    axes[0].legend()

    for xi in [0.3, 0.6, 0.9, 1.2]:
        p = {**base, "sigma": xi}
        iv = implied_vol_smile(lambda K: heston_price(S0, K, T, R, option_type="call", q=Q, **p),
                               S0, strikes, T, R, option_type="call")
        axes[1].plot(strikes / S0, iv * 100, marker=".", ms=4, label=fr"$\xi={xi}$")
    axes[1].set_title(r"Effect of vol-of-vol $\xi$ (convexity)")
    axes[1].set_xlabel("Moneyness (K / S)")
    axes[1].legend()

    fig.suptitle(f"Heston Smile Sensitivity to Parameters (around {TICK} calibration)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "smile_sensitivity.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Generating calibrated-Heston surface...")
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
    print(f"\nFigures saved to {RESULTS_DIR}")
