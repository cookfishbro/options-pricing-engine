"""Empirical study on REAL market data (SPY options from Yahoo Finance).

Loads the committed snapshot in data/ (run data/fetch_data.py to refresh it),
computes the market implied-volatility skew, calibrates the Heston and Merton
models to the liquid quotes, and produces the real-data figures used in the
paper. Calibrated parameters and a numerical summary are written to results/ so
the other demo scripts and the paper draw on the same real, fitted parameters.

Run:  python demo/real_data_demo.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options_pricing import (
    bs_price, bs_greeks, binomial_price, mc_price, qmc_european_price,
    implied_vol, heston_price, heston_mc_price, merton_price, merton_mc_price,
    longstaff_schwartz_price, calibrate_heston, calibrate_merton,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_PER_EXPIRY = 18   # thin each expiry's strikes to keep calibration fast


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "options.csv"))
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    # Market implied vol and BS vega (dC/dsigma) for each quote.
    ivs, vegas = [], []
    for _, row in df.iterrows():
        iv = implied_vol(row["mid"], row["spot"], row["strike"], row["T"],
                         row["r"], row["type"], row["q"])
        ivs.append(iv)
        vegas.append(bs_greeks(row["spot"], row["strike"], row["T"], row["r"],
                               iv, row["type"], row["q"])["vega"] * 100.0)
    df["iv"], df["vega"] = ivs, vegas
    return df, meta


def thin(df):
    parts = []
    for _, g in df.groupby("expiry"):
        g = g.sort_values("strike")
        if len(g) > MAX_PER_EXPIRY:
            idx = np.linspace(0, len(g) - 1, MAX_PER_EXPIRY).round().astype(int)
            g = g.iloc[np.unique(idx)]
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)


def calibrate(df, meta):
    S0 = meta["spot"]
    cal = thin(df)
    # Per-quote (r, q): q is the maturity-dependent carry implied from the forward.
    quotes = list(zip(cal["strike"], cal["T"], cal["mid"], cal["type"], cal["r"], cal["q"]))
    weights = 1.0 / np.maximum(cal["vega"].values, 1e-2)  # ~ implied-vol residuals

    print(f"Calibrating to {len(quotes)} liquid quotes ({df['expiry'].nunique()} expiries)...")
    heston = calibrate_heston(quotes, S0, weights=weights)
    merton = calibrate_merton(quotes, S0, weights=weights)

    # Report fit quality in implied-vol points (more interpretable than $ RMSE).
    def vol_rmse(price_of):
        errs = []
        for _, row in cal.iterrows():
            K, T, ot, ri, qi = row["strike"], row["T"], row["type"], row["r"], row["q"]
            model_iv = implied_vol(price_of(K, T, ot, ri, qi), S0, K, T, ri, ot, qi)
            errs.append(model_iv - row["iv"])
        return float(np.sqrt(np.mean(np.square(errs))))

    heston["vol_rmse"] = vol_rmse(
        lambda K, T, ot, ri, qi: heston_price(S0, K, T, ri, heston["v0"], heston["kappa"],
                                              heston["theta"], heston["sigma"], heston["rho"], ot, qi))
    merton["vol_rmse"] = vol_rmse(
        lambda K, T, ot, ri, qi: merton_price(S0, K, T, ri, merton["sigma"], merton["lam"],
                                              merton["muJ"], merton["sigmaJ"], ot, qi))
    return heston, merton, S0


def market_smile_plot(df, meta):
    fig, ax = plt.subplots(figsize=(8, 5))
    expiries = sorted(df["expiry"].unique())
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(expiries)))
    for color, exp in zip(cmap, expiries):
        g = df[df["expiry"] == exp].sort_values("strike")
        ax.plot(g["strike"] / g["spot"], g["iv"] * 100, marker=".", ms=5,
                color=color, label=f"{exp} (T={g['T'].iloc[0]:.2f})")
    ax.axvline(1.0, color="grey", lw=0.8, alpha=0.6)
    ax.set_xlabel("Moneyness (strike / spot)")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"{meta['ticker']} Implied Volatility Skew (as of {meta['asof']}, "
                 f"spot {meta['spot']:.2f})")
    ax.legend(fontsize=8, title="Expiry")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "market_smile.png"), dpi=150)
    plt.close()


def calibration_fit_plot(df, heston, merton, S0):
    expiries = sorted(df["expiry"].unique())
    chosen = [expiries[0], expiries[len(expiries) // 2], expiries[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for ax, exp in zip(axes, chosen):
        g = df[df["expiry"] == exp].sort_values("strike")
        T, ri, qi, F = g["T"].iloc[0], g["r"].iloc[0], g["q"].iloc[0], g["forward"].iloc[0]
        otm = lambda K: "call" if K >= F else "put"
        ax.plot(g["strike"] / S0, g["iv"] * 100, "o", ms=4, color="black",
                label="Market", zorder=3)
        grid = np.linspace(g["strike"].min(), g["strike"].max(), 40)
        for label, price_func, style in (
            ("Heston", lambda K: heston_price(S0, K, T, ri, heston["v0"], heston["kappa"],
                                              heston["theta"], heston["sigma"], heston["rho"],
                                              otm(K), qi), "-"),
            ("Merton", lambda K: merton_price(S0, K, T, ri, merton["sigma"], merton["lam"],
                                              merton["muJ"], merton["sigmaJ"], otm(K), qi), "--"),
        ):
            iv_curve = [implied_vol(price_func(K), S0, K, T, ri, otm(K), qi) * 100 for K in grid]
            ax.plot(grid / S0, iv_curve, style, label=label)
        ax.set_title(f"{exp}  (T = {T:.2f} yr)")
        ax.set_xlabel("Moneyness (K / S)")
    axes[0].set_ylabel("Implied volatility (%)")
    axes[0].legend()
    fig.suptitle(f"Heston & Merton Calibrated to {df['ticker'].iloc[0]} Options "
                 f"(Heston vol-RMSE {heston['vol_rmse']*100:.2f}%, "
                 f"Merton {merton['vol_rmse']*100:.2f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "calibration_fit_real.png"), dpi=150)
    plt.close()


def cross_validation_table(df, heston, merton, S0):
    """Reprice a real near-ATM contract every available way (all real inputs)."""
    expiries = sorted(df["expiry"].unique())
    exp = expiries[len(expiries) // 2]
    g = df[df["expiry"] == exp]
    atm = g.iloc[(g["strike"] - S0).abs().argmin()]
    K, T, sigma = float(atm["strike"]), float(atm["T"]), float(atm["iv"])
    r, q = float(atm["r"]), float(atm["q"])

    hp = dict(v0=heston["v0"], kappa=heston["kappa"], theta=heston["theta"],
              sigma=heston["sigma"], rho=heston["rho"])
    mp = dict(sigma=merton["sigma"], lam=merton["lam"], muJ=merton["muJ"], sigmaJ=merton["sigmaJ"])

    heston_mc, heston_se = heston_mc_price(S0, K, T, r, option_type="call", q=q,
                                           n_paths=200_000, n_steps=200, seed=1, **hp)
    merton_mc, merton_se = merton_mc_price(S0, K, T, r, option_type="call", q=q,
                                           n_paths=500_000, seed=2, **mp)
    lsm, lsm_se = longstaff_schwartz_price(S0, K, T, r, sigma, "put", q=q,
                                           n_paths=100_000, n_steps=50, seed=3)
    table = {
        "contract": {"ticker": df["ticker"].iloc[0], "S0": S0, "K": K, "T": round(T, 4),
                     "r": r, "q": q, "atm_iv": round(sigma, 4), "expiry": exp},
        "european_call": {
            "black_scholes": round(bs_price(S0, K, T, r, sigma, "call", q), 4),
            "binomial_500": round(binomial_price(S0, K, T, r, sigma, 500, "call", q=q), 4),
            "monte_carlo_200k": [round(mc_price(S0, K, T, r, sigma, "call", n_paths=200_000, q=q, seed=4)[0], 4),
                                 round(mc_price(S0, K, T, r, sigma, "call", n_paths=200_000, q=q, seed=4)[1], 4)],
            "qmc_sobol": round(qmc_european_price(S0, K, T, r, sigma, "call", q=q, n_paths=65536, seed=5), 4),
        },
        "heston_call": {"fourier": round(heston_price(S0, K, T, r, option_type="call", q=q, **hp), 4),
                        "monte_carlo": [round(heston_mc, 4), round(heston_se, 4)]},
        "merton_call": {"series": round(merton_price(S0, K, T, r, option_type="call", q=q, **mp), 4),
                        "monte_carlo": [round(merton_mc, 4), round(merton_se, 4)]},
        "american_put": {"binomial": round(binomial_price(S0, K, T, r, sigma, 1000, "put", american=True, q=q), 4),
                         "lsm": [round(lsm, 4), round(lsm_se, 4)]},
    }
    return table


def main():
    df, meta = load_data()
    print(f"Loaded {len(df)} {meta['ticker']} quotes, asof {meta['asof']}, spot {meta['spot']}")
    heston, merton, S0 = calibrate(df, meta)

    market_smile_plot(df, meta)
    calibration_fit_plot(df, heston, merton, S0)
    table = cross_validation_table(df, heston, merton, S0)

    native = lambda o: o.item() if hasattr(o, "item") else str(o)
    summary = {"meta": meta, "heston": heston, "merton": merton,
               "cross_validation": table}
    with open(os.path.join(RESULTS_DIR, "real_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=native)
    # Also save just the calibrated params for the other demo scripts to consume.
    with open(os.path.join(RESULTS_DIR, "calibrated_params.json"), "w") as f:
        json.dump({"meta": meta, "heston": heston, "merton": merton}, f, indent=2, default=native)

    print("\n=== Heston calibration (real SPY data) ===")
    print(json.dumps(heston, indent=2))
    print("\n=== Merton calibration (real SPY data) ===")
    print(json.dumps(merton, indent=2))
    print("\n=== Cross-validation on real ATM contract ===")
    print(json.dumps(table, indent=2))
    print(f"\nFigures + summary saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
