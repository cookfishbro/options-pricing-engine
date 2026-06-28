"""Fetch a real option chain from Yahoo Finance and write a clean snapshot.

The snapshot (``options.csv`` + ``meta.json``) is committed to the repository so
that all downstream results are reproducible even though live quotes change.

Methodology (standard market practice):
  * Underlying spot and option bid/ask quotes from Yahoo Finance.
  * Risk-free rate ``r`` from the 13-week US Treasury-bill yield (``^IRX``).
  * Dividend yield ``q`` from the underlying's trailing annual dividend yield.
  * Per-expiry forward ``F = S e^{(r-q)T}``; the put-call-parity forward implied
    from the quotes is also recorded in ``meta.json`` as a data-quality check.
  * Only liquid out-of-the-money quotes are kept (positive bid/ask, tight
    relative spread, minimum open interest, moneyness band): puts below the
    forward and calls above it. These OTM contracts are the liquid instruments
    that define the volatility smile.

Run:  python data/fetch_data.py [TICKER]
"""

import json
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))

# --- filters ---------------------------------------------------------------
MIN_DAYS, MAX_DAYS = 20, 130      # maturity window (calendar days)
MAX_EXPIRIES = 6                  # number of expiries to keep
MONEYNESS_LO, MONEYNESS_HI = 0.85, 1.15
MAX_REL_SPREAD = 0.20             # (ask - bid) / mid
MIN_OPEN_INTEREST = 25
PARITY_BAND = 0.05                # +/-5% of spot for the parity cross-check


def _clean_side(df):
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])].copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    return df[["strike", "bid", "ask", "mid", "openInterest"]]


def _parity_forward(calls, puts, spot, r, T):
    """Median put-call-parity forward across near-ATM strikes (quality check)."""
    merged = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    band = merged[(merged["strike"] > spot * (1 - PARITY_BAND)) &
                  (merged["strike"] < spot * (1 + PARITY_BAND))]
    if len(band) < 3:
        return None
    fwd = band["strike"] + np.exp(r * T) * (band["mid_c"] - band["mid_p"])
    return float(np.median(fwd))


def fetch(ticker="SPY"):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    spot = float(hist["Close"].iloc[-1])
    asof = hist.index[-1].date().isoformat()

    r = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100.0
    q = tk.info.get("trailingAnnualDividendYield") or 0.0
    q = float(q)

    today = dt.date.fromisoformat(asof)
    expiries = [(e, (dt.date.fromisoformat(e) - today).days) for e in tk.options]
    expiries = [(e, d) for e, d in expiries if MIN_DAYS <= d <= MAX_DAYS]
    if not expiries:
        raise RuntimeError("No expiries in the maturity window.")
    idx = np.linspace(0, len(expiries) - 1, min(MAX_EXPIRIES, len(expiries)))
    expiries = [expiries[int(round(i))] for i in idx]

    rows, parity_check = [], {}
    for expiry, days in expiries:
        T = days / 365.0
        chain = tk.option_chain(expiry)
        calls, puts = _clean_side(chain.calls), _clean_side(chain.puts)

        # Use the put-call-parity-implied forward (market-consistent) when
        # available; fall back to the dividend forward otherwise. The implied
        # carry q := r - ln(F/S)/T makes the put and call wings line up at the
        # money regardless of dividends, repo, or quote timing.
        F_div = spot * np.exp((r - q) * T)
        F_parity = _parity_forward(calls, puts, spot, r, T)
        F = F_parity if F_parity is not None else F_div
        q_eff = r - np.log(F / spot) / T
        parity_check[expiry] = {"forward_dividend": round(F_div, 2),
                                "forward_parity": None if F_parity is None else round(F_parity, 2),
                                "q_implied": round(float(q_eff), 5)}

        for side, df, otm_mask in (
            ("put", puts, puts["strike"] < F),
            ("call", calls, calls["strike"] >= F),
        ):
            sub = df[otm_mask].copy()
            sub = sub[(sub["strike"] >= spot * MONEYNESS_LO) &
                      (sub["strike"] <= spot * MONEYNESS_HI)]
            sub = sub[(sub["openInterest"] >= MIN_OPEN_INTEREST) &
                      ((sub["ask"] - sub["bid"]) / sub["mid"] <= MAX_REL_SPREAD)]
            for _, row in sub.iterrows():
                rows.append({
                    "asof": asof, "ticker": ticker, "expiry": expiry,
                    "T": round(T, 6), "type": side, "strike": float(row["strike"]),
                    "bid": float(row["bid"]), "ask": float(row["ask"]),
                    "mid": float(row["mid"]), "spot": round(spot, 4),
                    "forward": round(float(F), 4), "r": round(r, 6),
                    "q": round(float(q_eff), 6),
                    "open_interest": int(row["openInterest"]),
                })

    out = pd.DataFrame(rows).sort_values(["T", "strike"]).reset_index(drop=True)
    out.to_csv(os.path.join(HERE, "options.csv"), index=False)
    meta = {
        "ticker": ticker, "asof": asof, "spot": round(spot, 4),
        "risk_free_rate_irx": round(r, 6), "trailing_dividend_yield": round(q, 6),
        "n_quotes": len(out), "n_expiries": out["expiry"].nunique(),
        "expiries": sorted(out["expiry"].unique().tolist()),
        "forward_and_implied_carry": parity_check,
        "source": "Yahoo Finance (yfinance)",
        "fetched_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(HERE, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(out)} quotes across {out['expiry'].nunique()} expiries to data/options.csv")
    print(json.dumps(meta, indent=2))
    return out, meta


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else "SPY")
