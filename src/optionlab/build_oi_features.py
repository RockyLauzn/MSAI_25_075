"""Build conservative, one-session-lagged SPY open-interest features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_oi_features(quotes: pd.DataFrame, min_dte: int = 21, max_dte: int = 40,
                      moneyness_band: float = 0.05, lag_sessions: int = 1) -> pd.DataFrame:
    """Aggregate near-ATM OI, then lag by trading sessions to avoid timing leakage.

    OI provenance within this public mirror is not documented. The extra session
    lag is deliberately conservative; it can reduce signal freshness.
    """
    required = {"quote_time", "expiration", "right", "strike", "underlying_price",
                "open_interest", "volume"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    q = quotes.loc[:, list(required)].copy()
    q["date"] = pd.to_datetime(q["quote_time"], utc=True).dt.tz_convert("America/New_York").dt.date
    q["expiration"] = pd.to_datetime(q["expiration"], errors="coerce").dt.date
    q["dte"] = (pd.to_datetime(q["expiration"]) - pd.to_datetime(q["date"].astype(str))).dt.days
    q["right"] = q["right"].astype(str).str.upper()
    q["moneyness"] = q["strike"] / q["underlying_price"] - 1.0
    q["open_interest"] = pd.to_numeric(q["open_interest"], errors="coerce").fillna(0).clip(lower=0)
    q["volume"] = pd.to_numeric(q["volume"], errors="coerce").fillna(0).clip(lower=0)
    q = q.loc[q["dte"].between(min_dte, max_dte)
              & q["moneyness"].abs().le(moneyness_band)
              & q["right"].isin(["C", "P"])]
    if q.empty:
        raise ValueError("No option rows remain after DTE and moneyness filters")
    q["call_oi"] = q["open_interest"].where(q["right"].eq("C"), 0)
    q["put_oi"] = q["open_interest"].where(q["right"].eq("P"), 0)
    q["call_volume"] = q["volume"].where(q["right"].eq("C"), 0)
    q["put_volume"] = q["volume"].where(q["right"].eq("P"), 0)
    daily = q.groupby("date", as_index=False).agg(
        call_oi=("call_oi", "sum"), put_oi=("put_oi", "sum"),
        call_volume=("call_volume", "sum"), put_volume=("put_volume", "sum"),
        contract_count=("open_interest", "size"),
    ).sort_values("date")
    total = daily.call_oi + daily.put_oi
    daily["oi_call_log"] = np.log1p(daily.call_oi)
    daily["oi_put_log"] = np.log1p(daily.put_oi)
    daily["oi_pc_ratio"] = daily.put_oi / daily.call_oi.replace(0, np.nan)
    daily["oi_total_log"] = np.log1p(total)
    daily["oi_total_change_5d"] = total.pct_change(5, fill_method=None)
    daily["oi_volume_to_oi"] = (daily.call_volume + daily.put_volume) / total.replace(0, np.nan)
    feature_cols = ["oi_call_log", "oi_put_log", "oi_pc_ratio", "oi_total_log",
                    "oi_total_change_5d", "oi_volume_to_oi"]
    # Shift over observed option-chain sessions, not calendar days.
    daily[feature_cols] = daily[feature_cols].shift(lag_sessions)
    return daily.loc[:, ["date", *feature_cols]].dropna().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-dte", type=int, default=21)
    parser.add_argument("--max-dte", type=int, default=40)
    parser.add_argument("--moneyness-band", type=float, default=0.05)
    parser.add_argument("--lag-sessions", type=int, default=1)
    args = parser.parse_args()
    quotes = pd.read_parquet(args.quotes, columns=["quote_time", "expiration", "right", "strike",
                                                    "underlying_price", "open_interest", "volume"])
    features = build_oi_features(quotes, args.min_dte, args.max_dte,
                                 args.moneyness_band, args.lag_sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features):,} daily OI feature rows to {args.output}")


if __name__ == "__main__":
    main()
