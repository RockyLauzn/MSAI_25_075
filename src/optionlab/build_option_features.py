"""Build daily approximately 30-DTE ATM implied-volatility features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_atm_iv_features(quotes: pd.DataFrame, max_relative_spread: float = 0.35) -> pd.DataFrame:
    required = {"quote_time", "expiration", "strike", "right", "bid", "ask", "ask_size",
                "implied_volatility", "underlying_price", "contract_symbol"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Quotes missing columns: {', '.join(sorted(missing))}")
    frame = quotes.copy()
    frame["session_date"] = pd.to_datetime(frame["quote_time"], utc=True).dt.tz_convert(
        "America/New_York").dt.date
    frame["expiration"] = pd.to_datetime(frame["expiration"]).dt.date
    frame["dte"] = (pd.to_datetime(frame["expiration"]) - pd.to_datetime(frame["session_date"].astype(str))).dt.days
    for col in ("strike", "bid", "ask", "ask_size", "implied_volatility", "underlying_price"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    mid = (frame["bid"] + frame["ask"]) / 2
    spread = (frame["ask"] - frame["bid"]) / mid.where(mid > 0)
    frame = frame.loc[
        frame["dte"].between(25, 40) & (frame["bid"] > 0) & (frame["ask"] >= frame["bid"])
        & (frame["ask_size"] >= 1) & (frame["implied_volatility"] > 0)
        & (spread <= max_relative_spread)
    ].copy()
    keys = ["session_date", "expiration", "strike"]
    calls = frame.loc[frame["right"].eq("C")].set_index(keys)
    puts = frame.loc[frame["right"].eq("P")].set_index(keys)
    pairs = calls.add_suffix("_call").join(puts.add_suffix("_put"), how="inner").reset_index()
    if pairs.empty:
        return pd.DataFrame(columns=["date", "atm_iv_30"])
    pairs["dte"] = (pd.to_datetime(pairs["expiration"]) - pd.to_datetime(pairs["session_date"].astype(str))).dt.days
    pairs["expiry_distance"] = (pairs["dte"] - 30).abs()
    spot = pairs[["underlying_price_call", "underlying_price_put"]].mean(axis=1)
    pairs["atm_distance"] = (pairs["strike"] - spot).abs() / spot
    pairs["atm_iv_30"] = (pairs["implied_volatility_call"] + pairs["implied_volatility_put"]) / 2
    selected = pairs.sort_values(["session_date", "expiry_distance", "atm_distance"]).drop_duplicates(
        "session_date", keep="first")
    result = selected.rename(columns={"session_date": "date"})[
        ["date", "atm_iv_30", "expiration", "dte", "strike", "atm_distance"]
    ].copy()
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-relative-spread", type=float, default=0.35)
    args = parser.parse_args()
    columns = ["quote_time", "expiration", "strike", "right", "bid", "ask", "ask_size",
               "implied_volatility", "underlying_price", "contract_symbol"]
    quotes = (pd.read_parquet(args.quotes, columns=columns) if args.quotes.suffix.lower() == ".parquet"
              else pd.read_csv(args.quotes, usecols=columns))
    features = build_atm_iv_features(quotes, args.max_relative_spread)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features):,} daily IV rows to {args.output}")


if __name__ == "__main__":
    main()
