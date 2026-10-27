"""Conservative long ATM straddle backtest driven by volatility forecasts.

This is an exploratory research baseline. It enters at asks, exits at bids,
uses one standard 100-share contract per leg, and never opens overlapping trades.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from .option_chain import validate_quote_snapshots


def _quote_sessions(quotes: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    required = {
        "quote_time", "contract_symbol", "underlying", "expiration", "right", "strike",
        "bid", "ask", "ask_size", "underlying_price", "contract_multiplier",
        "deliverable_shares", "deliverable_cash", "implied_volatility",
    }
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Quotes need fields for backtesting: {', '.join(sorted(missing))}")
    frame = validate_quote_snapshots(quotes)
    local_time = frame["quote_time"].dt.tz_convert("America/New_York")
    frame["session_date"] = local_time.dt.date
    frame["minute_of_day"] = local_time.dt.hour * 60 + local_time.dt.minute
    frame["expiration"] = pd.to_datetime(frame["expiration"], errors="raise").dt.date
    for column in ("strike", "bid", "ask", "ask_size", "underlying_price",
                   "contract_multiplier", "deliverable_shares", "deliverable_cash",
                   "implied_volatility"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["strike", "bid", "ask", "ask_size", "underlying_price",
                             "contract_multiplier", "deliverable_shares", "deliverable_cash",
                             "implied_volatility"]].to_numpy()).all():
        raise ValueError("Quote data contains non-finite numeric values")
    frame = frame.sort_values("quote_time")
    frame = frame.loc[frame["minute_of_day"].between(9 * 60 + 30, 16 * 60)]
    return frame, sorted(frame["session_date"].unique())


def _snapshot(frame: pd.DataFrame, day, start_minute: int, end_minute: int) -> pd.DataFrame:
    window = frame.loc[
        (frame["session_date"] == day)
        & (frame["minute_of_day"] >= start_minute)
        & (frame["minute_of_day"] <= end_minute)
    ]
    return window.drop_duplicates("contract_symbol", keep="last")


def backtest_long_atm_straddle(
    predictions: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    hold_days: int = 20,
    target_dte: int = 30,
    min_dte: int = 25,
    max_dte: int = 40,
    edge_threshold: float = 0.10,
    fee_per_contract: float = 0.65,
    max_relative_spread: float = 0.35,
    quote_mode: str = "intraday",
) -> tuple[dict, pd.DataFrame]:
    """Test long ATM straddles when predicted RV exceeds paired ATM IV.

    Signal: ridge forecast > mean(call IV, put IV) * (1 + edge_threshold).
    The forecast is formed after session t's close. Entry quotes are taken from
    09:30-09:45 ET on the next session; exit quotes are taken from 15:45-16:00 ET
    on session t+hold_days. Entry uses asks and exit uses bids. Trades use
    standard 100-share contracts and do not overlap.
    """
    if hold_days < 1 or not 0 <= min_dte <= target_dte <= max_dte:
        raise ValueError("Require hold_days >= 1 and min_dte <= target_dte <= max_dte")
    if edge_threshold < 0 or fee_per_contract < 0 or max_relative_spread <= 0:
        raise ValueError("Threshold and fees must be nonnegative; spread limit must be positive")
    if not {"date", "ridge_prediction"}.issubset(predictions.columns):
        raise ValueError("Predictions require date and ridge_prediction columns")

    if quote_mode not in {"intraday", "eod"}:
        raise ValueError("quote_mode must be intraday or eod")
    quote_frame, _ = _quote_sessions(quotes)
    quote_by_day = {day: group for day, group in quote_frame.groupby("session_date", sort=False)}
    prediction_rows = predictions.copy()
    prediction_rows["date"] = pd.to_datetime(prediction_rows["date"], errors="raise").dt.date
    prediction_rows = prediction_rows.sort_values("date").drop_duplicates("date", keep="last")
    sessions = sorted(prediction_rows["date"].unique())
    session_index = {session: i for i, session in enumerate(sessions)}
    trades = []
    next_available_index = 0

    for signal in prediction_rows.itertuples(index=False):
        signal_day = signal.date
        if signal_day not in session_index:
            continue
        signal_index = session_index[signal_day]
        entry_index = signal_index + 1
        exit_index = signal_index + hold_days + (1 if quote_mode == "eod" else 0)
        if entry_index < next_available_index or exit_index >= len(sessions):
            continue
        forecast = float(signal.ridge_prediction)
        if not np.isfinite(forecast) or forecast <= 0:
            continue
        entry_day = sessions[entry_index]
        exit_day = sessions[exit_index]
        entry_window = (16 * 60, 16 * 60) if quote_mode == "eod" else (9 * 60 + 30, 9 * 60 + 45)
        exit_window = (16 * 60, 16 * 60) if quote_mode == "eod" else (15 * 60 + 45, 16 * 60)
        candidates = _snapshot(quote_by_day.get(entry_day, quote_frame.iloc[0:0]), entry_day, *entry_window)
        if candidates.empty:
            continue
        dte = (pd.to_datetime(candidates["expiration"]) - pd.Timestamp(entry_day)).dt.days
        candidates = candidates.loc[
            (candidates["underlying"] == "SPY")
            & (candidates["contract_multiplier"] == 100)
            & (candidates["deliverable_shares"] == 100)
            & (candidates["deliverable_cash"] == 0)
            & (candidates["expiration"] > exit_day)
            & (dte >= min_dte)
            & (dte <= max_dte)
            & (candidates["bid"] > 0)
            & (candidates["ask"] >= candidates["bid"])
            & (candidates["ask_size"] >= 1)
            & (candidates["implied_volatility"] > 0)
        ]
        if candidates.empty:
            continue

        spot = float(candidates["underlying_price"].median())
        mid = (candidates["bid"] + candidates["ask"]) / 2
        spread = (candidates["ask"] - candidates["bid"]) / mid.where(mid > 0)
        liquid = candidates.loc[(mid > 0) & (spread <= max_relative_spread)]
        calls = liquid.loc[liquid["right"] == "C"].set_index(["expiration", "strike"])
        puts = liquid.loc[liquid["right"] == "P"].set_index(["expiration", "strike"])
        paired = calls.add_suffix("_call").join(puts.add_suffix("_put"), how="inner").reset_index()
        if paired.empty:
            continue
        paired["expiry_distance"] = (
            (pd.to_datetime(paired["expiration"]) - pd.Timestamp(entry_day)).dt.days - target_dte
        ).abs()
        paired["atm_distance"] = (paired["strike"] - spot).abs() / spot
        selected = paired.sort_values(["expiry_distance", "atm_distance"]).iloc[0]
        call = selected.filter(regex="_call$")
        put = selected.filter(regex="_put$")
        call.index = call.index.str.removesuffix("_call")
        put.index = put.index.str.removesuffix("_put")
        call["expiration"] = selected["expiration"]
        put["expiration"] = selected["expiration"]
        call["strike"] = float(selected["strike"])
        put["strike"] = float(selected["strike"])
        implied_vol = float((call["implied_volatility"] + put["implied_volatility"]) / 2)
        if forecast <= implied_vol * (1 + edge_threshold):
            continue

        exit_group = _snapshot(quote_by_day.get(exit_day, quote_frame.iloc[0:0]), exit_day, *exit_window)
        if exit_group.empty:
            continue
        call_exit = exit_group.loc[exit_group["contract_symbol"] == call["contract_symbol"]]
        put_exit = exit_group.loc[exit_group["contract_symbol"] == put["contract_symbol"]]
        if call_exit.empty or put_exit.empty:
            continue
        call_exit, put_exit = call_exit.iloc[-1], put_exit.iloc[-1]
        if call_exit["expiration"] <= exit_day:
            continue
        if call_exit["bid_size"] < 1 or put_exit["bid_size"] < 1:
            continue

        entry_debit = float(call["ask"] + put["ask"]) * 100
        exit_credit = float(call_exit["bid"] + put_exit["bid"]) * 100
        fees = fee_per_contract * 4  # two legs, entry and exit
        pnl = exit_credit - entry_debit - fees
        trades.append({
            "entry_date": str(entry_day),
            "exit_date": str(exit_day),
            "expiration": str(call["expiration"]),
            "call_symbol": call["contract_symbol"],
            "put_symbol": put["contract_symbol"],
            "spot": spot,
            "strike": float(call["strike"]),
            "forecast_rv": forecast,
            "paired_implied_vol": implied_vol,
            "entry_debit": entry_debit,
            "exit_credit": exit_credit,
            "fees": fees,
            "net_pnl": pnl,
        })
        next_available_index = exit_index + 1

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        summary = {"trade_count": 0, "message": "No trades passed the signal and quote-quality filters."}
    else:
        pnl = trade_frame["net_pnl"].to_numpy(dtype=float)
        positive = pnl[pnl > 0].sum()
        negative = -pnl[pnl < 0].sum()
        summary = {
            "trade_count": int(len(pnl)),
            "net_pnl": float(pnl.sum()),
            "average_pnl": float(pnl.mean()),
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": float(positive / negative) if negative else None,
            "total_entry_debit": float(trade_frame["entry_debit"].sum()),
            "total_fees": float(trade_frame["fees"].sum()),
            "assumptions": f"Long straddles; {quote_mode} snapshot; entry at asks; exit at bids; 100 multiplier; one open trade at a time.",
        }
    return summary, trade_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--quotes", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/straddle"))
    parser.add_argument("--edge-threshold", type=float, default=0.10)
    parser.add_argument("--fee-per-contract", type=float, default=0.65)
    parser.add_argument("--quote-mode", choices=("intraday", "eod"), default="intraday")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    quotes = pd.read_parquet(args.quotes) if args.quotes.suffix.lower() == ".parquet" else pd.read_csv(args.quotes)
    summary, trades = backtest_long_atm_straddle(
        predictions,
        quotes,
        edge_threshold=args.edge_threshold,
        fee_per_contract=args.fee_per_contract,
        quote_mode=args.quote_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trades.to_csv(args.output_dir / "trades.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
