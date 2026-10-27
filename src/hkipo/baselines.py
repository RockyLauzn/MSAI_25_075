"""Simple event-study baselines for HK IPO first-month returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _statutory_fee_rate(date: pd.Series) -> pd.Series:
    """Per-side proportional costs; stamp duty and exchange fees vary by date."""
    date = pd.to_datetime(date)
    stamp = pd.Series(np.select(
        [date < pd.Timestamp("2021-08-01"), date < pd.Timestamp("2023-11-17")],
        [.001, .0013], default=.001), index=date.index)
    afrc = pd.Series(np.where(date < pd.Timestamp("2022-01-01"), 0.0, .0000015), index=date.index)
    exchange = pd.Series(np.where(date < pd.Timestamp("2023-01-01"), .00005, .0000565), index=date.index)
    return stamp + .000027 + afrc + exchange


def event_hold_baselines(panel: pd.DataFrame, holding_sessions: tuple[int, ...] = (1, 5, 10, 20, 30),
                         brokerage_per_side: float = .0025) -> tuple[pd.DataFrame, dict]:
    required = {"listing_event_id", "stock_code", "date", "listed_day", "open", "close", "ohlc_consistent"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Event panel missing columns: {', '.join(sorted(missing))}")
    if brokerage_per_side < 0:
        raise ValueError("Brokerage rate cannot be negative")
    rows = []
    for event, group in panel.groupby("listing_event_id", sort=False):
        group = group.sort_values("listed_day")
        entry = group.loc[group.listed_day.eq(1)]
        if entry.empty or not bool(entry.iloc[0].ohlc_consistent):
            continue
        entry = entry.iloc[0]
        for horizon in holding_sessions:
            exit_rows = group.loc[group.listed_day.eq(horizon)]
            if exit_rows.empty or not bool(exit_rows.iloc[0].ohlc_consistent):
                continue
            exit_row = exit_rows.iloc[0]
            gross = float(exit_row.close / entry.open - 1.0)
            buy_rate = float(_statutory_fee_rate(pd.Series([entry.date])).iloc[0]) + brokerage_per_side
            sell_rate = float(_statutory_fee_rate(pd.Series([exit_row.date])).iloc[0]) + brokerage_per_side
            net = gross - buy_rate - sell_rate * (1.0 + gross)
            rows.append({
                "listing_event_id": event, "stock_code": entry.stock_code,
                "listing_date": entry.date, "holding_sessions": horizon,
                "entry_date": entry.date, "entry_open": float(entry.open),
                "exit_date": exit_row.date, "exit_close": float(exit_row.close),
                "gross_return": gross, "estimated_net_return": net,
                "entry_fee_rate": buy_rate, "exit_fee_rate": sell_rate,
            })
    trades = pd.DataFrame(rows)
    summary = {"brokerage_per_side_assumption": brokerage_per_side,
               "return_basis": "fractional/notional event study; no lot-size or stamp-duty rounding",
               "by_holding_sessions": {}}
    if trades.empty:
        return trades, summary
    for horizon, group in trades.groupby("holding_sessions"):
        net, gross = group.estimated_net_return, group.gross_return
        summary["by_holding_sessions"][str(int(horizon))] = {
            "events": int(len(group)),
            "gross_mean": float(gross.mean()), "gross_median": float(gross.median()),
            "estimated_net_mean": float(net.mean()), "estimated_net_median": float(net.median()),
            "estimated_net_win_rate": float(net.gt(0).mean()),
            "estimated_net_p05": float(net.quantile(.05)),
            "estimated_net_worst": float(net.min()),
        }
    return trades, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--brokerage-per-side", type=float, default=.0025,
                        help="Broker commission assumption as a fraction per side (default: 0.25%%)")
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, parse_dates=["date"])
    trades, summary = event_hold_baselines(panel, brokerage_per_side=args.brokerage_per_side)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.output_dir / "hold_baseline_trades.csv", index=False)
    (args.output_dir / "hold_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
