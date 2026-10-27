"""Exploratory first-day return study for Hong Kong IPOs.

This is an IPO outcome study, not an expected subscription P&L model: it does
not model allotment probability, allocated lots, application financing, or
subscription fees. IPO events are kept intact and all forecasts are
chronological.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_first_day_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"listing_event_id", "stock_code", "listing_date", "date", "listed_day",
                "ipo_price", "open", "high", "low", "close", "ohlc_consistent"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Event panel missing columns: {', '.join(sorted(missing))}")
    frame = panel.copy()
    frame["listing_date"] = pd.to_datetime(frame.listing_date, errors="raise")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    first = frame.loc[frame.listed_day.eq(1)].copy()
    first["ohlc_consistent"] = first.ohlc_consistent.astype("boolean")
    first = first.loc[first.ohlc_consistent.eq(True)].copy()
    first["first_day_gross_return"] = first.close / first.ipo_price - 1.0
    first["first_day_open_return"] = first.open / first.ipo_price - 1.0
    first["first_day_intraday_return"] = first.close / first.open - 1.0
    first["listing_year"] = first.listing_date.dt.year
    # Preserve available pre-listing reference fields for later feature work.
    keep = ["listing_event_id", "stock_code", "company_name", "listing_date", "listing_year",
            "prospectus_date", "sponsors", "ipo_price", "funds_raised_hkd", "lot_size",
            "open", "high", "low", "close", "volume", "turnover", "source",
            "first_day_open_return", "first_day_intraday_return", "first_day_gross_return"]
    return first[[col for col in keep if col in first]].sort_values("listing_date").reset_index(drop=True)


def _year_stats(group: pd.DataFrame) -> dict:
    values = group.first_day_gross_return
    return {
        "ipos": int(len(group)),
        "positive_share": float(values.gt(0).mean()),
        "mean_return": float(values.mean()),
        "median_return": float(values.median()),
        "p05_return": float(values.quantile(.05)),
        "p95_return": float(values.quantile(.95)),
        "worst_return": float(values.min()),
    }


def chronological_benchmarks(outcomes: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare a zero forecast and an expanding historical-mean forecast.

    The expanding mean uses only earlier listing dates. Predictions are
    descriptive benchmarks, not trade recommendations or allocation-adjusted
    subscription returns.
    """
    rows = outcomes.sort_values(["listing_date", "listing_event_id"]).copy()
    rows["expanding_mean_prediction"] = rows.first_day_gross_return.shift(1).expanding(min_periods=20).mean()
    rows["zero_prediction"] = 0.0
    rows["prediction_split"] = np.select(
        [rows.listing_year.le(2023), rows.listing_year.eq(2024), rows.listing_year.eq(2025)],
        ["train_history", "validation", "test"], default="latest_partial")
    scored = rows.loc[rows.expanding_mean_prediction.notna()].copy()
    by_year = {str(int(year)): _year_stats(group)
               for year, group in rows.groupby("listing_year", sort=True)}
    by_split = {}
    for split, group in scored.groupby("prediction_split", sort=True):
        actual = group.first_day_gross_return.to_numpy()
        mean_pred = group.expanding_mean_prediction.to_numpy()
        by_split[split] = {
            "ipos": int(len(group)),
            "actual_mean_return": float(actual.mean()),
            "actual_median_return": float(np.median(actual)),
            "positive_share": float(np.mean(actual > 0)),
            "zero_forecast_mae": float(np.mean(np.abs(actual))),
            "expanding_mean_forecast_mae": float(np.mean(np.abs(actual - mean_pred))),
            "zero_forecast_directional_accuracy": float(np.mean((actual > 0) == (0 > 0))),
            "expanding_mean_directional_accuracy": float(np.mean((mean_pred > 0) == (actual > 0))),
        }
    summary = {
        "target": "first_listing_day_close_vs_final_ipo_price_gross_return",
        "events": int(len(rows)),
        "event_date_range": [rows.listing_date.min().date().isoformat(), rows.listing_date.max().date().isoformat()],
        "yearly_outcomes": by_year,
        "chronological_benchmarks": by_split,
        "split_note": "Listing years 2021-2023 show prior history; 2024 validation; 2025 test; 2026 is an incomplete latest sample.",
        "limitations": [
            "This target is first-day price return, not expected IPO subscription P&L.",
            "No allotment probability or allocated share count is modeled.",
            "No pre-subscription feature set with verified as-of timestamps is present in the current panel.",
            "Market prices are Yahoo/yfinance exploratory data and are not exchange-verified.",
            "The expanding-mean forecast is a weak benchmark, not a machine-learning result.",
        ],
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, parse_dates=["listing_date", "date"])
    outcomes = build_first_day_outcomes(panel)
    summary, forecasts = chronological_benchmarks(outcomes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(args.output_dir / "first_day_outcomes.csv", index=False)
    forecasts.to_csv(args.output_dir / "chronological_benchmarks.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
