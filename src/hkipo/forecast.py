"""IPO-level chronological supervised baseline for next-session intraday return."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                       x_eval: np.ndarray, alpha: float) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    train = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
    evaluate = np.column_stack([np.ones(len(x_eval)), (x_eval - mean) / scale])
    penalty = np.eye(train.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(train.T @ train + penalty, train.T @ y_train)
    return evaluate @ coef


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "directional_accuracy": (None if np.all(prediction == 0) else
                                 float(np.mean(np.sign(prediction) == np.sign(actual)))),
        "r2_vs_zero_return": float(1 - np.sum(error**2) / max(np.sum(actual**2), 1e-12)),
    }


def walk_forward_ipos(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    required = {"listing_event_id", "listing_date", "date", "listed_day", "open", "close",
                "volume", "premium_to_ipo", "return_1d", "return_5d", "rv_5d",
                "ohlc_consistent"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Event panel missing columns: {', '.join(sorted(missing))}")
    frame = panel.copy()
    frame["listing_date"] = pd.to_datetime(frame.listing_date, errors="raise")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    frame = frame.sort_values(["listing_event_id", "listed_day"]).reset_index(drop=True)
    grouped = frame.groupby("listing_event_id", sort=False)
    next_open = grouped.open.shift(-1)
    next_close = grouped.close.shift(-1)
    next_quality = grouped.ohlc_consistent.shift(-1).eq(True)
    frame["target_next_session_intraday_return"] = next_close / next_open - 1
    prior_close = grouped.close.shift(1)
    frame["overnight_gap"] = frame.open / prior_close - 1
    frame["intraday_return"] = frame.close / frame.open - 1
    frame["log_volume"] = np.log1p(frame.volume)
    features = ["listed_day", "premium_to_ipo", "return_1d", "return_5d", "rv_5d",
                "overnight_gap", "intraday_return", "log_volume"]
    valid = (frame.ohlc_consistent.astype(bool) & next_quality
             & frame.target_next_session_intraday_return.notna())
    frame = frame.loc[valid].copy()
    frame["listing_year"] = frame.listing_date.dt.year
    train = frame.listing_year.le(2023)
    valid_year = frame.listing_year.eq(2024)
    test = frame.listing_year.eq(2025)
    if train.sum() < 100 or valid_year.sum() < 50 or test.sum() < 50:
        raise ValueError("Insufficient IPO-level chronological samples for 2021-23/2024/2025 split")
    x = frame[features].replace([np.inf, -np.inf], np.nan)
    complete = x.notna().all(axis=1)
    frame = frame.loc[complete].copy()
    x = x.loc[complete].to_numpy(dtype=float)
    y = frame.target_next_session_intraday_return.to_numpy(dtype=float)
    test_year = int(frame.listing_year.max())
    validation_year = test_year - 1
    train_end_year = validation_year - 1
    train = frame.listing_year.le(train_end_year).to_numpy()
    valid_year = frame.listing_year.eq(validation_year).to_numpy()
    test = frame.listing_year.eq(test_year).to_numpy()
    alphas = (.01, .1, 1.0, 10.0, 100.0)
    alpha_scores = {}
    for alpha in alphas:
        pred = _ridge_fit_predict(x[train], y[train], x[valid_year], alpha)
        alpha_scores[str(alpha)] = _metrics(y[valid_year], pred)["mae"]
    chosen = min(alphas, key=lambda a: alpha_scores[str(a)])
    fit = train | valid_year
    prediction = _ridge_fit_predict(x[fit], y[fit], x[test], chosen)
    out = frame.loc[test, ["listing_event_id", "stock_code", "listing_date", "date",
                           "listed_day", "target_next_session_intraday_return"]].copy()
    out["ridge_prediction"] = prediction
    out["zero_return_prediction"] = 0.0
    score = {
        "target": "next_session_open_to_close_return",
        "feature_columns": features,
        "train_listing_years": f"2021-{train_end_year}",
        "validation_listing_year": validation_year,
        "test_listing_year": test_year,
        "train_rows": int(train.sum()), "validation_rows": int(valid_year.sum()),
        "test_rows": int(test.sum()), "test_ipo_count": int(frame.loc[test, "listing_event_id"].nunique()),
        "selected_ridge_alpha": float(chosen), "validation_mae_by_alpha": alpha_scores,
        "ridge": _metrics(y[test], prediction),
        "zero_return_baseline": _metrics(y[test], np.zeros(test.sum())),
    }
    return score, out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, parse_dates=["listing_date", "date"])
    metrics, predictions = walk_forward_ipos(panel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
