"""Build a leakage-aware realized-volatility forecasting dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LOOKBACKS = (5, 20, 60, 120)
MOMENTUM_WINDOWS = (5, 20, 60)
DEFAULT_HORIZON = 20


def _prepare_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = {"date", "adj_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    frame = prices.loc[:, ["date", "adj_close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="raise")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame["adj_close"].isna().any() or (frame["adj_close"] <= 0).any():
        raise ValueError("adj_close must contain only finite positive prices")
    if not np.isfinite(frame["adj_close"].to_numpy()).all():
        raise ValueError("adj_close must contain only finite positive prices")
    returns = np.log(frame["adj_close"]).diff()
    return frame, returns


def _feature_frame(frame: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    for window in LOOKBACKS:
        features[f"rv_{window}"] = (
            returns.pow(2).rolling(window, min_periods=window).mean().pow(0.5)
            * np.sqrt(TRADING_DAYS)
        )
    for window in MOMENTUM_WINDOWS:
        features[f"mom_{window}"] = np.log(frame["adj_close"] / frame["adj_close"].shift(window))
    return features


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return trailing features for every eligible date, including latest date."""
    frame, returns = _prepare_prices(prices)
    features = _feature_frame(frame, returns)
    result = features.assign(date=frame["date"]).dropna().reset_index(drop=True)
    return result.loc[:, ["date", *features.columns]]


def build_dataset(prices: pd.DataFrame, horizon: int = DEFAULT_HORIZON, target_offset: int = 1,
                  extra_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create trailing features and a forward annualized realized-vol target.

    Input must have ``date`` and ``adj_close`` columns. At date t, features use
    returns through t; the target uses returns from t+target_offset through the next `horizon` returns.
    """
    if horizon < 1 or target_offset < 1:
        raise ValueError("horizon and target_offset must be at least 1")
    frame, returns = _prepare_prices(prices)
    features = _feature_frame(frame, returns)
    if extra_features is not None:
        if "date" not in extra_features.columns:
            raise ValueError("Extra features require a date column")
        extra_names = [name for name in extra_features.columns if name != "date"]
        if not extra_names:
            raise ValueError("At least one extra feature column is required")
        if set(extra_names).intersection(features.columns):
            raise ValueError("Extra feature names must not overwrite price features")
        extra = extra_features.loc[:, ["date", *extra_names]].copy()
        extra["date"] = pd.to_datetime(extra["date"], errors="raise")
        for name in extra_names:
            extra[name] = pd.to_numeric(extra[name], errors="raise")
        if extra["date"].duplicated().any():
            raise ValueError("Extra features must contain at most one row per date")
        for name in extra_names:
            features[name] = frame["date"].map(extra.set_index("date")[name])

    # Reverse the return series so each row sees only the next `horizon` returns.
    future_squared = returns.shift(-target_offset).iloc[::-1]
    target = (
        future_squared.pow(2)
        .rolling(horizon, min_periods=horizon)
        .mean()
        .pow(0.5)
        .mul(np.sqrt(TRADING_DAYS))
        .iloc[::-1]
    )

    target_name = f"target_rv_{horizon}"
    result = features.assign(**{target_name: target}, date=frame["date"])
    if "atm_iv_30" in features:
        result[f"target_vrp_{horizon}"] = result[target_name] ** 2 - result["atm_iv_30"] ** 2
    result = result.dropna().reset_index(drop=True)
    result[target_name] = result[target_name].astype(float)
    columns = ["date", *features.columns, target_name]
    if f"target_vrp_{horizon}" in result:
        columns.append(f"target_vrp_{horizon}")
    return result.loc[:, columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV with date,adj_close")
    parser.add_argument("--output", required=True, type=Path, help="Output labeled dataset CSV")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="Forward realized-volatility horizon in trading days")
    parser.add_argument("--target-offset", type=int, default=1,
                        help="Skip this many returns from the signal date (default: 1)")
    parser.add_argument("--option-features", type=Path,
                        help="CSV with one daily atm_iv_30 value per date")
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    option_features = pd.read_csv(args.option_features, parse_dates=["date"]) if args.option_features else None
    dataset = build_dataset(source, args.horizon, args.target_offset, option_features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset):,} rows to {args.output}")


if __name__ == "__main__":
    main()
