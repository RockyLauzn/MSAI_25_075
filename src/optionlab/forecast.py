"""Evaluate a simple volatility model using a purged chronological split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from .dataset import build_features


def _feature_names(dataset: pd.DataFrame) -> list[str]:
    base = ["rv_5", "rv_20", "rv_60", "rv_120", "mom_5", "mom_20", "mom_60"]
    return [*base, *( ["atm_iv_30"] if "atm_iv_30" in dataset.columns else []),
            *sorted(c for c in dataset.columns if c.startswith("oi_"))]


def chronological_split(n_rows: int, horizon: int, target_offset: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation/test indices and purge labels crossing boundaries."""
    if horizon < 1 or target_offset < 1:
        raise ValueError("horizon and target_offset must be at least 1")
    if n_rows < 30:
        raise ValueError("Need at least 30 labeled observations")
    first = int(n_rows * 0.70)
    second = int(n_rows * 0.85)
    purge = horizon + target_offset - 1
    train = np.arange(0, first - purge)
    valid = np.arange(first, second - purge)
    test = np.arange(second, n_rows)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(
            "Not enough rows for these splits after purging the forward target window; "
            "use a longer history or shorter horizon"
        )
    return train, valid, test


def _fit_ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float
) -> np.ndarray:
    """Fit standardized ridge regression in log-volatility space."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_eval = (x_eval - mean) / scale
    # Add an unpenalized intercept, then solve the regularized normal equation.
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_eval = np.column_stack([np.ones(len(x_eval)), x_eval])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return np.exp(x_eval @ beta)


def _fit_ridge_linear(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float
) -> np.ndarray:
    """Fit standardized ridge regression to an untransformed target."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_eval = (x_eval - mean) / scale
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_eval = np.column_stack([np.ones(len(x_eval)), x_eval])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return x_eval @ beta


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    actual_variance = np.maximum(actual**2, 1e-12)
    forecast_variance = np.maximum(predicted**2, 1e-12)
    qlike = actual_variance / forecast_variance - np.log(actual_variance / forecast_variance) - 1.0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "qlike_variance": float(np.mean(qlike)),
    }


def evaluate(
    dataset: pd.DataFrame,
    horizon: int = 20,
    latest_features: pd.DataFrame | None = None,
    target_offset: int = 1,
) -> tuple[dict, pd.DataFrame]:
    target = f"target_rv_{horizon}"
    feature_names = _feature_names(dataset)
    missing = set(["date", target, *feature_names]).difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    frame = dataset.dropna(subset=[target, *feature_names]).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    train_idx, valid_idx, test_idx = chronological_split(len(frame), horizon, target_offset)
    x = frame[feature_names].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Features and target must be finite; target volatility must be positive")

    # Tune the ridge penalty on validation only, then refit on train + validation.
    candidates = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
    # Keep implementation transparent: the base solver accepts an explicit alpha.
    best_alpha, best_error = None, float("inf")
    for alpha in candidates:
        prediction = _fit_ridge_predict(x[train_idx], np.log(y[train_idx]), x[valid_idx], alpha)
        error = _metrics(y[valid_idx], prediction)["mae"]
        if error < best_error:
            best_alpha, best_error = alpha, error

    fit_idx = np.concatenate([train_idx, valid_idx])
    model_prediction = _fit_ridge_predict(x[fit_idx], np.log(y[fit_idx]), x[test_idx], best_alpha)
    baseline_prediction = x[test_idx, feature_names.index("rv_20")]
    actual = y[test_idx]
    nonoverlap_positions = np.arange(0, len(test_idx), horizon)
    nonoverlap_actual = actual[nonoverlap_positions]
    nonoverlap_model = model_prediction[nonoverlap_positions]
    nonoverlap_baseline = baseline_prediction[nonoverlap_positions]
    scores = {
        "target": target,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(valid_idx)),
        "test_rows": int(len(test_idx)),
        "test_start": str(frame.loc[test_idx[0], "date"]),
        "test_end": str(frame.loc[test_idx[-1], "date"]),
        "selected_ridge_alpha": float(best_alpha),
        "model": _metrics(actual, model_prediction),
        "baseline_rv_20": _metrics(actual, baseline_prediction),
        "model_oos_r2_vs_rv_20": float(
            1.0
            - np.sum((actual - model_prediction) ** 2)
            / max(np.sum((actual - baseline_prediction) ** 2), 1e-12)
        ),
        "test_nonoverlap_every_horizon": {
            "sample_count": int(len(nonoverlap_positions)),
            "model": _metrics(nonoverlap_actual, nonoverlap_model),
            "baseline_rv_20": _metrics(nonoverlap_actual, nonoverlap_baseline),
            "model_oos_r2_vs_rv_20": float(
                1.0
                - np.sum((nonoverlap_actual - nonoverlap_model) ** 2)
                / max(np.sum((nonoverlap_actual - nonoverlap_baseline) ** 2), 1e-12)
            ),
        },
    }
    if "atm_iv_30" in feature_names:
        iv_baseline = x[test_idx, feature_names.index("atm_iv_30")]
        scores["baseline_atm_iv_30"] = _metrics(actual, iv_baseline)
        scores["model_oos_r2_vs_atm_iv_30"] = float(
            1.0 - np.sum((actual - model_prediction) ** 2)
            / max(np.sum((actual - iv_baseline) ** 2), 1e-12)
        )
    if latest_features is not None:
        latest = latest_features.dropna(subset=feature_names).copy()
        latest["date"] = pd.to_datetime(latest["date"], errors="raise")
        latest = latest.sort_values("date").drop_duplicates("date", keep="last")
        if latest.empty:
            raise ValueError("No complete latest feature row was provided")
        if "atm_iv_30" in feature_names and "atm_iv_30" not in latest.columns:
            raise ValueError("Latest forecast with option features also requires a same-date atm_iv_30 value")
        refit_prediction = _fit_ridge_predict(
            x, np.log(y), latest.loc[:, feature_names].tail(1).to_numpy(dtype=float), best_alpha
        )[0]
        scores["latest_forecast"] = {
            "as_of": str(latest.iloc[-1]["date"]),
            "trained_through_labeled_date": str(frame.iloc[-1]["date"]),
            "predicted_annualized_rv": float(refit_prediction),
        }
    predictions = pd.DataFrame({
        "date": frame.loc[test_idx, "date"].to_numpy(),
        "actual_target_rv": actual,
        "ridge_prediction": model_prediction,
        "rv_20_baseline": baseline_prediction,
    })
    return scores, predictions


def walk_forward_predictions(dataset: pd.DataFrame, horizon: int = 20, target_offset: int = 2,
                             start_year: int = 2015) -> tuple[dict, pd.DataFrame]:
    """Predict each calendar year using only labels fully observable before that year."""
    target = f"target_rv_{horizon}"
    feature_names = _feature_names(dataset)
    frame = dataset.dropna(subset=[target, *feature_names]).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    x_all = frame[feature_names].to_numpy(dtype=float)
    y_all = frame[target].to_numpy(dtype=float)
    alphas = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
    predictions = []
    fold_summary = []
    purge = horizon + target_offset - 1
    for year in sorted(frame.loc[frame.date.dt.year >= start_year, "date"].dt.year.unique()):
        test_idx = np.flatnonzero(frame.date.dt.year.to_numpy() == year)
        first_idx = int(test_idx[0])
        history_end = max(0, first_idx - purge)
        history_idx = np.arange(history_end)
        if len(history_idx) < 250 or len(test_idx) == 0:
            continue
        validation_start = int(len(history_idx) * 0.80)
        train_rel = np.arange(0, validation_start - purge)
        valid_rel = np.arange(validation_start, len(history_idx))
        if len(train_rel) < 30 or len(valid_rel) < 20:
            continue
        best_alpha, best_error = None, float("inf")
        for alpha in alphas:
            p = _fit_ridge_predict(x_all[history_idx[train_rel]], np.log(y_all[history_idx[train_rel]]),
                                   x_all[history_idx[valid_rel]], alpha)
            error = _metrics(y_all[history_idx[valid_rel]], p)["mae"]
            if error < best_error:
                best_alpha, best_error = alpha, error
        fit_rel = np.arange(len(history_idx))
        forecast = _fit_ridge_predict(x_all[history_idx[fit_rel]], np.log(y_all[history_idx[fit_rel]]),
                                      x_all[test_idx], best_alpha)
        fold = pd.DataFrame({
            "date": frame.loc[test_idx, "date"].to_numpy(),
            "actual_target_rv": y_all[test_idx], "ridge_prediction": forecast,
            "rv_20_baseline": x_all[test_idx, feature_names.index("rv_20")],
        })
        if "atm_iv_30" in feature_names:
            fold["iv_30_baseline"] = x_all[test_idx, feature_names.index("atm_iv_30")]
        predictions.append(fold)
        fold_summary.append({"year": int(year), "train_through": str(frame.loc[history_idx[fit_rel[-1]], "date"].date()),
                             "selected_ridge_alpha": float(best_alpha), "prediction_rows": int(len(fold))})
    if not predictions:
        raise ValueError("No walk-forward folds had sufficient prior history")
    out = pd.concat(predictions, ignore_index=True)
    actual = out.actual_target_rv.to_numpy()
    model = out.ridge_prediction.to_numpy()
    baseline = out.rv_20_baseline.to_numpy()
    yearly_metrics = {}
    for year, group in out.groupby(out.date.dt.year):
        y = group.actual_target_rv.to_numpy()
        m = group.ridge_prediction.to_numpy()
        b = group.rv_20_baseline.to_numpy()
        yearly = {
            "prediction_rows": int(len(group)), "model": _metrics(y, m),
            "baseline_rv_20": _metrics(y, b),
            "model_oos_r2_vs_rv_20": float(1 - np.sum((y-m)**2) / max(np.sum((y-b)**2), 1e-12)),
        }
        if "iv_30_baseline" in group:
            iv = group.iv_30_baseline.to_numpy()
            yearly["baseline_atm_iv_30"] = _metrics(y, iv)
            yearly["model_oos_r2_vs_atm_iv_30"] = float(
                1 - np.sum((y-m)**2) / max(np.sum((y-iv)**2), 1e-12)
            )
        yearly_metrics[str(int(year))] = yearly
    scores = {"target": target, "target_offset": target_offset, "start_year": start_year,
              "end_date": str(out.date.max()), "prediction_rows": int(len(out)), "folds": fold_summary,
              "model": _metrics(actual, model), "baseline_rv_20": _metrics(actual, baseline),
              "metrics_by_year": yearly_metrics,
              "model_oos_r2_vs_rv_20": float(1 - np.sum((actual-model)**2) /
                                             max(np.sum((actual-baseline)**2), 1e-12))}
    if "iv_30_baseline" in out:
        iv = out.iv_30_baseline.to_numpy()
        scores["baseline_atm_iv_30"] = _metrics(actual, iv)
        scores["model_oos_r2_vs_atm_iv_30"] = float(
            1 - np.sum((actual-model)**2) / max(np.sum((actual-iv)**2), 1e-12)
        )
    return scores, out


def walk_forward_variance_premium(dataset: pd.DataFrame, horizon: int = 20, target_offset: int = 2,
                                  start_year: int = 2015) -> tuple[dict, pd.DataFrame]:
    """Predict future realized variance minus signal-date 30-DTE ATM implied variance."""
    target = f"target_vrp_{horizon}"
    rv_target = f"target_rv_{horizon}"
    if "atm_iv_30" not in dataset or target not in dataset:
        raise ValueError("Variance-premium mode requires a dataset with atm_iv_30 option features")
    feature_names = _feature_names(dataset)
    frame = dataset.dropna(subset=[target, rv_target, *feature_names]).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    x_all = frame[feature_names].to_numpy(dtype=float)
    y_all = frame[target].to_numpy(dtype=float)
    iv_all = frame.atm_iv_30.to_numpy(dtype=float)
    rv_all = frame[rv_target].to_numpy(dtype=float)
    alphas = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
    purge = horizon + target_offset - 1
    predictions, folds = [], []
    for year in sorted(frame.loc[frame.date.dt.year >= start_year, "date"].dt.year.unique()):
        test_idx = np.flatnonzero(frame.date.dt.year.to_numpy() == year)
        history_idx = np.arange(max(0, int(test_idx[0]) - purge))
        if len(history_idx) < 250 or len(test_idx) == 0:
            continue
        validation_start = int(len(history_idx) * 0.80)
        train_rel = np.arange(0, validation_start - purge)
        valid_rel = np.arange(validation_start, len(history_idx))
        if len(train_rel) < 30 or len(valid_rel) < 20:
            continue
        best_alpha, best_error = None, float("inf")
        for alpha in alphas:
            p = _fit_ridge_linear(x_all[history_idx[train_rel]], y_all[history_idx[train_rel]],
                                  x_all[history_idx[valid_rel]], alpha)
            error = float(np.mean(np.abs(y_all[history_idx[valid_rel]] - p)))
            if error < best_error:
                best_alpha, best_error = alpha, error
        premium_prediction = _fit_ridge_linear(x_all[history_idx], y_all[history_idx],
                                               x_all[test_idx], best_alpha)
        rv_prediction = np.sqrt(np.maximum(iv_all[test_idx] ** 2 + premium_prediction, 1e-10))
        predictions.append(pd.DataFrame({
            "date": frame.loc[test_idx, "date"].to_numpy(),
            "actual_target_rv": rv_all[test_idx],
            "actual_vrp": y_all[test_idx],
            "predicted_vrp": premium_prediction,
            "ridge_prediction": rv_prediction,
            "rv_20_baseline": x_all[test_idx, feature_names.index("rv_20")],
            "iv_30_baseline": iv_all[test_idx],
        }))
        folds.append({"year": int(year), "train_through": str(frame.loc[history_idx[-1], "date"].date()),
                      "selected_ridge_alpha": float(best_alpha), "prediction_rows": int(len(test_idx))})
    if not predictions:
        raise ValueError("No walk-forward folds had sufficient prior history")
    out = pd.concat(predictions, ignore_index=True)
    actual_vrp = out.actual_vrp.to_numpy()
    predicted_vrp = out.predicted_vrp.to_numpy()
    zero = np.zeros_like(actual_vrp)
    actual_rv = out.actual_target_rv.to_numpy()
    rv_pred = out.ridge_prediction.to_numpy()
    iv = out.iv_30_baseline.to_numpy()
    score = {
        "target": target, "target_offset": target_offset, "start_year": start_year,
        "end_date": str(out.date.max()), "prediction_rows": int(len(out)), "folds": folds,
        "variance_premium_mae": float(np.mean(np.abs(predicted_vrp - actual_vrp))),
        "variance_premium_rmse": float(np.sqrt(np.mean((predicted_vrp - actual_vrp) ** 2))),
        "variance_premium_oos_r2_vs_zero": float(
            1 - np.sum((actual_vrp - predicted_vrp) ** 2) /
            max(np.sum((actual_vrp - zero) ** 2), 1e-12)
        ),
        "implied_vol_forecast_of_rv": _metrics(actual_rv, iv),
        "vrp_model_rv_forecast": _metrics(actual_rv, rv_pred),
        "rv_20_baseline": _metrics(actual_rv, out.rv_20_baseline.to_numpy()),
    }
    return score, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Labeled dataset CSV")
    parser.add_argument("--prices", type=Path, help="Raw price CSV for a current unlabeled forecast")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--target-offset", type=int, default=1)
    parser.add_argument("--walk-forward-start-year", type=int)
    parser.add_argument("--predict-variance-premium", action="store_true",
                        help="Predict future realized variance minus signal-date ATM implied variance")
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    latest_features = None
    if args.prices:
        latest_features = build_features(pd.read_csv(args.prices))
    if args.walk_forward_start_year is not None:
        if args.predict_variance_premium:
            scores, predictions = walk_forward_variance_premium(
                data, args.horizon, args.target_offset, args.walk_forward_start_year
            )
        else:
            scores, predictions = walk_forward_predictions(
                data, args.horizon, args.target_offset, args.walk_forward_start_year
            )
    else:
        scores, predictions = evaluate(data, args.horizon, latest_features, args.target_offset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "volatility_metrics.json").write_text(
        json.dumps(scores, indent=2), encoding="utf-8"
    )
    predictions.to_csv(args.output_dir / "volatility_predictions.csv", index=False)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
