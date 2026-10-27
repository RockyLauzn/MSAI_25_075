"""Paired event comparison of DRL training windows on identical test folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["horizon", "test_year", "event_id", "policy", "fee_bps"]


def load_event_average(path: Path, label: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {*KEYS, "seed", "final_return", "max_drawdown", "turnover"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    # Average stochastic training seeds within each held-out IPO before
    # resampling; buy-and-hold has one deterministic row per IPO.
    out = raw.groupby(KEYS, as_index=False).agg(
        final_return=("final_return", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        turnover=("turnover", "mean"),
    )
    return out.rename(columns={
        "final_return": f"final_return_{label}",
        "max_drawdown": f"max_drawdown_{label}",
        "turnover": f"turnover_{label}",
    })


def load_seed_event(path: Path, label: str, horizon: int,
                    policies: list[str] | None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.loc[raw.horizon == horizon].copy()
    if policies:
        raw = raw.loc[raw.policy.isin(policies)].copy()
    seed_keys = KEYS + ["seed"]
    out = raw.groupby(seed_keys, as_index=False).agg(
        final_return=("final_return", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        turnover=("turnover", "mean"),
    )
    return out.rename(columns={
        "final_return": f"final_return_{label}",
        "max_drawdown": f"max_drawdown_{label}",
        "turnover": f"turnover_{label}",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="per_event_seed_results.csv for the 2021 training start")
    parser.add_argument("--expanded", type=Path, required=True,
                        help="per_event_seed_results.csv for the 2015 training start")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=30,
                        help="Predeclared holding horizon to compare (default: primary 30 sessions)")
    parser.add_argument("--policies", nargs="+",
                        help="Optional exact policy names to include in the paired comparison")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    base = load_event_average(args.baseline, "2021")
    expanded = load_event_average(args.expanded, "2015")
    base = base.loc[base.horizon == args.horizon].copy()
    expanded = expanded.loc[expanded.horizon == args.horizon].copy()
    if args.policies:
        base = base.loc[base.policy.isin(args.policies)].copy()
        expanded = expanded.loc[expanded.policy.isin(args.policies)].copy()
    keys = KEYS
    if base.duplicated(keys).any() or expanded.duplicated(keys).any():
        raise ValueError("Duplicate event/policy keys; seed averaging failed")
    paired = base.merge(expanded, on=keys, how="outer", indicator=True, validate="one_to_one")
    if not (paired["_merge"] == "both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"The two runs do not have identical test keys: {counts}")
    paired = paired.drop(columns="_merge")
    paired["return_delta_2015_minus_2021"] = paired.final_return_2015 - paired.final_return_2021
    paired["drawdown_delta_2015_minus_2021"] = paired.max_drawdown_2015 - paired.max_drawdown_2021
    paired["turnover_delta_2015_minus_2021"] = paired.turnover_2015 - paired.turnover_2021

    rng = np.random.default_rng(20261027)
    metrics = ["final_return", "max_drawdown", "turnover"]
    summary = []
    group_cols = ["horizon", "fee_bps"]
    groups = [(h, fee, "pooled", g) for (h, fee), g in paired.groupby(group_cols, sort=True)]
    groups += [(h, fee, int(year), yr) for (h, fee), g in paired.groupby(group_cols, sort=True)
               for year, yr in g.groupby("test_year", sort=True)]
    for horizon, fee, year_label, g in groups:
        for policy, p in g.groupby("policy", sort=True):
            row = {"horizon": int(horizon), "test_year": year_label,
                   "fee_bps": float(fee), "policy": policy, "paired_events": p.event_id.nunique()}
            for metric in metrics:
                delta = (p[f"{metric}_2015"] - p[f"{metric}_2021"]).to_numpy(float)
                if year_label == "pooled":
                    # Stratified bootstrap preserves the test-year weights.
                    draws = []
                    for _, year_group in p.groupby("test_year", sort=True):
                        values = (year_group[f"{metric}_2015"] - year_group[f"{metric}_2021"]).to_numpy(float)
                        ix = rng.integers(0, len(values), size=(args.bootstrap, len(values)))
                        draws.append(values[ix])
                    boot = np.concatenate(draws, axis=1).mean(axis=1)
                else:
                    ix = rng.integers(0, len(delta), size=(args.bootstrap, len(delta)))
                    boot = delta[ix].mean(axis=1)
                row[f"{metric}_delta"] = float(delta.mean())
                row[f"{metric}_ci_low"] = float(np.quantile(boot, .025))
                row[f"{metric}_ci_high"] = float(np.quantile(boot, .975))
            summary.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_event_deltas.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output_dir / "training_history_comparison_ci.csv", index=False)

    # Separate random-seed uncertainty from event uncertainty. First compute
    # paired out-of-sample metrics for each matched training seed, then
    # bootstrap those seed-level effects.
    base_seed = load_seed_event(args.baseline, "2021", args.horizon, args.policies)
    expanded_seed = load_seed_event(args.expanded, "2015", args.horizon, args.policies)
    seed_key_columns = KEYS + ["seed"]
    seed_paired = base_seed.merge(expanded_seed, on=seed_key_columns, how="outer",
                                  indicator=True, validate="one_to_one")
    if not (seed_paired["_merge"] == "both").all():
        raise ValueError(
            "The two runs do not have identical event/seed keys after deduplication: "
            f"{seed_paired['_merge'].value_counts().to_dict()}"
        )
    seed_paired = seed_paired.drop(columns="_merge")
    for metric in metrics:
        seed_paired[f"{metric}_delta"] = seed_paired[f"{metric}_2015"] - seed_paired[f"{metric}_2021"]
    seed_level = []
    seed_groups = [(h, fee, "pooled", g) for (h, fee), g in seed_paired.groupby(["horizon", "fee_bps"], sort=True)]
    seed_groups += [(h, fee, int(year), yr) for (h, fee), g in seed_paired.groupby(["horizon", "fee_bps"], sort=True)
                    for year, yr in g.groupby("test_year", sort=True)]
    for horizon, fee, year_label, g in seed_groups:
        if year_label != "pooled":
            g = g.loc[g.test_year == year_label]
        for policy, p in g.groupby("policy", sort=True):
            per_seed = p.groupby("seed", as_index=False)[[f"{m}_delta" for m in metrics]].mean()
            row = {"horizon": int(horizon), "test_year": year_label,
                   "fee_bps": float(fee), "policy": policy,
                   "matched_seeds": int(per_seed.seed.nunique())}
            for metric in metrics:
                values = per_seed[f"{metric}_delta"].to_numpy(float)
                ix = rng.integers(0, len(values), size=(args.bootstrap, len(values)))
                boot = values[ix].mean(axis=1)
                row[f"{metric}_delta_mean"] = float(values.mean())
                row[f"{metric}_delta_median"] = float(np.median(values))
                row[f"{metric}_delta_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_delta_seed_ci_low"] = float(np.quantile(boot, .025))
                row[f"{metric}_delta_seed_ci_high"] = float(np.quantile(boot, .975))
                row[f"{metric}_delta_min"] = float(values.min())
                row[f"{metric}_delta_max"] = float(values.max())
            row["positive_return_seed_fraction"] = float((per_seed.final_return_delta > 0).mean())
            row["drawdown_improved_seed_fraction"] = float((per_seed.max_drawdown_delta < 0).mean())
            seed_level.append(row)
    seed_paired.to_csv(args.output_dir / "paired_event_seed_deltas.csv", index=False)
    pd.DataFrame(seed_level).to_csv(args.output_dir / "seed_stability_summary.csv", index=False)
    print(f"Wrote {args.output_dir}; paired rows={len(paired)}; events={paired.event_id.nunique()}")


if __name__ == "__main__":
    main()
