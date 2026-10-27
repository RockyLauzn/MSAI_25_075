"""IPO-cluster paired bootstrap and transaction-cost sensitivity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    samples = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[samples].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/hkipo_dqn_aftermarket/per_event.csv"))
    parser.add_argument("--finrl-input", type=Path,
                        default=Path("reports/hkipo_dqn_aftermarket/finrl_dqn_per_event.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hkipo_dqn_aftermarket"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    if args.finrl_input.exists():
        df = pd.concat([df, pd.read_csv(args.finrl_input)], ignore_index=True)
    required = {"event_id", "split", "fee_bps", "policy", "final_return", "max_drawdown", "turnover", "seed"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Per-event file missing columns: {sorted(missing)}")
    rng = np.random.default_rng(args.seed)
    event_rows, paired_rows, finrl_paired_rows = [], [], []

    for split in sorted(df.split.unique()):
        for fee in sorted(df.loc[df.split.eq(split), "fee_bps"].unique()):
            subset = df.loc[df.split.eq(split) & df.fee_bps.eq(fee)]
            # Aggregate seed repetitions within each IPO first. Bootstrap units
            # are IPO events, never daily bars or repeated random seeds.
            agg = subset.groupby(["event_id", "policy"], as_index=False).agg(
                final_return=("final_return", "mean"),
                max_drawdown=("max_drawdown", "mean"),
                turnover=("turnover", "mean"))
            for policy, group in agg.groupby("policy"):
                returns = group.final_return.to_numpy()
                dd = group.max_drawdown.to_numpy()
                turnover = group.turnover.to_numpy()
                ret_ci = mean_ci(returns, rng, args.bootstrap)
                dd_ci = mean_ci(dd, rng, args.bootstrap)
                event_rows.append({
                    "split": split, "fee_bps": float(fee), "policy": policy,
                    "events": len(group), "mean_return": float(returns.mean()),
                    "mean_return_ci_low": ret_ci[0], "mean_return_ci_high": ret_ci[1],
                    "median_return": float(np.median(returns)),
                    "win_rate": float((returns > 0).mean()),
                    "mean_max_drawdown": float(dd.mean()),
                    "mean_max_drawdown_ci_low": dd_ci[0], "mean_max_drawdown_ci_high": dd_ci[1],
                    "mean_turnover": float(turnover.mean()),
                })

            ret_p = agg.pivot(index="event_id", columns="policy", values="final_return")
            dd_p = agg.pivot(index="event_id", columns="policy", values="max_drawdown")
            if {"DQN", "buy_and_hold"}.issubset(ret_p.columns):
                ids = ret_p.dropna(subset=["DQN", "buy_and_hold"]).index.intersection(
                    dd_p.dropna(subset=["DQN", "buy_and_hold"]).index)
                ret_diff = (ret_p.loc[ids, "DQN"] - ret_p.loc[ids, "buy_and_hold"]).to_numpy()
                # Positive drawdown improvement means DQN had a smaller drawdown.
                dd_improvement = (dd_p.loc[ids, "buy_and_hold"] - dd_p.loc[ids, "DQN"]).to_numpy()
                win_diff = ((ret_p.loc[ids, "DQN"] > 0).astype(float)
                            - (ret_p.loc[ids, "buy_and_hold"] > 0).astype(float)).to_numpy()
                rci = mean_ci(ret_diff, rng, args.bootstrap)
                dci = mean_ci(dd_improvement, rng, args.bootstrap)
                wci = mean_ci(win_diff, rng, args.bootstrap)
                paired_rows.append({
                    "split": split, "fee_bps": float(fee), "paired_events": len(ids),
                    "mean_return_diff_DQN_minus_buy_hold": float(ret_diff.mean()),
                    "return_diff_ci_low": rci[0], "return_diff_ci_high": rci[1],
                    "mean_drawdown_improvement_bh_minus_DQN": float(dd_improvement.mean()),
                    "drawdown_improvement_ci_low": dci[0], "drawdown_improvement_ci_high": dci[1],
                    "win_rate_diff_DQN_minus_buy_hold": float(win_diff.mean()),
                    "win_rate_diff_ci_low": wci[0], "win_rate_diff_ci_high": wci[1],
                })
            if {"DQN", "FinRL_SB3_DQN"}.issubset(ret_p.columns):
                ids = ret_p.dropna(subset=["DQN", "FinRL_SB3_DQN"]).index.intersection(
                    dd_p.dropna(subset=["DQN", "FinRL_SB3_DQN"]).index)
                ret_diff = (ret_p.loc[ids, "DQN"] - ret_p.loc[ids, "FinRL_SB3_DQN"]).to_numpy()
                dd_improvement = (dd_p.loc[ids, "FinRL_SB3_DQN"] - dd_p.loc[ids, "DQN"]).to_numpy()
                win_diff = ((ret_p.loc[ids, "DQN"] > 0).astype(float)
                            - (ret_p.loc[ids, "FinRL_SB3_DQN"] > 0).astype(float)).to_numpy()
                rci = mean_ci(ret_diff, rng, args.bootstrap)
                dci = mean_ci(dd_improvement, rng, args.bootstrap)
                wci = mean_ci(win_diff, rng, args.bootstrap)
                finrl_paired_rows.append({
                    "split": split, "fee_bps": float(fee), "paired_events": len(ids),
                    "mean_return_diff_custom_DQN_minus_FinRL_SB3_DQN": float(ret_diff.mean()),
                    "return_diff_ci_low": rci[0], "return_diff_ci_high": rci[1],
                    "mean_drawdown_improvement_FinRL_minus_custom_DQN": float(dd_improvement.mean()),
                    "drawdown_improvement_ci_low": dci[0], "drawdown_improvement_ci_high": dci[1],
                    "win_rate_diff_custom_DQN_minus_FinRL_SB3_DQN": float(win_diff.mean()),
                    "win_rate_diff_ci_low": wci[0], "win_rate_diff_ci_high": wci[1],
                })

    event_metrics = pd.DataFrame(event_rows)
    paired = pd.DataFrame(paired_rows)
    finrl_paired = pd.DataFrame(finrl_paired_rows)
    event_metrics.to_csv(args.output_dir / "cost_sensitivity_bootstrap.csv", index=False)
    paired.to_csv(args.output_dir / "paired_bootstrap_dqn_vs_buyhold.csv", index=False)
    finrl_paired.to_csv(args.output_dir / "paired_bootstrap_custom_dqn_vs_finrl_dqn.csv", index=False)
    summary = {
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "IPO event; DQN random-seed repetitions averaged within IPO before resampling",
        "confidence_interval": "percentile 95% interval",
        "cost_sensitivity": "same DQN parameters trained at 25 bps per side, re-evaluated at each cost; no retraining by cost",
        "paired_comparison": "DQN minus buy-and-hold return and win rate; buy-and-hold minus DQN maximum drawdown (positive favors DQN)",
        "paired_results": paired.to_dict(orient="records"),
        "custom_dqn_vs_finrl_sb3_dqn_results": finrl_paired.to_dict(orient="records"),
    }
    (args.output_dir / "robustness_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Paired bootstrap (IPO-level):")
    print(paired.round(4).to_string(index=False))
    print("\nCost sensitivity (IPO-level mean-return intervals):")
    print(event_metrics[["split", "fee_bps", "policy", "events", "mean_return",
                         "mean_return_ci_low", "mean_return_ci_high", "mean_max_drawdown"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
