"""IPO-event bootstrap for the DQN sample-size/capacity sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ci_mean(x, rng, n_boot):
    x = np.asarray(x, dtype=float)
    indices = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = x[indices].mean(axis=1)
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-event", type=Path, default=Path("reports/hkipo_dqn_sample_complexity/per_event.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/hkipo_dqn_sample_complexity/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/hkipo_dqn_sample_complexity/bootstrap.csv"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    data = pd.read_csv(args.per_event)
    summ = pd.read_csv(args.summary)
    rng = np.random.default_rng(args.seed)
    rows = []
    keys = ["split", "train_events", "architecture"]
    for key, sub in data.groupby(keys):
        split, n_events, arch = key
        event_policy = sub.groupby(["event_id", "policy"], as_index=False).agg(
            final_return=("final_return", "mean"), max_drawdown=("max_drawdown", "mean"))
        p_ret = event_policy.pivot(index="event_id", columns="policy", values="final_return")
        p_dd = event_policy.pivot(index="event_id", columns="policy", values="max_drawdown")
        dqn = p_ret["custom_DQN"].dropna()
        dqn_dd = p_dd.loc[dqn.index, "custom_DQN"]
        ret_ci = ci_mean(dqn.to_numpy(), rng, args.bootstrap)
        dd_ci = ci_mean(dqn_dd.to_numpy(), rng, args.bootstrap)
        row = {"split": split, "train_events": int(n_events), "architecture": arch,
               "events": len(dqn), "dqn_mean_return": float(dqn.mean()),
               "dqn_return_ci_low": ret_ci[0], "dqn_return_ci_high": ret_ci[1],
               "dqn_mean_max_drawdown": float(dqn_dd.mean()),
               "dqn_drawdown_ci_low": dd_ci[0], "dqn_drawdown_ci_high": dd_ci[1]}
        if "buy_and_hold" in p_ret:
            common = p_ret.dropna(subset=["custom_DQN", "buy_and_hold"]).index
            diff = (p_ret.loc[common, "custom_DQN"] - p_ret.loc[common, "buy_and_hold"]).to_numpy()
            dd_improve = (p_dd.loc[common, "buy_and_hold"] - p_dd.loc[common, "custom_DQN"]).to_numpy()
            rci = ci_mean(diff, rng, args.bootstrap)
            dci = ci_mean(dd_improve, rng, args.bootstrap)
            row.update({"dqn_minus_buyhold_mean_return": float(diff.mean()),
                        "return_diff_ci_low": rci[0], "return_diff_ci_high": rci[1],
                        "buyhold_minus_dqn_drawdown": float(dd_improve.mean()),
                        "drawdown_improvement_ci_low": dci[0], "drawdown_improvement_ci_high": dci[1]})
        rows.append(row)
    result = pd.DataFrame(rows)
    seed_var = (summ[summ.policy.eq("custom_DQN")].groupby(keys).agg(
        seed_mean_return_std=("mean_return", "std"),
        seed_mean_drawdown_std=("mean_max_drawdown", "std")).reset_index())
    result = result.merge(seed_var, on=keys, how="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[result.split.str.startswith("test")][[
        "split", "train_events", "architecture", "dqn_mean_return",
        "dqn_mean_max_drawdown", "dqn_minus_buyhold_mean_return",
        "return_diff_ci_low", "return_diff_ci_high",
        "buyhold_minus_dqn_drawdown", "drawdown_improvement_ci_low",
        "drawdown_improvement_ci_high", "seed_mean_return_std"]]
          .round(4).to_string(index=False))
    print("\nWrote", args.output)


if __name__ == "__main__":
    main()
