"""IPO-level paired bootstrap for validation-selected DQN candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ci_mean(x, rng, n_boot):
    x = np.asarray(x, dtype=float)
    draw = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[draw].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/hkipo_dqn_candidate_seed_stability/per_event.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/hkipo_dqn_candidate_seed_stability/bootstrap.csv"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    rng = np.random.default_rng(args.seed)
    rows = []
    for (split, candidate), sub in df.groupby(["split", "candidate"]):
        agg = sub.groupby(["event_id", "policy"], as_index=False).agg(
            ret=("final_return", "mean"), dd=("max_drawdown", "mean"))
        p_ret = agg.pivot(index="event_id", columns="policy", values="ret")
        p_dd = agg.pivot(index="event_id", columns="policy", values="dd")
        r = p_ret["custom_DQN"]
        d = p_dd.loc[r.index, "custom_DQN"]
        rci, dci = ci_mean(r.to_numpy(), rng, args.bootstrap), ci_mean(d.to_numpy(), rng, args.bootstrap)
        diff = (p_ret["custom_DQN"] - p_ret["buy_and_hold"]).dropna()
        dd_improve = (p_dd["buy_and_hold"] - p_dd["custom_DQN"]).dropna()
        diff_ci = ci_mean(diff.to_numpy(), rng, args.bootstrap)
        dd_ci = ci_mean(dd_improve.to_numpy(), rng, args.bootstrap)
        agent_rows = sub[sub.policy.eq("custom_DQN")]
        seed_group = agent_rows.groupby("seed")
        rows.append({"split": split, "candidate": candidate, "events": len(r),
                     "mean_return": float(r.mean()), "return_ci_low": rci[0], "return_ci_high": rci[1],
                     "seed_std_of_mean_return": float(seed_group.final_return.mean().std()),
                     "seed_std_of_mean_drawdown": float(seed_group.max_drawdown.mean().std()),
                     "seed_std_of_mean_turnover": float(seed_group.turnover.mean().std()),
                     "mean_max_drawdown": float(d.mean()), "drawdown_ci_low": dci[0],
                     "drawdown_ci_high": dci[1], "mean_return_diff_vs_buyhold": float(diff.mean()),
                     "return_diff_ci_low": diff_ci[0], "return_diff_ci_high": diff_ci[1],
                     "mean_drawdown_improvement_vs_buyhold": float(dd_improve.mean()),
                     "drawdown_improvement_ci_low": dd_ci[0], "drawdown_improvement_ci_high": dd_ci[1]})
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.round(4).to_string(index=False))
    print("\nWrote", args.output)


if __name__ == "__main__":
    main()
