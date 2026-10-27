"""Validation-only comparison of full-training-set DQN capacities."""

from pathlib import Path

import pandas as pd


def main():
    sources = {
        "fixed_12_epochs_per_ipo": Path("reports/hkipo_dqn_sample_complexity/summary.csv"),
        "fixed_40k_interactions": Path("reports/hkipo_dqn_sample_complexity_fixed_budget/summary.csv"),
    }
    frames = []
    for mode, path in sources.items():
        data = pd.read_csv(path)
        data = data.loc[(data.split == "validation") & (data.policy == "custom_DQN")
                        & (data.train_events == 231)].copy()
        data["training_mode"] = mode
        frames.append(data)
    out = pd.concat(frames, ignore_index=True)
    summary = out.groupby(["training_mode", "architecture", "parameter_count"], as_index=False).agg(
        validation_events=("events", "first"),
        mean_return=("mean_return", "mean"), seed_return_std=("mean_return", "std"),
        median_return=("median_return", "mean"), win_rate=("win_rate", "mean"),
        mean_max_drawdown=("mean_max_drawdown", "mean"),
        mean_turnover=("mean_turnover", "mean"))
    target = Path("reports/hkipo_dqn_sample_complexity/validation_candidates.csv")
    summary.to_csv(target, index=False)
    print(summary.round(4).to_string(index=False))
    print("\nWrote", target)


if __name__ == "__main__":
    main()
