"""Combine validation-selected candidates' original and new random seeds."""

from pathlib import Path

import pandas as pd


def _extract(path: Path, architecture: str, candidate: str, split_map: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[(df.train_events == 231) & (df.architecture == architecture)].copy()
    df["candidate"] = candidate
    df["split"] = df.split.map(split_map)
    df["fee_bps"] = 25
    df["policy"] = df.policy.replace({"custom_DQN": "custom_DQN", "buy_and_hold": "buy_and_hold"})
    keep = ["seed", "candidate", "split", "policy", "event_id", "year",
            "train_events", "fee_bps", "final_return", "max_drawdown", "turnover"]
    return df[keep]


def main():
    historic_splits = {"validation": "validation_2024", "test_2025": "retrospective_test_2025",
                       "test_2026": "retrospective_test_2026"}
    old_small = _extract(Path("reports/hkipo_dqn_sample_complexity/per_event.csv"),
                         "small_32", "small_32_full_12epochs", historic_splits)
    old_medium = _extract(Path("reports/hkipo_dqn_sample_complexity_fixed_budget/per_event.csv"),
                          "medium_64x64", "medium_64x64_full_6epochs", historic_splits)
    new = pd.read_csv("reports/hkipo_dqn_candidate_seed_stability/per_event.csv")
    combined = pd.concat([old_small, old_medium, new], ignore_index=True)
    output = Path("reports/hkipo_dqn_candidate_seed_stability/per_event_all_13_seeds.csv")
    combined.to_csv(output, index=False)
    seed_summary = combined.groupby(["seed", "candidate", "split", "policy"], as_index=False).agg(
        events=("event_id", "nunique"), mean_return=("final_return", "mean"),
        median_return=("final_return", "median"), win_rate=("final_return", lambda x: (x > 0).mean()),
        mean_max_drawdown=("max_drawdown", "mean"), mean_turnover=("turnover", "mean"))
    seed_summary.to_csv("reports/hkipo_dqn_candidate_seed_stability/summary_all_13_seeds.csv", index=False)
    print(combined.groupby(["candidate", "policy", "split"]).seed.nunique().to_string())
    print("\nWrote", output)


if __name__ == "__main__":
    main()
