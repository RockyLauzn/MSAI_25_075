"""Seed-stability replay for candidates selected on the 2024 validation IPOs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .dqn_aftermarket import DQN, load_episodes, run_episode, summarize


CANDIDATES = {
    # Chosen using 2024 validation only: smallest architecture under a higher
    # exposure budget, plus the lower-drawdown medium architecture at equal steps.
    "small_32_full_12epochs": {"hidden_dims": (32,), "epochs": 12},
    "medium_64x64_full_6epochs": {"hidden_dims": (64, 64), "epochs": 6},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hkipo_event_panel_2021_2026.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hkipo_dqn_candidate_seed_stability"))
    parser.add_argument("--fee-bps", type=float, default=25)
    parser.add_argument("--seed-start", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    train = load_episodes(args.panel, {2021, 2022, 2023})
    splits = {"validation_2024": load_episodes(args.panel, {2024}),
              "retrospective_test_2025": load_episodes(args.panel, {2025}),
              "retrospective_test_2026": load_episodes(args.panel, {2026})}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows, per_event = [], []

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for candidate, config in CANDIDATES.items():
            model = DQN(seed, args.fee_bps / 10000, hidden_dims=config["hidden_dims"])
            model.fit(train, epochs=config["epochs"])
            model.q.eval()
            torch_path = model_dir / f"{candidate}_seed_{seed}.pt"
            import torch
            torch.save(model.q.state_dict(), torch_path)

            for split, episodes in splits.items():
                results = [run_episode(ep, model.greedy, args.fee_bps / 10000) for ep in episodes]
                rows.append({"seed": seed, "candidate": candidate, "split": split,
                             "train_events": len(train), "epochs": config["epochs"],
                             "hidden_dims": "x".join(map(str, config["hidden_dims"])),
                             "fee_bps": args.fee_bps, "policy": "custom_DQN", **summarize(results)})
                per_event.extend({"seed": seed, "candidate": candidate, "split": split,
                                  "train_events": len(train), "epochs": config["epochs"],
                                  "fee_bps": args.fee_bps, "policy": "custom_DQN", **result.__dict__}
                                 for result in results)
                bh = [run_episode(ep, lambda _s, _i, _ep: 4, args.fee_bps / 10000) for ep in episodes]
                rows.append({"seed": seed, "candidate": candidate, "split": split,
                             "train_events": len(train), "epochs": config["epochs"],
                             "hidden_dims": "x".join(map(str, config["hidden_dims"])),
                             "fee_bps": args.fee_bps, "policy": "buy_and_hold", **summarize(bh)})
                per_event.extend({"seed": seed, "candidate": candidate, "split": split,
                                  "train_events": len(train), "epochs": config["epochs"],
                                  "fee_bps": args.fee_bps, "policy": "buy_and_hold", **result.__dict__}
                                 for result in bh)

    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(per_event).to_csv(args.output_dir / "per_event.csv", index=False)
    metadata = {
        "candidate_selection": "2024 validation only; small 32x1 under 12 epochs and medium 64x64 under fixed ~40k interactions",
        "not_a_fresh_confirmation": "2025/2026 aggregates were inspected in prior exploratory work; these new seeds measure seed stability only",
        "future_confirmation_requirement": "Reserve future IPO listing events not seen in any prior experiment",
        "train_years": [2021, 2022, 2023], "validation_year": 2024,
        "retrospective_test_years": [2025, 2026],
        "training_events": len(train), "training_seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
        "fee_bps_per_side": args.fee_bps, "candidates": {
            name: {"hidden_dims": list(cfg["hidden_dims"]), "epochs": cfg["epochs"],
                   "transitions_per_seed": cfg["epochs"] * len(train) * 29}
            for name, cfg in CANDIDATES.items()},
        "execution_and_states": "Same as dqn_aftermarket environment",
        "interpretation": "Exploratory seed-stability and retrospective robustness; not a new untouched test",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).query("policy == 'custom_DQN'").groupby(
        ["split", "candidate"])[["mean_return", "median_return", "win_rate",
        "mean_max_drawdown", "mean_turnover"]].agg(["mean", "std"]).round(4).to_string())
    print("\nWrote", args.output_dir)


if __name__ == "__main__":
    main()
