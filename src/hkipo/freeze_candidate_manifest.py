"""Create an immutable, hash-based manifest for prospective IPO research."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .dqn_aftermarket import FEATURES, TARGET_WEIGHTS


CONFIGS = {
    "small_32_full_12epochs": {"hidden_dims": [32], "epochs": 12, "seeds": list(range(3, 13))},
    "medium_64x64_full_6epochs": {"hidden_dims": [64, 64], "epochs": 6, "seeds": list(range(3, 13))},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("reports/hkipo_dqn_candidate_seed_stability/models"))
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hkipo_event_panel_2021_2026.csv"))
    parser.add_argument("--freeze-date", default="2026-10-27")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/hkipo_dqn_candidate_seed_stability/frozen_policy_manifest.json"))
    args = parser.parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)
    files = {}
    for candidate, config in CONFIGS.items():
        models = []
        for seed in config["seeds"]:
            path = args.model_dir / f"{candidate}_seed_{seed}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing candidate model for seed {seed}: {path}")
            models.append({"seed": seed, "path": str(path.as_posix()), "sha256": sha256(path)})
        files[candidate] = {**config, "ensemble_rule": "majority vote of each member's greedy action; ties choose the lowest exposure action index",
                            "models": models}
    baseline_dir = Path("reports/hkipo_dqn_aftermarket/finrl_dqn_models")
    baseline_members = []
    for seed in range(3):
        path = baseline_dir / f"finrl_sb3_dqn_seed_{seed}.zip"
        if not path.exists():
            raise FileNotFoundError(f"Missing FinRL/SB3 baseline model for seed {seed}: {path}")
        baseline_members.append({"seed": seed, "path": str(path.as_posix()), "sha256": sha256(path)})
    files["finrl_sb3_standard_dqn_baseline"] = {
        "algorithm": "Stable-Baselines3 DQN", "training_steps_per_seed": 80388,
        "ensemble_rule": "majority vote of deterministic greedy actions; ties choose the lowest exposure action index",
        "members": baseline_members,
    }
    panel = pd.read_csv(args.panel, parse_dates=["date", "listing_date"])
    meta = {
        "freeze_date": args.freeze_date,
        "data_cutoff_date": str(panel.date.max().date()),
        "latest_listing_date_in_panel": str(panel.listing_date.max().date()),
        "prospective_listing_date_minimum": args.freeze_date,
        "purpose": "Prospective research signal logging and out-of-sample evaluation only; no brokerage execution",
        "selection_basis": "2024 validation only; no 2025/2026 metrics used to select these candidates",
        "training_years": [2021, 2022, 2023], "validation_year": 2024,
        "previously_inspected_test_years": [2025, 2026],
        "true_confirmation": "Only IPO listing events after prospective_listing_date_minimum that were not in prior research",
        "state_features": FEATURES,
        "actions_target_position_weights": TARGET_WEIGHTS.tolist(),
        "execution_assumption": "observe close t; signal for open t+1; mark at close t+1",
        "fee_bps_per_side": 25,
        "models": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "freeze_date": args.freeze_date,
                      "data_cutoff_date": meta["data_cutoff_date"],
                      "model_counts": {k: len(v.get("models", v.get("members", [])))
                                       for k, v in files.items()}}, indent=2))


if __name__ == "__main__":
    main()
