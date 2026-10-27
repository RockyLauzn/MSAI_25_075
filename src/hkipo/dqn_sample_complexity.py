"""Learning-curve experiment: IPO training sample size vs DQN capacity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dqn_aftermarket import DQN, load_episodes, run_episode, summarize


ARCHITECTURES = {
    "small_32": (32,),
    "medium_64x64": (64, 64),
    "large_128x128": (128, 128),
}


def stratified_nested_samples(episodes, sample_sizes, seed):
    by_year = {}
    for ep in episodes:
        year = int(ep.listing_date.iloc[0].year)
        by_year.setdefault(year, []).append(ep)
    total = len(episodes)
    rng = np.random.default_rng(seed)
    ranked = {year: [events[i] for i in rng.permutation(len(events))]
              for year, events in by_year.items()}
    outputs = {}
    for size in sample_sizes:
        exact = {year: size * len(events) / total for year, events in by_year.items()}
        alloc = {year: math.floor(value) for year, value in exact.items()}
        remainder = size - sum(alloc.values())
        for year in sorted(exact, key=lambda y: exact[y] - alloc[y], reverse=True)[:remainder]:
            alloc[year] += 1
        outputs[size] = [ep for year in sorted(ranked) for ep in ranked[year][:alloc[year]]]
    return outputs


def parameter_count(hidden_dims, n_inputs=8, n_outputs=5):
    widths = [n_inputs, *hidden_dims, n_outputs]
    return sum((widths[i] + 1) * widths[i + 1] for i in range(len(widths) - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hkipo_event_panel_2021_2026.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hkipo_dqn_sample_complexity"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--budget-transitions", type=int, default=None,
                        help="if set, keep total environment transitions approximately fixed across sample sizes")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--fee-bps", type=float, default=25)
    args = parser.parse_args()

    train = load_episodes(args.panel, {2021, 2022, 2023})
    splits = {"validation": load_episodes(args.panel, {2024}),
              "test_2025": load_episodes(args.panel, {2025}),
              "test_2026": load_episodes(args.panel, {2026})}
    sample_sizes = sorted({max(1, round(len(train) * p)) for p in (0.25, 0.50, 1.0)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, per_event = [], []

    for seed in range(args.seeds):
        subsets = stratified_nested_samples(train, sample_sizes, seed + 1000)
        for n_events in sample_sizes:
            subset = subsets[n_events]
            config_epochs = (args.epochs if args.budget_transitions is None
                             else max(1, round(args.budget_transitions / (n_events * 29))))
            config_transitions = config_epochs * n_events * 29
            for arch_name, hidden_dims in ARCHITECTURES.items():
                model = DQN(seed, args.fee_bps / 10000, hidden_dims=hidden_dims)
                model.fit(subset, epochs=config_epochs)
                for split, episodes in splits.items():
                    dqn_results = [run_episode(ep, model.greedy, args.fee_bps / 10000)
                                   for ep in episodes]
                    dqn_metrics = summarize(dqn_results)
                    rows.append({"seed": seed, "train_events": n_events,
                                 "train_fraction": n_events / len(train),
                                 "architecture": arch_name,
                                 "hidden_dims": "x".join(map(str, hidden_dims)),
                                 "parameter_count": parameter_count(hidden_dims),
                                 "epochs": config_epochs, "train_transitions": config_transitions,
                                 "split": split, "policy": "custom_DQN", **dqn_metrics})
                    per_event.extend({"seed": seed, "train_events": n_events,
                                      "train_fraction": n_events / len(train),
                                      "architecture": arch_name,
                                      "parameter_count": parameter_count(hidden_dims),
                                      "split": split, "policy": "custom_DQN", **result.__dict__}
                                     for result in dqn_results)
                    bh_results = [run_episode(ep, lambda _s, _i, _ep: 4, args.fee_bps / 10000)
                                  for ep in episodes]
                    bh_metrics = summarize(bh_results)
                    rows.append({"seed": seed, "train_events": n_events,
                                 "train_fraction": n_events / len(train),
                                 "architecture": arch_name,
                                 "hidden_dims": "x".join(map(str, hidden_dims)),
                                 "parameter_count": parameter_count(hidden_dims),
                                 "epochs": config_epochs, "train_transitions": config_transitions,
                                 "split": split, "policy": "buy_and_hold", **bh_metrics})
                    per_event.extend({"seed": seed, "train_events": n_events,
                                      "train_fraction": n_events / len(train),
                                      "architecture": arch_name,
                                      "parameter_count": parameter_count(hidden_dims),
                                      "split": split, "policy": "buy_and_hold", **result.__dict__}
                                     for result in bh_results)

    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(per_event).to_csv(args.output_dir / "per_event.csv", index=False)
    metadata = {
        "research_question": "How do IPO-level training sample size and DQN network capacity affect cross-IPO generalization?",
        "panel": str(args.panel), "train_years": [2021, 2022, 2023],
        "validation_year": 2024, "test_years": [2025, 2026],
        "training_events_total": len(train), "sample_sizes_stratified_by_listing_year": sample_sizes,
        "nested_samples_within_seed": True, "training_seeds": list(range(args.seeds)),
        "architectures": {name: {"hidden_dims": list(dims),
                                  "parameters": parameter_count(dims)}
                          for name, dims in ARCHITECTURES.items()},
        "budget_mode": "fixed_interactions" if args.budget_transitions is not None else "fixed_epochs_per_ipo",
        "epochs_per_ipo_if_fixed_epoch_mode": args.epochs,
        "target_training_transitions": args.budget_transitions,
        "training_epochs_by_sample_size": {
            str(n): (args.epochs if args.budget_transitions is None
                     else max(1, round(args.budget_transitions / (n * 29))))
            for n in sample_sizes},
        "training_transitions_per_config": {
            str(n): ((args.epochs if args.budget_transitions is None
                      else max(1, round(args.budget_transitions / (n * 29)))) * n * 29)
            for n in sample_sizes},
        "evaluation_cost_bps_per_side": args.fee_bps,
        "matched_state_action_reward_execution": True,
        "selection": "No architecture or sample size selected using test outcomes; results are descriptive learning curves",
        "limitations": ["Only 231 training IPOs", "Repeated event episodes do not create independent issuers",
                        "In fixed-epoch mode, more IPOs also increase total interaction/update opportunities"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = pd.DataFrame(rows)
    print(summary[summary.policy.eq("custom_DQN")].groupby(
        ["split", "train_events", "architecture"])[["mean_return", "median_return",
        "win_rate", "mean_max_drawdown", "mean_turnover"]].mean().round(4).to_string())
    print("\nWrote", args.output_dir)


if __name__ == "__main__":
    main()
