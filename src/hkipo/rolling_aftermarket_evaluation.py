"""Retrospective expanding-window evaluation for HK IPO aftermarket DQNs.

For each test listing year, train only on earlier IPO events. The evaluation
years are historical and have been seen in prior exploratory work, so this is
an exploratory rolling-origin analysis, not a blinded confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN as SB3DQN

from .dqn_aftermarket import DQN, load_episodes, run_episode
from .finrl_dqn_baseline import IPOTradingEnv


def _write_outputs(frame: pd.DataFrame, out: Path, bootstrap: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "rolling_per_event.csv", index=False)
    rng = np.random.default_rng(seed)
    fee_rows, paired_rows = [], []
    groups = [(int(year), group) for year, group in frame.groupby("test_year", sort=True)]
    groups.append(("pooled_2023_2026", frame))
    for test_year, group in groups:
        agg = group.groupby(["test_year", "event_id", "policy", "fee_bps"], as_index=False).agg(
            final_return=("final_return", "mean"), max_drawdown=("max_drawdown", "mean"),
            turnover=("turnover", "mean"))
        for (policy, fee), p in agg.groupby(["policy", "fee_bps"], sort=True):
            vals = p[["final_return", "max_drawdown", "turnover"]].to_numpy(float)
            if test_year == "pooled_2023_2026":
                year_indices = [
                    np.flatnonzero(p.test_year.to_numpy() == year)
                    for year in sorted(p.test_year.unique())
                ]
                samples = np.concatenate([
                    idx[rng.integers(0, len(idx), size=(bootstrap, len(idx)))]
                    for idx in year_indices
                ], axis=1)
            else:
                samples = rng.integers(0, len(p), size=(bootstrap, len(p)))
            ret_samples = vals[samples, 0].mean(axis=1)
            dd_samples = vals[samples, 1].mean(axis=1)
            fee_rows.append({
                "test_year": test_year, "events": len(p), "policy": policy,
                "fee_bps": float(fee), "mean_return": float(vals[:, 0].mean()),
                "median_return": float(np.median(vals[:, 0])),
                "win_rate": float((vals[:, 0] > 0).mean()),
                "mean_return_ci_low": float(np.quantile(ret_samples, .025)),
                "mean_return_ci_high": float(np.quantile(ret_samples, .975)),
                "mean_max_drawdown": float(vals[:, 1].mean()),
                "mean_max_drawdown_ci_low": float(np.quantile(dd_samples, .025)),
                "mean_max_drawdown_ci_high": float(np.quantile(dd_samples, .975)),
                "mean_turnover": float(vals[:, 2].mean()),
            })
        for fee, p in agg.groupby("fee_bps", sort=True):
            wide = p.pivot(index=["test_year", "event_id"], columns="policy", values="final_return")
            dd = p.pivot(index=["test_year", "event_id"], columns="policy", values="max_drawdown")
            for comparator in ("buy_and_hold", "FinRL_SB3_DQN"):
                if not {"custom_DQN_medium", comparator}.issubset(wide.columns):
                    continue
                pair = wide[["custom_DQN_medium", comparator]].dropna()
                ddraw = dd.loc[pair.index, ["custom_DQN_medium", comparator]]
                ret_diff = (pair.custom_DQN_medium - pair[comparator]).to_numpy()
                # Positive means custom DQN has lower mean event maximum drawdown.
                dd_benefit = (ddraw[comparator] - ddraw.custom_DQN_medium).to_numpy()
                if test_year == "pooled_2023_2026":
                    year_indices = [
                        np.flatnonzero(pair.index.get_level_values("test_year").to_numpy() == year)
                        for year in sorted(pair.index.get_level_values("test_year").unique())
                    ]
                    samples = np.concatenate([
                        idx[rng.integers(0, len(idx), size=(bootstrap, len(idx)))]
                        for idx in year_indices
                    ], axis=1)
                else:
                    samples = rng.integers(0, len(pair), size=(bootstrap, len(pair)))
                r = ret_diff[samples].mean(axis=1)
                d = dd_benefit[samples].mean(axis=1)
                paired_rows.append({
                    "test_year": test_year, "fee_bps": float(fee),
                    "comparator": comparator, "paired_events": len(pair),
                    "mean_return_difference_custom_minus_comparator": float(ret_diff.mean()),
                    "return_ci_low": float(np.quantile(r, .025)),
                    "return_ci_high": float(np.quantile(r, .975)),
                    "mean_drawdown_improvement_comparator_minus_custom": float(dd_benefit.mean()),
                    "drawdown_ci_low": float(np.quantile(d, .025)),
                    "drawdown_ci_high": float(np.quantile(d, .975)),
                })

    pd.DataFrame(fee_rows).to_csv(out / "rolling_event_metrics_ci.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "rolling_paired_comparisons_ci.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hkipo_event_panel_2021_2026.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hkipo_dqn_rolling_aftermarket"))
    parser.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=6,
                        help="Fixed per-training-IPO budget for both DQN implementations")
    parser.add_argument("--fees-bps", type=float, nargs="+", default=[10, 25, 50])
    parser.add_argument("--train-fee-bps", type=float, default=25)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    all_events: list[pd.DataFrame] = []
    fold_rows = []
    for test_year in args.test_years:
        train_years = set(range(2021, test_year))
        train = load_episodes(args.panel, train_years)
        test = load_episodes(args.panel, {test_year})
        if not train or not test:
            raise ValueError(f"Empty expanding-window fold: train<{test_year}, test={test_year}")
        train_ids = {str(ep.listing_event_id.iloc[0]) for ep in train}
        test_ids = {str(ep.listing_event_id.iloc[0]) for ep in test}
        if train_ids.intersection(test_ids):
            raise AssertionError(f"IPO event leakage in test year {test_year}")
        steps = args.epochs * len(train) * 29
        fold_rows.append({"test_year": test_year, "train_years": sorted(train_years),
                          "train_events": len(train), "test_events": len(test),
                          "steps_per_seed": steps})
        print(f"Fold test={test_year}, train_events={len(train)}, test_events={len(test)}, steps={steps}", flush=True)

        for model_seed in args.seeds:
            custom = DQN(model_seed, args.train_fee_bps / 10000, hidden_dims=(64, 64))
            custom.fit(train, epochs=args.epochs)
            env = IPOTradingEnv(train, args.train_fee_bps / 10000, model_seed)
            finrl = SB3DQN(
                "MlpPolicy", env, learning_rate=1e-4, buffer_size=1_000_000,
                learning_starts=100, batch_size=32, tau=1.0, gamma=0.99,
                train_freq=4, gradient_steps=1, target_update_interval=10_000,
                exploration_fraction=0.1, exploration_initial_eps=1.0,
                exploration_final_eps=0.05, max_grad_norm=10,
                policy_kwargs={"net_arch": [64, 64]}, verbose=0,
                seed=model_seed, device="cpu")
            finrl.learn(total_timesteps=steps, progress_bar=False)
            env.close()

            def finrl_policy(state, _i, _ep):
                action, _ = finrl.predict(state, deterministic=True)
                return int(action)

            policies = {"custom_DQN_medium": custom.greedy, "FinRL_SB3_DQN": finrl_policy}
            for policy_name, policy in policies.items():
                for fee_bps in args.fees_bps:
                    for ep in test:
                        result = run_episode(ep, policy, fee_bps / 10000)
                        all_events.append(pd.DataFrame([{
                            "test_year": test_year, "train_events": len(train),
                            "seed": model_seed, "event_id": result.event_id,
                            "year": result.year, "policy": policy_name,
                            "fee_bps": fee_bps, "final_return": result.final_return,
                            "max_drawdown": result.max_drawdown, "turnover": result.turnover,
                        }]))

        # Deterministic buy-and-hold is written once per IPO and cost scenario.
        for fee_bps in args.fees_bps:
            for ep in test:
                result = run_episode(ep, lambda _s, _i, _ep: 4, fee_bps / 10000)
                all_events.append(pd.DataFrame([{
                    "test_year": test_year, "train_events": len(train), "seed": -1,
                    "event_id": result.event_id, "year": result.year,
                    "policy": "buy_and_hold", "fee_bps": fee_bps,
                    "final_return": result.final_return, "max_drawdown": result.max_drawdown,
                    "turnover": result.turnover,
                }]))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(all_events, ignore_index=True)
    frame.to_csv(out / "rolling_per_event.csv", index=False)
    _write_outputs(frame, out, args.bootstrap, seed=20261027)
    metadata = {
        "analysis": "retrospective expanding-window rolling-origin evaluation",
        "confirmation_status": "exploratory; all historical periods have been inspected in prior experiments",
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "python_packages": {"numpy": np.__version__, "pandas": pd.__version__,
                            "stable_baselines3": __import__("stable_baselines3").__version__},
        "folds": fold_rows,
        "custom_dqn": {"candidate": "validation-selected Medium 64x64", "hidden_dims": [64, 64],
                        "epochs_per_training_event": args.epochs,
                        "learning_rate": 0.0005, "gamma": 0.97, "replay_capacity": 30000,
                        "batch_size": 64, "gradient_update_every_steps": 32,
                        "training_fee_bps_per_side": args.train_fee_bps, "seeds": args.seeds},
        "finrl_dqn": {"implementation": "Stable-Baselines3 standard DQN in matched IPO environment",
                       "network": [64, 64], "same_environment_steps_as_custom": True,
                       "training_fee_bps_per_side": args.train_fee_bps, "seeds": args.seeds},
        "evaluation_fee_bps_per_side": args.fees_bps,
        "bootstrap": {"replicates": args.bootstrap, "unit": "IPO event; random seeds averaged within event",
                      "interval": "percentile 95% CI", "limitation": "does not account for cross-IPO market-regime dependence"},
        "execution": "observe close t, rebalance at open t+1, mark at close t+1; 30-session episode",
        "data_limitations": "Yahoo Finance daily bars, fractional shares, proportional cost proxy; no realistic lot size, opening auction fill, slippage, or market impact",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out}; events={frame.event_id.nunique()}, rows={len(frame)}")


if __name__ == "__main__":
    main()
