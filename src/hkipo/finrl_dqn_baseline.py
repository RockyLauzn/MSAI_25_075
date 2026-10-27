"""FinRL-compatible Stable-Baselines3 DQN on the matched HK IPO environment.

This uses the standard SB3 DQN algorithm and defaults (with explicit MLP width),
while keeping the IPO-specific state, percentage target-position actions,
execution timing, reward, data splits, and simulator identical to the custom
DQN-P-style experiment. It is an algorithm baseline in a matched environment,
not an unmodified FinRL stock-trading environment or an exact replication of
the Korean paper's undisclosed FinRL configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from stable_baselines3 import DQN as SB3DQN

from .dqn_aftermarket import (
    FEATURES, TARGET_WEIGHTS, load_episodes, observation, run_episode, state_features, summarize,
)


class IPOTradingEnv(gym.Env):
    """Randomized IPO-episode environment with the same transitions as DQN-P."""

    metadata = {"render_modes": []}

    def __init__(self, episodes, fee: float, seed: int, include_ipo_features: bool = False):
        super().__init__()
        if not episodes:
            raise ValueError("At least one training IPO episode is required")
        self.episodes = episodes
        self.fee = fee
        self.include_ipo_features = include_ipo_features
        self.action_space = spaces.Discrete(len(TARGET_WEIGHTS))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(state_features(include_ipo_features)),), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.ep = None
        self.i = 0
        self.cash, self.shares = 1.0, 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.ep = self.episodes[int(self.rng.integers(len(self.episodes)))]
        self.i = 0
        self.cash, self.shares = 1.0, 0.0
        row = self.ep.iloc[self.i]
        return observation(row, 0.0, self.include_ipo_features), {"event_id": str(row.listing_event_id)}

    def step(self, action):
        row = self.ep.iloc[self.i]
        nxt = self.ep.iloc[self.i + 1]
        close = float(row.close)
        nav_close = self.cash + self.shares * close
        weight = self.shares * close / nav_close if nav_close > 0 else 0.0
        # State must reflect actual current holdings; SB3 passes the action
        # selected from the observation returned by the preceding step.
        open_px, next_close = float(nxt.open), float(nxt.close)
        nav_open = self.cash + self.shares * open_px
        delta_value = float(TARGET_WEIGHTS[int(action)]) * nav_open - self.shares * open_px
        self.cash -= delta_value + self.fee * abs(delta_value)
        self.shares += delta_value / open_px
        nav = max(self.cash + self.shares * next_close, 1e-12)
        reward = float(np.log(nav / max(nav_close, 1e-12)))
        self.i += 1
        terminated = self.i >= len(self.ep) - 1
        obs_row = self.ep.iloc[self.i]
        next_weight = self.shares * float(obs_row.close) / nav
        obs = observation(obs_row, next_weight, self.include_ipo_features)
        return obs, reward, terminated, False, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hkipo_event_panel_2021_2026.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hkipo_dqn_aftermarket"))
    parser.add_argument("--fee", type=float, default=0.0025, help="training cost per side")
    parser.add_argument("--eval-fees-bps", type=float, nargs="+", default=[10, 25, 50])
    parser.add_argument("--epochs", type=int, default=12,
                        help="matched interaction budget: epochs * train IPOs * 29 transitions")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--learning-starts", type=int, default=100,
                        help="SB3 DQN current default; exposed in metadata for reproducibility")
    args = parser.parse_args()

    train = load_episodes(args.panel, {2021, 2022, 2023})
    validation = load_episodes(args.panel, {2024})
    test_2025 = load_episodes(args.panel, {2025})
    test_2026 = load_episodes(args.panel, {2026})
    splits = {"validation": validation, "test_2025": test_2025, "test_2026": test_2026}
    total_steps = args.epochs * len(train) * 29
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "finrl_dqn_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows, per_event = [], []

    for seed in range(args.seeds):
        env = IPOTradingEnv(train, args.fee, seed)
        # SB3 DQN's standard hyperparameters, with only MLP width made explicit.
        model = SB3DQN(
            "MlpPolicy", env, learning_rate=1e-4, buffer_size=1_000_000,
            learning_starts=args.learning_starts, batch_size=32, tau=1.0,
            gamma=0.99, train_freq=4, gradient_steps=1,
            target_update_interval=10_000, exploration_fraction=0.1,
            exploration_initial_eps=1.0, exploration_final_eps=0.05,
            max_grad_norm=10, policy_kwargs={"net_arch": [64, 64]},
            verbose=0, seed=seed, device="cpu",
        )
        model.learn(total_timesteps=total_steps, progress_bar=False)
        model.save(model_dir / f"finrl_sb3_dqn_seed_{seed}")
        env.close()

        def policy(state, _i, _ep):
            action, _ = model.predict(state, deterministic=True)
            return int(action)

        for split, episodes in splits.items():
            for fee_bps in args.eval_fees_bps:
                results = [run_episode(ep, policy, fee_bps / 10000) for ep in episodes]
                summary = summarize(results)
                rows.append({"seed": seed, "split": split, "fee_bps": fee_bps,
                             "policy": "FinRL_SB3_DQN", **summary})
                per_event.extend({"seed": seed, "split": split, "fee_bps": fee_bps,
                                  "policy": "FinRL_SB3_DQN", **r.__dict__} for r in results)

    pd.DataFrame(rows).to_csv(args.output_dir / "finrl_dqn_summary.csv", index=False)
    pd.DataFrame(per_event).to_csv(args.output_dir / "finrl_dqn_per_event.csv", index=False)
    config = {
        "baseline_name": "FinRL-compatible SB3 standard DQN",
        "algorithm_implementation": "Stable-Baselines3 DQN; custom IPO environment retained for matched comparison",
        "not_claimed": "Not an unmodified FinRL stock environment and not an exact reproduction of the Korean paper's FinRL baseline settings",
        "data_panel": str(args.panel), "train_years": [2021, 2022, 2023],
        "validation_year": 2024, "test_years": [2025, 2026],
        "episode_count": {"train": len(train), **{k: len(v) for k, v in splits.items()}},
        "episode_horizon_sessions": 30, "training_cost_per_side": args.fee,
        "evaluation_costs_bps_per_side": args.eval_fees_bps,
        "total_timesteps_per_seed": total_steps, "training_seeds": args.seeds,
        "training_episodes_sampled_with_replacement": True,
        "environment": "Same 8 IPO state features, 5 target-weight actions, log-NAV reward, and close-t / open-t+1 / close-t+1 timing as custom DQN experiment",
        "sb3_version": __import__("stable_baselines3").__version__,
        "hyperparameters": {
            "policy": "MlpPolicy", "net_arch": [64, 64], "learning_rate": 1e-4,
            "buffer_size": 1_000_000, "learning_starts": args.learning_starts,
            "batch_size": 32, "tau": 1.0, "gamma": 0.99,
            "train_freq": 4, "gradient_steps": 1, "target_update_interval": 10_000,
            "exploration_fraction": 0.1, "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05, "max_grad_norm": 10,
        },
        "cost_sensitivity": "Model is trained once at training_cost_per_side and frozen for all evaluation cost scenarios",
    }
    (args.output_dir / "finrl_dqn_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).groupby(["split", "fee_bps"])[["mean_return", "median_return",
          "win_rate", "mean_max_drawdown", "mean_turnover"]].mean().round(4).to_string())
    print(f"\nWrote FinRL/SB3 DQN baseline: {args.output_dir}")


if __name__ == "__main__":
    main()
