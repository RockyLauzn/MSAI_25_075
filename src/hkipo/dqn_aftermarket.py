"""Exploratory percentage-action DQN for Hong Kong IPO aftermarket trading.

The agent observes each IPO after its daily close and rebalances at the next
session's open. This timing convention prevents same-bar close lookahead.
Each IPO is an independent 30-session episode with a normalized cash account.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)


TARGET_WEIGHTS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
PRICE_FEATURES = ["premium", "ret1", "ret5", "intraday_range", "volume_change",
                  "rv5", "age", "position_weight"]
IPO_FEATURES = ["ipo_price_log", "sponsor_count_scaled", "prospectus_lead_days_scaled"]
# Keep the original state schema as the default for existing checkpoints and
# prospective logs. New IPO-feature experiments opt in explicitly.
FEATURES = PRICE_FEATURES


def state_features(include_ipo_features: bool = False) -> list[str]:
    return PRICE_FEATURES + IPO_FEATURES if include_ipo_features else PRICE_FEATURES


def load_episodes(path: Path, years: set[int], horizon: int = 30) -> list[pd.DataFrame]:
    if horizon < 2:
        raise ValueError("horizon must be at least 2 sessions")
    df = pd.read_csv(path, parse_dates=["date", "listing_date"])
    df["year"] = df.listing_date.dt.year
    df = df.loc[df.year.isin(years)].sort_values(["listing_event_id", "date"])
    episodes: list[pd.DataFrame] = []
    for _, group in df.groupby("listing_event_id", sort=False):
        group = group.reset_index(drop=True)
        # Keep only complete episodes of the selected common horizon.
        if len(group) < horizon:
            continue
        group = group.iloc[:horizon].copy()
        episodes.append(add_event_features(group))
    return episodes


def add_event_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute the model's daily features from this event's available rows only."""
    group = group.sort_values("date").reset_index(drop=True).copy()
    close = group.close.to_numpy(dtype=float)
    volume = group.volume.to_numpy(dtype=float)
    ret1 = np.r_[0.0, close[1:] / close[:-1] - 1]
    ret5 = np.r_[np.zeros(min(5, len(close))), close[5:] / close[:-5] - 1]
    rv5 = pd.Series(ret1).rolling(5, min_periods=3).std().fillna(0).to_numpy() * np.sqrt(252)
    vol_change = np.r_[0.0, np.log1p(volume[1:]) - np.log1p(volume[:-1])]
    group["premium"] = np.clip(close / float(group.ipo_price.iloc[0]) - 1, -1, 5)
    group["ret1"] = np.clip(ret1, -1, 3)
    group["ret5"] = np.clip(ret5, -1, 5)
    group["intraday_range"] = np.clip((group.high.to_numpy() - group.low.to_numpy()) / close, 0, 3)
    group["volume_change"] = np.clip(vol_change, -10, 10)
    group["rv5"] = np.clip(rv5, 0, 10)
    group["age"] = group.listed_day.to_numpy(dtype=float) / 30
    ipo_price = max(float(group.ipo_price.iloc[0]), 0.0)
    sponsors = str(group.sponsors.iloc[0]) if "sponsors" in group and pd.notna(group.sponsors.iloc[0]) else ""
    sponsor_count = sponsors.count("/") + 1 if sponsors.strip() else 0
    prospectus_date = pd.to_datetime(group.prospectus_date.iloc[0], errors="coerce") if "prospectus_date" in group else pd.NaT
    listing_date = pd.to_datetime(group.listing_date.iloc[0], errors="coerce")
    lead_days = max((listing_date - prospectus_date).days, 0) if pd.notna(prospectus_date) else 0
    group["ipo_price_log"] = np.clip(np.log1p(ipo_price) / np.log1p(1000.0), 0.0, 1.1)
    group["sponsor_count_scaled"] = np.clip(sponsor_count / 5.0, 0.0, 1.0)
    group["prospectus_lead_days_scaled"] = np.clip(lead_days / 60.0, 0.0, 1.0)
    return group


def observation(row: pd.Series, weight: float, include_ipo_features: bool = False) -> np.ndarray:
    values = [row.premium, row.ret1, row.ret5, row.intraday_range,
              row.volume_change, row.rv5, row.age, weight]
    if include_ipo_features:
        values.extend([row.ipo_price_log, row.sponsor_count_scaled, row.prospectus_lead_days_scaled])
    return np.asarray(values, dtype=np.float32)


@dataclass
class Result:
    event_id: str
    year: int
    final_return: float
    max_drawdown: float
    turnover: float


def run_episode(ep: pd.DataFrame, choose_action, fee: float,
                include_ipo_features: bool = False) -> Result:
    """Observe close t, execute target weight at open t+1, mark at close t+1."""
    cash, shares, peak, max_dd, total_turnover = 1.0, 0.0, 1.0, 0.0, 0.0
    prev_nav = 1.0
    for i in range(len(ep) - 1):
        row = ep.iloc[i]
        close = float(row.close)
        nav_close = cash + shares * close
        weight = (shares * close / nav_close) if nav_close > 0 else 0.0
        state = observation(row, weight, include_ipo_features)
        target = float(TARGET_WEIGHTS[int(choose_action(state, i, ep))])
        nxt = ep.iloc[i + 1]
        open_px, next_close = float(nxt.open), float(nxt.close)
        nav_open = cash + shares * open_px
        wanted_value = target * nav_open
        delta_value = wanted_value - shares * open_px
        trade_value = abs(delta_value)
        cost = fee * trade_value
        cash -= delta_value + cost
        shares += delta_value / open_px
        nav = cash + shares * next_close
        if nav <= 0:
            nav = 1e-12
        peak = max(peak, nav)
        max_dd = max(max_dd, (peak - nav) / peak)
        total_turnover += trade_value / max(prev_nav, 1e-12)
        prev_nav = nav
    return Result(str(ep.listing_event_id.iloc[0]), int(ep.listing_date.iloc[0].year),
                  float(prev_nav - 1), float(max_dd), float(total_turnover))


class QNet(nn.Module):
    def __init__(self, n_features: int, n_actions: int, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        width = n_features
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(0.10)])
            width = hidden
        layers.append(nn.Linear(width, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQN:
    def __init__(self, seed: int, fee: float, hidden_dims=(64, 64),
                 include_ipo_features: bool = False,
                 target_update_interval: int | None = None):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.hidden_dims = tuple(hidden_dims)
        self.include_ipo_features = include_ipo_features
        if target_update_interval is not None and target_update_interval < 1:
            raise ValueError("target_update_interval must be positive")
        self.target_update_interval = target_update_interval
        n_features = len(state_features(include_ipo_features))
        self.q = QNet(n_features, len(TARGET_WEIGHTS), self.hidden_dims).to(self.device)
        self.target = QNet(n_features, len(TARGET_WEIGHTS), self.hidden_dims).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=0.0005)
        self.memory = deque(maxlen=30000)
        self.update_count = 0
        self.gamma, self.fee, self.rng = 0.97, fee, np.random.default_rng(seed)

    def act(self, state: np.ndarray, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(len(TARGET_WEIGHTS)))
        self.q.eval()
        with torch.no_grad():
            return int(self.q(torch.tensor(state).unsqueeze(0)).argmax(dim=1).item())

    def greedy(self, state: np.ndarray, _i: int, _ep: pd.DataFrame) -> int:
        return self.act(state, 0.0)

    def train_step(self, batch_size: int = 128) -> None:
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, done = zip(*batch)
        s = torch.tensor(np.asarray(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float32)
        ns = torch.tensor(np.asarray(ns), dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        self.q.train()
        qval = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            self.q.eval()
            next_a = self.q(ns).argmax(dim=1)
            self.q.train()
            next_q = self.target(ns).gather(1, next_a[:, None]).squeeze(1)
            target = r + self.gamma * (1 - done) * next_q
        loss = nn.functional.smooth_l1_loss(qval, target)
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), 1.0); self.opt.step()

    def fit(self, episodes: list[pd.DataFrame], epochs: int = 30,
            total_transitions: int | None = None,
            exploration_reference_transitions: int | None = None) -> None:
        if not episodes:
            raise ValueError("At least one IPO episode is required")
        transition_budget = (int(total_transitions) if total_transitions is not None
                             else epochs * sum(len(ep) - 1 for ep in episodes))
        if transition_budget < 1:
            raise ValueError("Training transition budget must be positive")
        # Replay from many IPO events; randomize event order each pass. When a
        # fixed interaction budget is supplied, stop exactly at that count.
        epoch = 0
        target_update_count = self.update_count + transition_budget
        while self.update_count < target_update_count:
            eps = max(0.05, 0.8 * (0.97 ** epoch))
            order = self.rng.permutation(len(episodes))
            for idx in order:
                ep = episodes[int(idx)]
                cash, shares = 1.0, 0.0
                for i in range(len(ep) - 1):
                    if exploration_reference_transitions is not None:
                        virtual_epoch = (self.update_count // exploration_reference_transitions)
                        eps = max(0.05, 0.8 * (0.97 ** virtual_epoch))
                    row, nxt = ep.iloc[i], ep.iloc[i + 1]
                    close = float(row.close)
                    nav_close = cash + shares * close
                    weight = shares * close / nav_close if nav_close > 0 else 0.0
                    s = observation(row, weight, self.include_ipo_features)
                    a = self.act(s, eps)
                    open_px, next_close = float(nxt.open), float(nxt.close)
                    nav_open = cash + shares * open_px
                    desired = float(TARGET_WEIGHTS[a]) * nav_open
                    delta = desired - shares * open_px
                    cash -= delta + self.fee * abs(delta)
                    shares += delta / open_px
                    nav = cash + shares * next_close
                    reward = float(np.log(max(nav, 1e-12) / max(nav_close, 1e-12)))
                    done = i == len(ep) - 2
                    ns = observation(nxt, (shares * next_close / nav) if nav > 0 else 0.0,
                                      self.include_ipo_features)
                    self.memory.append((s, a, reward, ns, done))
                    self.update_count += 1
                    if self.update_count % 32 == 0:
                        self.train_step(batch_size=64)
                    if (self.target_update_interval is not None
                            and self.update_count % self.target_update_interval == 0):
                        self.target.load_state_dict(self.q.state_dict())
                        self.target.eval()
                    if self.update_count >= target_update_count:
                        break
            epoch += 1
            if self.target_update_interval is None and epoch % 10 == 0:
                self.target.load_state_dict(self.q.state_dict())
                self.target.eval()


def summarize(results: list[Result]) -> dict:
    rets = np.array([r.final_return for r in results])
    return {"events": len(results), "mean_return": float(rets.mean()),
            "median_return": float(np.median(rets)), "win_rate": float((rets > 0).mean()),
            "mean_max_drawdown": float(np.mean([r.max_drawdown for r in results])),
            "mean_turnover": float(np.mean([r.turnover for r in results]))}


def eval_policy(episodes, policy, fee):
    return [run_episode(ep, policy, fee) for ep in episodes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hkipo_event_panel_2021_2026.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hkipo_dqn_aftermarket"))
    parser.add_argument("--fee", type=float, default=0.0025, help="per-side proportional cost; 0.0025 = 25 bps")
    parser.add_argument("--eval-fees-bps", type=float, nargs="+", default=[10, 25, 50],
                        help="evaluation fee scenarios in basis points per side; policy is trained once at --fee")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    splits = {"train": load_episodes(args.panel, {2021, 2022, 2023}),
              "validation": load_episodes(args.panel, {2024}),
              "test_2025": load_episodes(args.panel, {2025}),
              "test_2026": load_episodes(args.panel, {2026})}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, per_event = [], []
    for seed in range(args.seeds):
        model = DQN(seed, args.fee)
        model.fit(splits["train"], args.epochs)
        policies = {
            "DQN": model.greedy,
            "buy_and_hold": lambda _s, _i, _ep: 4,
            "momentum_1d": lambda s, _i, _ep: 4 if s[1] > 0.01 else (2 if s[1] < -0.03 else 0),
            "mean_reversion_1d": lambda s, _i, _ep: 4 if s[1] < -0.03 else (2 if s[1] > 0.03 else 0),
        }
        for split, eps in splits.items():
            if split == "train":
                continue
            for fee_bps in args.eval_fees_bps:
                for name, policy in policies.items():
                    results = eval_policy(eps, policy, fee_bps / 10000)
                    summary = summarize(results)
                    rows.append({"seed": seed, "split": split, "fee_bps": fee_bps,
                                 "policy": name, **summary})
                    per_event.extend({"seed": seed, "split": split, "fee_bps": fee_bps,
                                      "policy": name, **r.__dict__} for r in results)
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(per_event).to_csv(args.output_dir / "per_event.csv", index=False)
    metadata = {"panel": str(args.panel), "fee_per_side": args.fee, "epochs": args.epochs,
                "seeds": args.seeds, "action_target_weights": TARGET_WEIGHTS.tolist(),
                "evaluation_fee_scenarios_bps_per_side": args.eval_fees_bps,
                "cost_sensitivity_design": "Train each seed once at --fee, then hold learned parameters fixed and re-evaluate at each fee scenario",
                "train_years": [2021, 2022, 2023], "validation_year": 2024,
                "test_years": [2025, 2026], "horizon_sessions": 30,
                "execution": "observe close t, rebalance at open t+1, mark to close t+1",
                "price_data_note": "Yahoo Finance daily OHLCV; turnover is estimated; idealized fractional shares; no intraday liquidity or auction fills",
                "episodes": {key: len(val) for key, val in splits.items()}}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).groupby(["split", "policy"])[["mean_return", "median_return", "win_rate", "mean_max_drawdown"]].mean().round(4).to_string())
    print("\nWrote", args.output_dir)


if __name__ == "__main__":
    main()
