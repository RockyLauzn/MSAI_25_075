"""Exploratory DQN for finite-cash HK IPO one-lot allocation.

This is a small-sample research prototype. It uses expected allotment outcomes,
official maximum application prices, and Yahoo listing-day opens; it is not an
execution simulator or evidence that DRL is profitable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from hkipo.subscription_net_returns import ipo_side_costs, sale_side_costs


STATIC_COLUMNS = [
    "days_since_prospectus", "prior_verified_ipo_count",
    "prior_verified_mean_net_return", "prior_verified_positive_fraction",
    "prior_sponsor_ipo_count", "prior_sponsor_mean_net_pnl_hkd",
    "prior_sponsor_positive_fraction",
]
ACCOUNT_COLUMNS = [
    "available_cash_fraction", "frozen_cash_fraction", "nav_return",
    "held_value_fraction", "held_position_count_scaled",
    "pending_count_scaled", "pending_lock_0_2d", "pending_lock_3_5d",
    "pending_lock_6_10d", "pending_lock_11plus_d",
    "nearest_pending_result_days_scaled", "nearest_holding_listing_days_scaled",
    "decision_days_to_listing_scaled",
]


def _distribution(group: pd.DataFrame) -> list[tuple[float, int]]:
    counts = pd.to_numeric(group.applicants_in_outcome_group, errors="raise").astype(int)
    shares = pd.to_numeric(group.allocated_shares_per_applicant, errors="raise").astype(int)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("One-lot allocation group has no applicants")
    return [(int(n) / total, int(q)) for n, q in zip(counts, shares)]


@dataclass(frozen=True)
class EventEconomics:
    lock_amount: float
    expected_shares: float
    expected_purchase: float
    expected_ipo_fee: float
    expected_sale_proceeds: float
    result_date: date
    listing_date: date
    issue_price: float


class CashAllocationEnv:
    """One decision per IPO, with cash, pending applications and holdings."""

    def __init__(self, events: pd.DataFrame, features: pd.DataFrame,
                 outcomes: pd.DataFrame, cash_terms: pd.DataFrame,
                 *, initial_cash: float, static_mean: np.ndarray,
                 static_scale: np.ndarray, sale_commission: float = 0.001,
                 annual_opportunity_rate: float = 0.0,
                 fixed_application_fee: float = 0.0):
        self.initial_cash = float(initial_cash)
        self.sale_commission = Decimal(str(sale_commission))
        self.annual_opportunity_rate = float(annual_opportunity_rate)
        self.fixed_application_fee = float(fixed_application_fee)
        self.static_mean = static_mean.astype(np.float32)
        self.static_scale = static_scale.astype(np.float32)

        events = events.copy()
        features = features[["listing_event_id"] + STATIC_COLUMNS].copy()
        for frame in (events, features, outcomes, cash_terms):
            frame["listing_event_id"] = frame.listing_event_id.astype(str)
        events["days_to_listing"] = (
            pd.to_datetime(events.listing_date) - pd.to_datetime(events.decision_date)
        ).dt.days
        events = events.merge(features, on="listing_event_id", how="left", validate="one_to_one")
        terms = cash_terms[["listing_event_id", "exact_max_application_price_hkd",
                            "term_price_valid"]].copy()
        events = events.merge(terms, on="listing_event_id", how="left", validate="one_to_one")
        events["term_price_valid"] = events.term_price_valid.fillna(False).astype(bool)
        events["exact_max_application_price_hkd"] = pd.to_numeric(
            events.exact_max_application_price_hkd, errors="coerce")
        if not events.term_price_valid.all():
            raise ValueError("DQN environment requires verified maximum application prices")
        for col in ("decision_date", "result_date", "listing_date"):
            events[col] = pd.to_datetime(events[col]).dt.date
        events = events.sort_values(["decision_date", "listing_date", "stock_code"]).reset_index(drop=True)
        if events.listing_event_id.duplicated().any():
            raise ValueError("Duplicate event ids in environment")

        self.events = events
        self.row_by_id = {str(r.listing_event_id): r for r in events.itertuples(index=False)}
        self.economics = self._build_economics(outcomes)
        self.reset()

    def _build_economics(self, outcomes: pd.DataFrame) -> dict[str, EventEconomics]:
        economics: dict[str, EventEconomics] = {}
        for row in self.events.itertuples(index=False):
            event_id = str(row.listing_event_id)
            lot = int(row.minimum_application_shares)
            group = outcomes.loc[
                outcomes.listing_event_id.eq(event_id)
                & pd.to_numeric(outcomes.applied_shares, errors="coerce").eq(lot)
            ]
            if group.empty:
                raise ValueError(f"No verified one-lot outcome distribution for {event_id}")
            dist = _distribution(group)
            total = sum(p for p, _ in dist)
            if not np.isclose(total, 1.0):
                raise ValueError(f"Outcome probabilities do not sum to one for {event_id}")
            issue = Decimal(str(row.ipo_price_hkd))
            maximum = Decimal(str(row.exact_max_application_price_hkd))
            expected_shares = sum(p * shares for p, shares in dist)
            purchase = 0.0
            ipo_fee = 0.0
            proceeds = 0.0
            sale_price = Decimal(str(row.first_day_open_hkd))
            for probability, shares in dist:
                if shares <= 0:
                    continue
                purchase += probability * shares * float(issue)
                ipo_fee += probability * float(ipo_side_costs(shares, issue))
                proceeds += probability * float(
                    shares * sale_price - sale_side_costs(shares, sale_price, self.sale_commission))
            lock_amount = float(maximum * lot + ipo_side_costs(lot, maximum))
            economics[event_id] = EventEconomics(
                lock_amount=lock_amount,
                expected_shares=expected_shares,
                expected_purchase=purchase,
                expected_ipo_fee=ipo_fee,
                expected_sale_proceeds=proceeds,
                result_date=row.result_date,
                listing_date=row.listing_date,
                issue_price=float(issue),
            )
        return economics

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.index = 0
        self.cash = self.initial_cash
        self.frozen = 0.0
        self.holdings: dict[str, tuple[float, float]] = {}
        self.pending: dict[str, dict] = {}
        self.applied_ids: set[str] = set()
        self.actions: list[dict] = []
        self.nav_path: list[float] = [self.initial_cash]
        if len(self.events):
            self._process_until(self.events.iloc[0].decision_date)
            return self._observation(), self._action_mask()
        return np.zeros(len(STATIC_COLUMNS) + len(ACCOUNT_COLUMNS), dtype=np.float32), np.array([True, False])

    def _nav(self) -> float:
        return self.cash + self.frozen + sum(qty * mark for qty, mark in self.holdings.values())

    def _process_until(self, target: date) -> None:
        due_dates = {item["result_date"] for item in self.pending.values()
                     if item["result_date"] <= target}
        due_dates |= {self.row_by_id[event_id].listing_date for event_id in self.holdings
                      if self.row_by_id[event_id].listing_date <= target}
        due_dates |= {self.economics[event_id].listing_date for event_id, item in self.pending.items()
                      if item["result_date"] <= target
                      and self.economics[event_id].listing_date <= target}
        for current in sorted(due_dates):
            # Match the deterministic simulator's order: allotment/refund, then listing sale.
            due_results = sorted(event_id for event_id, item in self.pending.items()
                                 if item["result_date"] == current)
            for event_id in due_results:
                item = self.pending.pop(event_id)
                econ = self.economics[event_id]
                lock = item["locked_cash"]
                refund = max(lock - econ.expected_purchase - econ.expected_ipo_fee, 0.0)
                shortfall = max(econ.expected_purchase + econ.expected_ipo_fee - lock, 0.0)
                days = max((current - item["decision_date"]).days, 0)
                opportunity = lock * self.annual_opportunity_rate * days / 365.0
                self.frozen = max(0.0, self.frozen - lock)
                self.cash += refund - shortfall - opportunity
                self.holdings[event_id] = (econ.expected_shares, econ.issue_price)
                self.nav_path.append(self._nav())

            due_listings = sorted(event_id for event_id in self.holdings
                                  if self.row_by_id[event_id].listing_date == current)
            for event_id in due_listings:
                econ = self.economics[event_id]
                self.cash += econ.expected_sale_proceeds
                self.holdings.pop(event_id)
                self.nav_path.append(self._nav())

    def _observation(self) -> np.ndarray:
        row = self.events.iloc[self.index]
        raw = np.array([pd.to_numeric(row.get(c), errors="coerce") for c in STATIC_COLUMNS],
                       dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        static = np.clip((raw - self.static_mean) / self.static_scale, -8.0, 8.0)
        current_date = row.decision_date
        pending_items = list(self.pending.items())
        pending_locks = [item["locked_cash"] for _, item in pending_items]
        pending_days = [(item["result_date"] - current_date).days for _, item in pending_items]
        bins = [0.0, 0.0, 0.0, 0.0]
        for locked, days in zip(pending_locks, pending_days):
            bucket = 0 if days <= 2 else 1 if days <= 5 else 2 if days <= 10 else 3
            bins[bucket] += locked / self.initial_cash
        holding_days = [(self.row_by_id[event_id].listing_date - current_date).days
                        for event_id in self.holdings]
        held_value = sum(qty * mark for qty, mark in self.holdings.values())
        account = np.array([
            self.cash / self.initial_cash,
            self.frozen / self.initial_cash,
            self._nav() / self.initial_cash - 1.0,
            held_value / self.initial_cash,
            min(len(self.holdings), 10) / 10.0,
            min(len(self.pending), 10) / 10.0,
            *bins,
            min(max(min(pending_days, default=0), 0), 30) / 30.0,
            min(max(min(holding_days, default=0), 0), 30) / 30.0,
            min(max(int(row.days_to_listing), 0), 90) / 90.0,
        ], dtype=np.float32)
        return np.concatenate([static.astype(np.float32), account]).astype(np.float32)

    def _action_mask(self) -> np.ndarray:
        row = self.events.iloc[self.index]
        event_id = str(row.listing_event_id)
        lock = self.economics[event_id].lock_amount + self.fixed_application_fee
        return np.array([True, self.cash + 1e-9 >= lock], dtype=bool)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, np.ndarray, dict]:
        if action not in (0, 1):
            raise ValueError("Action must be 0 (skip) or 1 (apply one lot)")
        mask = self._action_mask()
        if action == 1 and not mask[1]:
            raise ValueError("DQN selected apply when cash constraint makes it unavailable")
        before = self._nav()
        row = self.events.iloc[self.index]
        event_id = str(row.listing_event_id)
        executed = int(action == 1 and mask[1])
        econ = self.economics[event_id]
        if executed:
            self.cash -= econ.lock_amount + self.fixed_application_fee
            self.frozen += econ.lock_amount
            self.pending[event_id] = {
                "locked_cash": econ.lock_amount,
                "decision_date": row.decision_date,
                "result_date": econ.result_date,
            }
            self.applied_ids.add(event_id)
        self.nav_path.append(self._nav())
        self.actions.append({"listing_event_id": event_id, "action": int(action),
                             "action_executed": executed,
                             "cash_before_hkd": self.cash + (econ.lock_amount + self.fixed_application_fee
                                                              if executed else 0.0)})

        self.index += 1
        done = self.index >= len(self.events)
        if done:
            final_date = max(self.events.listing_date)
            self._process_until(final_date)
            next_state = np.zeros(len(STATIC_COLUMNS) + len(ACCOUNT_COLUMNS), dtype=np.float32)
            next_mask = np.array([True, False], dtype=bool)
        else:
            next_decision = self.events.iloc[self.index].decision_date
            self._process_until(next_decision)
            next_state = self._observation()
            next_mask = self._action_mask()
        reward = (self._nav() - before) / self.initial_cash
        info = {"listing_event_id": event_id, "action_executed": executed,
                "nav_hkd": self._nav(), "cash_hkd": self.cash,
                "frozen_cash_hkd": self.frozen}
        return next_state, float(reward), done, next_mask, info


class QNetwork:
    def __init__(self, input_dim: int):
        import torch
        from torch import nn

        self.torch = torch
        self.module = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )


class DQNAgent:
    def __init__(self, input_dim: int, *, seed: int, learning_rate: float = 5e-4,
                 gamma: float = 1.0, batch_size: int = 64):
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.torch = torch
        self.online = QNetwork(input_dim).module
        self.target = QNetwork(input_dim).module
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory: deque = deque(maxlen=50_000)
        self.updates = 0

    def action(self, state: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        valid = np.flatnonzero(mask)
        if random.random() < epsilon:
            return int(random.choice(valid.tolist()))
        with self.torch.no_grad():
            q = self.online(self.torch.tensor(state, dtype=self.torch.float32).unsqueeze(0))[0]
            q[~self.torch.tensor(mask, dtype=self.torch.bool)] = -1e9
            return int(q.argmax().item())

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool, next_mask: np.ndarray) -> None:
        self.memory.append((state.copy(), int(action), float(reward), next_state.copy(),
                            bool(done), next_mask.copy()))

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = self.torch.tensor(np.stack([x[0] for x in batch]), dtype=self.torch.float32)
        actions = self.torch.tensor([x[1] for x in batch], dtype=self.torch.int64).unsqueeze(1)
        rewards = self.torch.tensor([x[2] for x in batch], dtype=self.torch.float32)
        next_states = self.torch.tensor(np.stack([x[3] for x in batch]), dtype=self.torch.float32)
        dones = self.torch.tensor([x[4] for x in batch], dtype=self.torch.float32)
        next_masks = self.torch.tensor(np.stack([x[5] for x in batch]), dtype=self.torch.bool)
        q = self.online(states).gather(1, actions).squeeze(1)
        with self.torch.no_grad():
            future = self.target(next_states).masked_fill(~next_masks, -1e9).max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * future
        loss = self.torch.nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())


def _simulate_policy(env: CashAllocationEnv, policy, *, epsilon: float = 0.0,
                     seed: int = 0) -> dict:
    random.seed(seed)
    state, mask = env.reset()
    total_reward = 0.0
    while len(env.events) and env.index < len(env.events):
        action = int(policy(state, mask, epsilon))
        state, reward, done, mask, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return {"ending_nav_hkd": env._nav(),
            "net_profit_hkd": env._nav() - env.initial_cash,
            "net_return": env._nav() / env.initial_cash - 1.0,
            "max_drawdown_pct": _max_drawdown(env.nav_path),
            "applications": int(sum(x["action_executed"] for x in env.actions)),
            "decisions": int(len(env.actions)),
            "skipped_for_insufficient_cash": int(sum(
                x["action"] == 1 and x["action_executed"] == 0 for x in env.actions)),
            "total_reward": total_reward,
            "actions": env.actions.copy()}


def _max_drawdown(nav_path: list[float]) -> float:
    values = np.asarray(nav_path, dtype=np.float64)
    if not len(values):
        return 0.0
    peaks = np.maximum.accumulate(values)
    return float(np.min(values / peaks - 1.0))


def _fit_static_scaler(features: pd.DataFrame, train_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    train = features.loc[features.listing_event_id.isin(train_ids), STATIC_COLUMNS].copy()
    train = train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mean = train.mean(axis=0).to_numpy(dtype=np.float32)
    scale = train.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean, scale


def _make_env(events: pd.DataFrame, features: pd.DataFrame, outcomes: pd.DataFrame,
              terms: pd.DataFrame, *, initial_cash: float,
              static_mean: np.ndarray, static_scale: np.ndarray,
              sale_commission: float = 0.001,
              annual_opportunity_rate: float = 0.0) -> CashAllocationEnv:
    return CashAllocationEnv(events, features, outcomes, terms,
                             initial_cash=initial_cash, static_mean=static_mean,
                             static_scale=static_scale, sale_commission=sale_commission,
                             annual_opportunity_rate=annual_opportunity_rate)


def train_and_evaluate(*, events_path: Path, features_path: Path,
                       outcomes_path: Path, cash_terms_path: Path,
                       output_dir: Path, initial_cash: float = 100_000.0,
                       episodes: int = 250, seeds: tuple[int, ...] = (7, 21),
                       validation_year: int = 2024, test_year: int = 2025,
                       selection_mode: str = "validation_best") -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional DQN dependencies with pip install -e '.[drl]'") from exc

    if selection_mode not in {"validation_best", "fixed_final_ensemble"}:
        raise ValueError("selection_mode must be validation_best or fixed_final_ensemble")
    if episodes < 1 or not seeds:
        raise ValueError("episodes and seeds must be non-empty and positive")

    events = pd.read_csv(events_path, dtype={"listing_event_id": str, "stock_code": str})
    events["listing_year"] = pd.to_datetime(events.listing_date).dt.year
    features = pd.read_csv(features_path, dtype={"listing_event_id": str})
    outcomes = pd.read_csv(outcomes_path, dtype={"listing_event_id": str})
    terms = pd.read_csv(cash_terms_path, dtype={"listing_event_id": str})
    train_events = events.loc[events.listing_year < validation_year].copy()
    validation_events = events.loc[events.listing_year == validation_year].copy()
    test_events = events.loc[events.listing_year == test_year].copy()
    if min(len(train_events), len(validation_events), len(test_events)) == 0:
        raise ValueError("Train, validation and test splits must all contain IPO events")
    train_ids = set(train_events.listing_event_id.astype(str))
    static_mean, static_scale = _fit_static_scaler(features, train_ids)
    train_env = _make_env(train_events, features, outcomes, terms,
                          initial_cash=initial_cash, static_mean=static_mean,
                          static_scale=static_scale)
    validation_env = _make_env(validation_events, features, outcomes, terms,
                               initial_cash=initial_cash, static_mean=static_mean,
                               static_scale=static_scale)
    test_env = _make_env(test_events, features, outcomes, terms,
                         initial_cash=initial_cash, static_mean=static_mean,
                         static_scale=static_scale)

    def eval_rule(env: CashAllocationEnv, field: str) -> dict:
        state, mask = env.reset()
        frame = features.set_index("listing_event_id")
        while env.index < len(env.events):
            event_id = str(env.events.iloc[env.index].listing_event_id)
            value = pd.to_numeric(frame.loc[event_id, field], errors="coerce")
            action = int(bool(mask[1]) and pd.notna(value) and value > 0)
            state, _, done, mask, _ = env.step(action)
            if done:
                break
        return _episode_summary(env)

    def evaluate_basic(env: CashAllocationEnv, *, all_apply: bool) -> dict:
        policy = (lambda state, mask, eps: int(mask[1])) if all_apply else (lambda state, mask, eps: 0)
        result = _simulate_policy(env, policy)
        return {k: v for k, v in result.items() if k != "actions"}

    def _run_rule_reset(env: CashAllocationEnv, field: str) -> dict:
        return eval_rule(env, field)

    def _episode_summary(env: CashAllocationEnv) -> dict:
        nav = env._nav()
        return {"ending_nav_hkd": nav, "net_profit_hkd": nav - env.initial_cash,
                "net_return": nav / env.initial_cash - 1.0,
                "max_drawdown_pct": _max_drawdown(env.nav_path),
                "applications": int(sum(x["action_executed"] for x in env.actions)),
                "decisions": int(len(env.actions)),
                "skipped_for_insufficient_cash": int(sum(
                    x["action"] == 1 and x["action_executed"] == 0 for x in env.actions))}

    seed_results = []
    best_model_state = None
    best_validation_profit = -float("inf")
    best_seed = None
    best_history: list[dict] = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        agent = DQNAgent(len(STATIC_COLUMNS) + len(ACCOUNT_COLUMNS), seed=seed)
        history = []
        seed_best_profit = -float("inf")
        seed_best_state = None
        last_losses: list[float] = []
        for episode in range(episodes):
            epsilon = max(0.05, 1.0 - 0.95 * episode / max(episodes - 1, 1))
            state, mask = train_env.reset()
            while train_env.index < len(train_env.events):
                action = agent.action(state, mask, epsilon)
                next_state, reward, done, next_mask, _ = train_env.step(action)
                agent.remember(state, action, reward, next_state, done, next_mask)
                loss = agent.learn()
                if loss is not None:
                    last_losses.append(loss)
                state, mask = next_state, next_mask
                if done:
                    break
            if (episode + 1) % 10 == 0 or episode + 1 == episodes:
                record = {"seed": seed, "episode": episode + 1,
                          "epsilon": epsilon,
                          "train_last_episode_return": train_env._nav() / initial_cash - 1.0,
                          "mean_recent_loss": float(np.mean(last_losses[-100:]))
                              if last_losses else None}
                if selection_mode == "validation_best":
                    result = _simulate_policy(
                        validation_env, lambda s, m, e: agent.action(s, m, 0.0), seed=seed)
                    record["validation_net_profit_hkd"] = result["net_profit_hkd"]
                    record["validation_applications"] = result["applications"]
                    if result["net_profit_hkd"] > seed_best_profit:
                        seed_best_profit = result["net_profit_hkd"]
                        seed_best_state = {k: v.detach().cpu().clone()
                                           for k, v in agent.online.state_dict().items()}
                history.append(record)
        if selection_mode == "fixed_final_ensemble":
            seed_best_state = {k: v.detach().cpu().clone()
                               for k, v in agent.online.state_dict().items()}
        if seed_best_state is None:
            seed_best_state = {k: v.detach().cpu().clone()
                               for k, v in agent.online.state_dict().items()}
        model = QNetwork(len(STATIC_COLUMNS) + len(ACCOUNT_COLUMNS)).module
        model.load_state_dict(seed_best_state)
        selected_agent = DQNAgent(len(STATIC_COLUMNS) + len(ACCOUNT_COLUMNS), seed=seed)
        selected_agent.online.load_state_dict(seed_best_state)
        validation_result = _simulate_policy(
            validation_env, lambda s, m, e: selected_agent.action(s, m, 0.0), seed=seed)
        test_result = _simulate_policy(
            test_env, lambda s, m, e: selected_agent.action(s, m, 0.0), seed=seed)
        seed_results.append({"seed": seed,
                             "selected_validation_net_profit_hkd": validation_result["net_profit_hkd"],
                             "selected_validation_max_drawdown_pct": validation_result["max_drawdown_pct"],
                             "test_net_profit_hkd": test_result["net_profit_hkd"],
                             "test_net_return": test_result["net_return"],
                             "test_max_drawdown_pct": test_result["max_drawdown_pct"],
                             "test_applications": test_result["applications"],
                             "validation_selected_episode": (
                                 max((x for x in history if "validation_net_profit_hkd" in x),
                                     key=lambda x: x["validation_net_profit_hkd"])["episode"]
                                 if selection_mode == "validation_best" else None),
                             "test_actions": test_result["actions"],
                             "training_history": history,
                             "model_state": seed_best_state})
        if validation_result["net_profit_hkd"] > best_validation_profit:
            best_validation_profit = validation_result["net_profit_hkd"]
            best_model_state = seed_best_state
            best_seed = seed
            best_history = history

    if selection_mode == "validation_best":
        if best_model_state is None:
            raise RuntimeError("DQN training did not produce a model")
        best_agent = DQNAgent(len(STATIC_COLUMNS) + len(ACCOUNT_COLUMNS), seed=int(best_seed or 0))
        best_agent.online.load_state_dict(best_model_state)
        best_agent.target.load_state_dict(best_model_state)
        dqn_validation = _simulate_policy(
            validation_env, lambda s, m, e: best_agent.action(s, m, 0.0), seed=0)
        dqn_test = _simulate_policy(test_env, lambda s, m, e: best_agent.action(s, m, 0.0), seed=0)
        primary_dqn_name = "dqn_validation_selected"
        primary_model_states = [best_model_state]
    else:
        ensemble_agents = []
        primary_model_states = [item["model_state"] for item in seed_results]
        for item in seed_results:
            ensemble_agent = DQNAgent(len(STATIC_COLUMNS) + len(ACCOUNT_COLUMNS),
                                      seed=int(item["seed"]))
            ensemble_agent.online.load_state_dict(item["model_state"])
            ensemble_agents.append(ensemble_agent)

        def majority_vote(state, mask, epsilon):
            apply_votes = sum(agent.action(state, mask, 0.0) for agent in ensemble_agents)
            # Ties resolve to skip; this rule is fixed before looking at any test fold.
            return int(apply_votes > len(ensemble_agents) / 2)

        dqn_validation = _simulate_policy(validation_env, majority_vote, seed=0)
        dqn_test = _simulate_policy(test_env, majority_vote, seed=0)
        primary_dqn_name = "dqn_fixed_final_majority_ensemble"

    # Fixed, predeclared rules use only decision-time historical features.
    validation_baselines = {
        "always_apply": evaluate_basic(validation_env, all_apply=True),
        "skip_all": evaluate_basic(validation_env, all_apply=False),
        "prior_sponsor_positive": _run_rule_reset(validation_env, "prior_sponsor_mean_net_pnl_hkd"),
        "prior_market_positive": _run_rule_reset(validation_env, "prior_verified_mean_net_return"),
    }
    test_baselines = {
        "always_apply": evaluate_basic(test_env, all_apply=True),
        "skip_all": evaluate_basic(test_env, all_apply=False),
        "prior_sponsor_positive": _run_rule_reset(test_env, "prior_sponsor_mean_net_pnl_hkd"),
        "prior_market_positive": _run_rule_reset(test_env, "prior_verified_mean_net_return"),
    }
    validation_baselines[primary_dqn_name] = {k: dqn_validation[k] for k in (
        "ending_nav_hkd", "net_profit_hkd", "net_return", "max_drawdown_pct",
        "applications", "decisions")}
    test_baselines[primary_dqn_name] = {k: dqn_test[k] for k in (
        "ending_nav_hkd", "net_profit_hkd", "net_return", "max_drawdown_pct",
        "applications", "decisions")}

    output_dir.mkdir(parents=True, exist_ok=True)
    if best_history:
        pd.DataFrame(best_history).to_csv(output_dir / "selected_training_history.csv", index=False)
    pd.DataFrame(dqn_test["actions"]).to_csv(output_dir / "dqn_primary_test_actions.csv", index=False)
    (output_dir / "static_scaler.json").write_text(json.dumps({
        "feature_names": STATIC_COLUMNS,
        "mean": static_mean.tolist(),
        "scale": static_scale.tolist(),
        "fit_partition": f"listing_year < {validation_year}",
    }, indent=2), encoding="utf-8")
    for seed_result in seed_results:
        pd.DataFrame(seed_result["training_history"]).to_csv(
            output_dir / f"training_history_seed_{seed_result['seed']}.csv", index=False)
        pd.DataFrame(seed_result["test_actions"]).to_csv(
            output_dir / f"test_actions_seed_{seed_result['seed']}.csv", index=False)
        if selection_mode == "fixed_final_ensemble":
            torch.save(seed_result["model_state"],
                       output_dir / f"dqn_final_model_seed_{seed_result['seed']}.pt")
    if best_model_state is not None:
        torch.save(best_model_state, output_dir / "dqn_selected_model.pt")
    summary = {
        "algorithm": "DQN with 64x32 MLP, replay buffer and target network",
        "decision": "skip or cash-apply one lot",
        "initial_cash_hkd": initial_cash,
        "training_episodes": episodes,
        "selection_mode": selection_mode,
        "primary_dqn_policy": primary_dqn_name,
        "random_seeds": list(seeds),
        "dqn_hyperparameters": {"learning_rate": 0.0005, "gamma": 1.0,
                                 "batch_size": 64, "replay_capacity": 50000,
                                 "target_update_steps": 100},
        "split_by_listing_year": {
            "train": [int(train_events.listing_year.min()), int(train_events.listing_year.max())],
            "validation": validation_year,
            "test": test_year,
            "train_events": int(len(train_events)),
            "validation_events": int(len(validation_events)),
            "test_events": int(len(test_events)),
        },
        "feature_names": STATIC_COLUMNS + ACCOUNT_COLUMNS,
        "static_scaler_fit_on_train_only": True,
        "selected_seed_by_validation": best_seed if selection_mode == "validation_best" else None,
        "selected_validation_profit_hkd": (
            best_validation_profit if selection_mode == "validation_best" else None),
        "validation_baselines": validation_baselines,
        "test_baselines": test_baselines,
        "test_seed_variation": [{k: v for k, v in item.items()
                                 if k not in {"training_history", "test_actions", "model_state"}}
                                for item in seed_results],
        "limitations": [
            (f"This fold has {len(train_events)} training IPOs, "
             f"{len(validation_events)} validation IPOs, and {len(test_events)} test IPOs."),
            "Repeated training episodes reuse the same historical IPO outcomes and do not create independent market samples.",
            "Allotment is the deterministic expected allocation distribution, so fractional expected shares are allowed.",
            "The account observation summarizes pending cash by time bins; unsettled allocation outcomes remain partially observed.",
            "Fees use exchange/statutory IPO charges and a 0.10% sale commission assumption; broker-specific fees are incomplete.",
            "Listing-day Yahoo open is a price proxy, not guaranteed execution; slippage, queue position, and halts are omitted.",
            "Each test partition is one calendar year and cannot establish broad out-of-sample robustness.",
        ],
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path(
        "reports/hkipo_simulator_data_2021_2025/verified_events_timetable_augmented.csv"))
    parser.add_argument("--features", type=Path, default=Path(
        "reports/hkipo_point_in_time_features_2021_2025_timetable_final/candidate_features.csv"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "reports/hkipo_simulator_data_2021_2025/one_lot_outcomes.csv"))
    parser.add_argument("--cash-terms", type=Path, default=Path(
        "data/processed/hkex_ipo_cash_terms_2021_2025.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "reports/hkipo_dqn_cash_allocator_2021_2025"))
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21])
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--selection-mode",
                        choices=["validation_best", "fixed_final_ensemble"],
                        default="validation_best")
    args = parser.parse_args()
    train_and_evaluate(
        events_path=args.events, features_path=args.features,
        outcomes_path=args.outcomes, cash_terms_path=args.cash_terms,
        output_dir=args.output_dir, initial_cash=args.initial_cash,
        episodes=args.episodes, seeds=tuple(args.seeds),
        validation_year=args.validation_year, test_year=args.test_year,
        selection_mode=args.selection_mode)


if __name__ == "__main__":
    main()
