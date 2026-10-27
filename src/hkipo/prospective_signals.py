"""Append point-in-time research signals for IPOs after a frozen-model cutoff.

This module logs candidate decisions only. It does not submit orders or imply
that the next open is an executable fill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN as SB3DQN
from torch import nn

from .dqn_aftermarket import FEATURES, TARGET_WEIGHTS, add_event_features, observation

torch.set_num_threads(1)

LOG_COLUMNS = [
    "model_freeze_id", "candidate", "listing_event_id", "stock_code", "company_name",
    "listing_date", "ipo_price", "decision_date", "listed_day", "observation_cutoff",
    "price_source", "state_premium", "state_ret1", "state_ret5", "state_intraday_range",
    "state_volume_change", "state_rv5", "state_age", "state_position_weight",
    "action_index", "target_weight", "current_nav", "next_session_date",
    "open_reference_price", "fee_bps_per_side", "data_snapshot_sha256", "signal_recorded_at",
    "execution_status", "notes",
]


class QNet(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        layers, width = [], len(FEATURES)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(0.10)])
            width = hidden
        layers.append(nn.Linear(width, len(TARGET_WEIGHTS)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_candidate_members(spec):
    return [(QNet(spec["hidden_dims"]).eval(), torch.load(item["path"], map_location="cpu", weights_only=True))
            for item in spec["models"]]


def member_action(member, state):
    if isinstance(member, SB3DQN):
        action, _ = member.predict(state, deterministic=True)
        return int(action)
    net, weights = member
    net.load_state_dict(weights)
    with torch.no_grad():
        return int(net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).argmax(1).item())


def majority_action(members, state):
    votes = [member_action(member, state) for member in members]
    counts = np.bincount(votes, minlength=len(TARGET_WEIGHTS))
    # np.argmax resolves ties toward the smallest exposure action.
    return int(counts.argmax())


def snapshot_hash(prefix: pd.DataFrame) -> str:
    cols = [c for c in ("listing_event_id", "stock_code", "listing_date", "date", "ipo_price",
                        "open", "high", "low", "close", "volume", "source") if c in prefix]
    blob = prefix[cols].to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def event_signals(group, candidate, members, freeze_id, fee_bps, logged_at):
    group = add_event_features(group.sort_values("date")).head(30).reset_index(drop=True)
    cash, shares, pending_target = 1.0, 0.0, None
    output = []
    # On a complete 30-session event, do not issue a signal beyond the defined
    # 30-session horizon. For a still-forming event, log the latest close signal.
    decision_count = min(len(group), 29) if len(group) >= 30 else len(group)
    for i in range(decision_count):
        row = group.iloc[i]
        if i > 0 and pending_target is not None:
            open_px = float(row.open)
            nav_open = cash + shares * open_px
            delta = pending_target * nav_open - shares * open_px
            cash -= delta + fee_bps / 10000 * abs(delta)
            shares += delta / open_px
        close = float(row.close)
        nav_close = max(cash + shares * close, 1e-12)
        position_weight = shares * close / nav_close if nav_close > 0 else 0.0
        state = observation(row, position_weight)
        action = majority_action(members, state)
        output.append({
            "model_freeze_id": freeze_id,
            "candidate": candidate,
            "listing_event_id": str(row.listing_event_id),
            "stock_code": str(row.stock_code),
            "company_name": str(row.company_name) if pd.notna(row.get("company_name")) else "",
            "listing_date": row.listing_date.date().isoformat(),
            "ipo_price": float(row.ipo_price),
            "decision_date": row.date.date().isoformat(),
            "listed_day": int(row.listed_day),
            "observation_cutoff": row.date.date().isoformat(),
            "price_source": str(row.get("source", "")),
            "state_premium": float(state[0]), "state_ret1": float(state[1]),
            "state_ret5": float(state[2]), "state_intraday_range": float(state[3]),
            "state_volume_change": float(state[4]), "state_rv5": float(state[5]),
            "state_age": float(state[6]), "state_position_weight": float(state[7]),
            "action_index": action, "target_weight": float(TARGET_WEIGHTS[action]),
            "current_nav": float(nav_close), "next_session_date": "", "open_reference_price": "",
            "fee_bps_per_side": fee_bps,
            "data_snapshot_sha256": snapshot_hash(group.iloc[:i + 1]),
            "signal_recorded_at": logged_at, "execution_status": "research_signal_unexecuted",
            "notes": "signal after close; no order submitted; next open is not assumed filled",
        })
        pending_target = float(TARGET_WEIGHTS[action])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True,
                        help="HK IPO event panel, containing rows only through the declared as-of date")
    parser.add_argument("--manifest", type=Path,
                        default=Path("reports/hkipo_dqn_candidate_seed_stability/frozen_policy_manifest.json"))
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD point-in-time cutoff")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/hkipo_dqn_candidate_seed_stability/prospective_signals.csv"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    as_of = pd.Timestamp(args.as_of).normalize()
    freeze_date = pd.Timestamp(manifest["prospective_listing_date_minimum"]).normalize()
    df = pd.read_csv(args.panel, parse_dates=["date", "listing_date"])
    if df.date.max() > as_of:
        raise ValueError(f"Panel contains data after --as-of ({df.date.max().date()} > {as_of.date()})")
    df = df.loc[df.listing_date.ge(freeze_date) & df.date.le(as_of)].copy()
    logged_at = datetime.now(timezone.utc).isoformat()
    fee_bps = float(manifest["fee_bps_per_side"])
    signals = []
    for candidate, spec in manifest["models"].items():
        if candidate == "finrl_sb3_standard_dqn_baseline":
            members = [SB3DQN.load(member["path"], device="cpu") for member in spec["members"]]
        else:
            members = load_candidate_members(spec)
        for _, group in df.groupby("listing_event_id", sort=False):
            signals.extend(event_signals(group, candidate, members, manifest["freeze_date"], fee_bps, logged_at))

    fresh = pd.DataFrame(signals, columns=LOG_COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        prior = pd.read_csv(args.output, dtype={"stock_code": str})
        fresh = pd.concat([prior, fresh], ignore_index=True)
    if not fresh.empty:
        fresh = fresh.drop_duplicates(["model_freeze_id", "candidate", "listing_event_id", "decision_date"],
                                      keep="first").sort_values(["listing_date", "listing_event_id", "decision_date", "candidate"])
    fresh.to_csv(args.output, index=False)
    print(json.dumps({"as_of": as_of.date().isoformat(), "eligible_events": int(df.listing_event_id.nunique()),
                      "new_signal_rows": len(signals), "stored_signal_rows": len(fresh),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
