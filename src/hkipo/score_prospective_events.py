"""Score frozen research signals after a new IPO completes 30 sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def replay(ep: pd.DataFrame, targets: list[float], fee: float) -> dict:
    cash, shares, peak, max_dd, turnover, prev_nav = 1.0, 0.0, 1.0, 0.0, 0.0, 1.0
    for i, target in enumerate(targets):
        nxt = ep.iloc[i + 1]
        open_px, close_px = float(nxt.open), float(nxt.close)
        nav_open = cash + shares * open_px
        delta = target * nav_open - shares * open_px
        trade_value = abs(delta)
        cash -= delta + fee * trade_value
        shares += delta / open_px
        nav = max(cash + shares * close_px, 1e-12)
        peak = max(peak, nav)
        max_dd = max(max_dd, (peak - nav) / peak)
        turnover += trade_value / max(prev_nav, 1e-12)
        prev_nav = nav
    return {"final_return": prev_nav - 1, "max_drawdown": max_dd, "one_way_turnover": turnover}


def snapshot_hash(group):
    cols = [c for c in ("listing_event_id", "stock_code", "listing_date", "date", "ipo_price",
                        "open", "high", "low", "close", "volume", "source") if c in group]
    return hashlib.sha256(group[cols].to_csv(index=False, date_format="%Y-%m-%d").encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--signals", type=Path,
                        default=Path("reports/hkipo_dqn_candidate_seed_stability/prospective_signals.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("reports/hkipo_dqn_candidate_seed_stability/frozen_policy_manifest.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/hkipo_dqn_candidate_seed_stability/prospective_outcomes.csv"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    panel = pd.read_csv(args.panel, parse_dates=["date", "listing_date"])
    signals = pd.read_csv(args.signals, parse_dates=["decision_date", "listing_date"])
    cutoff = pd.Timestamp(manifest["prospective_listing_date_minimum"])
    panel = panel.loc[panel.listing_date.ge(cutoff)]
    rows = []
    for event_id, ep in panel.sort_values("date").groupby("listing_event_id", sort=False):
        ep = ep.sort_values("date").head(30).reset_index(drop=True)
        if len(ep) < 30:
            continue
        event_signals = signals.loc[signals.listing_event_id.eq(event_id)]
        if event_signals.empty:
            continue
        for candidate, sig in event_signals.groupby("candidate"):
            sig = sig.sort_values("decision_date")
            sig = sig.loc[sig.decision_date.isin(ep.date.iloc[:-1])]
            if len(sig) != 29 or sig.decision_date.duplicated().any():
                continue
            sig = sig.set_index("decision_date").reindex(ep.date.iloc[:-1]).reset_index()
            targets = sig.target_weight.astype(float).tolist()
            metrics = replay(ep, targets, float(sig.fee_bps_per_side.iloc[0]) / 10000)
            rows.append({"model_freeze_id": manifest["freeze_date"], "candidate": candidate,
                         "listing_event_id": str(event_id), "stock_code": str(ep.stock_code.iloc[0]),
                         "listing_date": ep.listing_date.iloc[0].date().isoformat(),
                         "maturity_date": ep.date.iloc[-1].date().isoformat(), "sessions": len(ep),
                         "fee_bps_per_side": float(sig.fee_bps_per_side.iloc[0]),
                         "signal_count": len(sig), "signal_data_snapshot_sha256": str(sig.data_snapshot_sha256.iloc[-1]),
                         "outcome_panel_sha256": snapshot_hash(ep), "policy": "frozen_candidate", **metrics})
        bh_metrics = replay(ep, [1.0] * 29, float(manifest["fee_bps_per_side"]) / 10000)
        rows.append({"model_freeze_id": manifest["freeze_date"], "candidate": "buy_and_hold",
                     "listing_event_id": str(event_id), "stock_code": str(ep.stock_code.iloc[0]),
                     "listing_date": ep.listing_date.iloc[0].date().isoformat(),
                     "maturity_date": ep.date.iloc[-1].date().isoformat(), "sessions": len(ep),
                     "fee_bps_per_side": float(manifest["fee_bps_per_side"]), "signal_count": 29,
                     "signal_data_snapshot_sha256": "", "outcome_panel_sha256": snapshot_hash(ep),
                     "policy": "buy_and_hold", **bh_metrics})
    columns = ["model_freeze_id", "candidate", "listing_event_id", "stock_code", "listing_date",
               "maturity_date", "sessions", "fee_bps_per_side", "signal_count",
               "signal_data_snapshot_sha256", "outcome_panel_sha256", "policy",
               "final_return", "max_drawdown", "one_way_turnover"]
    fresh = pd.DataFrame(rows, columns=columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size > 0:
        try:
            prior = pd.read_csv(args.output)
        except pd.errors.EmptyDataError:
            prior = pd.DataFrame(columns=columns)
        if not prior.empty:
            fresh = pd.concat([prior, fresh], ignore_index=True)
    if not fresh.empty:
        fresh = fresh.drop_duplicates(["model_freeze_id", "candidate", "listing_event_id"], keep="first")
    fresh.to_csv(args.output, index=False)
    print(json.dumps({"eligible_complete_episodes": int(panel.groupby("listing_event_id").size().ge(30).sum()),
                      "new_scored_rows": len(rows), "stored_rows": len(fresh), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
