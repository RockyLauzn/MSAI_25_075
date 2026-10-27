"""Event-driven cash-account simulator for one-lot HK IPO subscriptions.

This is a deterministic expected-outcome research simulator, not an execution
backtest. It replays verified historical allocation groups and listing-day
Yahoo open prices while enforcing a finite cash budget.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from hkipo.subscription_net_returns import ipo_side_costs, sale_side_costs

EXACT_DATE_SOURCES = {"official_announcement", "official_prospectus_timetable",
                      "official_prospectus_pdf_timetable"}
INFERRED_DATE_SOURCES = {"official_prospectus_timetable_year_inferred",
                         "official_prospectus_pdf_timetable_year_inferred"}


def _distribution(group: pd.DataFrame) -> list[tuple[float, int]]:
    counts = pd.to_numeric(group.applicants_in_outcome_group, errors="raise").astype(int)
    shares = pd.to_numeric(group.allocated_shares_per_applicant, errors="raise").astype(int)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("One-lot allocation group has no applicants")
    return [(int(n) / total, int(q)) for n, q in zip(counts, shares)]


def _nav(cash: float, frozen: float, holdings: dict[str, tuple[float, float]]) -> float:
    return cash + frozen + sum(shares * mark for shares, mark in holdings.values())


def simulate(events: pd.DataFrame, outcomes: pd.DataFrame, *, initial_cash: float,
             policy: str = "always", cash_lock_multiplier: float = 1.0,
             fallback_lock_days: int = 5, sale_commission: float = 0.001,
             fixed_application_fee: float = 0.0, annual_opportunity_rate: float = 0.0,
             actions: dict[str, int] | None = None, cash_terms: pd.DataFrame | None = None,
             require_verified_cash_terms: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if initial_cash <= 0 or cash_lock_multiplier <= 0 or fallback_lock_days < 0:
        raise ValueError("Initial cash and cash lock multiplier must be positive; fallback days nonnegative")
    if policy not in {"always", "skip", "action_file"}:
        raise ValueError("policy must be always, skip, or action_file")
    if policy == "action_file" and actions is None:
        raise ValueError("action_file policy requires an event_id -> {0,1} mapping")

    events = events.copy()
    if cash_terms is not None:
        terms = cash_terms[["listing_event_id", "exact_max_application_price_hkd",
                            "cash_term_source", "term_price_valid"]].copy()
        terms["listing_event_id"] = terms.listing_event_id.astype(str)
        if terms.listing_event_id.duplicated().any():
            raise ValueError("Cash-term event ids must be unique")
        events["listing_event_id"] = events.listing_event_id.astype(str)
        events = events.merge(terms, on="listing_event_id", how="left", validate="one_to_one")
        events["term_price_valid"] = events.term_price_valid.fillna(False).astype(bool)
        events["exact_max_application_price_hkd"] = pd.to_numeric(
            events.exact_max_application_price_hkd, errors="coerce")
        events["cash_term_source"] = events.cash_term_source.fillna("missing")
    else:
        events["term_price_valid"] = False
        events["exact_max_application_price_hkd"] = pd.NA
        events["cash_term_source"] = "not_supplied"
    for col in ("decision_date", "result_date", "listing_date"):
        events[col] = pd.to_datetime(events[col]).dt.normalize()
    if "application_close_date_source" in events:
        trusted_dates = EXACT_DATE_SOURCES | INFERRED_DATE_SOURCES
        estimated = ~events.application_close_date_source.isin(trusted_dates)
        events.loc[estimated, "decision_date"] = (
            events.loc[estimated, "result_date"] - pd.to_timedelta(fallback_lock_days, unit="D"))
    events = events.sort_values(["decision_date", "listing_date", "stock_code"]).reset_index(drop=True)
    if events.listing_event_id.duplicated().any():
        raise ValueError("Simulator event ids must be unique")

    event_outcomes: dict[str, list[tuple[float, int]]] = {}
    for event_id, group in outcomes.groupby("listing_event_id", sort=False):
        event_outcomes[str(event_id)] = _distribution(group)
    if set(events.listing_event_id.astype(str)) - set(event_outcomes):
        raise ValueError("At least one simulator event has no one-lot outcome distribution")

    decisions_by_date: dict[date, list[pd.Series]] = defaultdict(list)
    results_by_date: dict[date, list[str]] = defaultdict(list)
    listings_by_date: dict[date, list[str]] = defaultdict(list)
    row_by_id: dict[str, pd.Series] = {}
    for _, row in events.iterrows():
        event_id = str(row.listing_event_id)
        decisions_by_date[row.decision_date.date()].append(row)
        results_by_date[row.result_date.date()].append(event_id)
        listings_by_date[row.listing_date.date()].append(event_id)
        row_by_id[event_id] = row

    cash = float(initial_cash)
    frozen = 0.0
    holdings: dict[str, tuple[float, float]] = {}
    pending: dict[str, dict] = {}
    applied: dict[str, dict] = {}
    ledger_rows: list[dict] = []
    decision_rows: list[dict] = []
    equity_rows: list[dict] = []
    all_dates = sorted(set(decisions_by_date) | set(results_by_date) | set(listings_by_date))
    total_ipo_fees = total_sale_fees = total_fixed_fees = total_opportunity = 0.0
    locked_cash_days = 0.0

    def record(date_value: date, event_id: str, entry_type: str, before: float,
               cash_delta: float = 0.0, frozen_delta: float = 0.0,
               detail: dict | None = None) -> None:
        after = _nav(cash, frozen, holdings)
        ledger_rows.append({"date": date_value.isoformat(), "listing_event_id": event_id,
                            "entry_type": entry_type, "cash_available_hkd": cash,
                            "frozen_cash_hkd": frozen, "portfolio_nav_hkd": after,
                            "nav_change_hkd": after - before, "cash_delta_hkd": cash_delta,
                            "frozen_delta_hkd": frozen_delta, **(detail or {})})
        equity_rows.append({"date": date_value.isoformat(), "portfolio_nav_hkd": after,
                            "cash_available_hkd": cash, "frozen_cash_hkd": frozen,
                            "held_expected_shares": sum(qty for qty, _ in holdings.values())})

    for current_date in all_dates:
        # Refund/settlement happens before listing-day sale and new applications.
        for event_id in results_by_date.get(current_date, []):
            if event_id not in pending:
                continue
            row = row_by_id[event_id]
            item = pending.pop(event_id)
            before = _nav(cash, frozen, holdings)
            distribution = event_outcomes[event_id]
            price = Decimal(str(row.ipo_price_hkd))
            expected_shares = sum(prob * shares for prob, shares in distribution)
            expected_purchase = 0.0
            expected_fee = 0.0
            expected_success_probability = 0.0
            for prob, shares in distribution:
                if shares <= 0:
                    continue
                expected_success_probability += prob
                expected_purchase += prob * shares * float(price)
                expected_fee += prob * float(ipo_side_costs(shares, price))
            locked = item["locked_cash_hkd"]
            refund = max(locked - expected_purchase - expected_fee, 0.0)
            shortfall = max(expected_purchase + expected_fee - locked, 0.0)
            lock_days = max((current_date - item["decision_date"]).days, 0)
            opportunity = locked * annual_opportunity_rate * lock_days / 365.0
            frozen = max(0.0, frozen - locked)
            cash += refund - shortfall - opportunity
            holdings[event_id] = (expected_shares, float(price))
            item.update({"expected_allocated_shares": expected_shares,
                         "success_probability": expected_success_probability,
                         "expected_purchase_cost_hkd": expected_purchase,
                         "expected_ipo_fee_hkd": expected_fee,
                         "expected_refund_hkd": refund,
                         "cash_shortfall_hkd": shortfall,
                         "fixed_application_fee_hkd": item["fixed_application_fee_hkd"],
                         "opportunity_cost_hkd": opportunity,
                         "lock_days": lock_days})
            applied[event_id] = item
            total_ipo_fees += expected_fee
            total_opportunity += opportunity
            locked_cash_days += locked * lock_days
            record(current_date, event_id, "allotment_and_refund", before,
                   cash_delta=refund - shortfall - opportunity, frozen_delta=-locked,
                   detail={"expected_allocated_shares": expected_shares,
                           "success_probability": expected_success_probability,
                           "expected_purchase_cost_hkd": expected_purchase,
                           "expected_ipo_fee_hkd": expected_fee,
                           "refund_hkd": refund, "cash_shortfall_hkd": shortfall,
                           "opportunity_cost_hkd": opportunity, "lock_days": lock_days})

        for event_id in listings_by_date.get(current_date, []):
            if event_id not in holdings:
                continue
            row = row_by_id[event_id]
            before = _nav(cash, frozen, holdings)
            distribution = event_outcomes[event_id]
            price = Decimal(str(row.first_day_open_hkd))
            expected_gross = 0.0
            expected_sale_fee = 0.0
            for prob, shares in distribution:
                if shares <= 0:
                    continue
                expected_gross += prob * shares * float(price)
                expected_sale_fee += prob * float(sale_side_costs(shares, price, Decimal(str(sale_commission))))
            proceeds = expected_gross - expected_sale_fee
            quantity, _mark = holdings.pop(event_id)
            cash += proceeds
            total_sale_fees += expected_sale_fee
            item = applied[event_id]
            item["expected_sale_gross_hkd"] = expected_gross
            item["expected_sale_fee_hkd"] = expected_sale_fee
            item["expected_sale_proceeds_hkd"] = proceeds
            item["realized_expected_pnl_hkd"] = (proceeds - item["expected_purchase_cost_hkd"]
                                                   - item["expected_ipo_fee_hkd"]
                                                   - item["fixed_application_fee_hkd"]
                                                   - item["opportunity_cost_hkd"])
            record(current_date, event_id, "listing_day_sale", before, cash_delta=proceeds,
                   detail={"expected_shares_sold": quantity,
                           "sale_price_proxy_hkd": float(price),
                           "sale_gross_hkd": expected_gross,
                           "sale_fee_hkd": expected_sale_fee,
                           "sale_proceeds_hkd": proceeds,
                           "realized_expected_pnl_hkd": item["realized_expected_pnl_hkd"]})

        for row in decisions_by_date.get(current_date, []):
            event_id = str(row.listing_event_id)
            if policy == "always":
                requested_action = 1
            elif policy == "skip":
                requested_action = 0
            else:
                requested_action = int(actions.get(event_id, 0))
            if requested_action not in (0, 1):
                raise ValueError(f"Invalid action for {event_id}: {requested_action}")
            valid_terms = bool(row.term_price_valid)
            principal = (float(row.exact_max_application_price_hkd)
                         * int(row.minimum_application_shares)
                         if valid_terms else float(row.one_lot_subscription_cash_proxy_hkd))
            lock_price = (Decimal(str(row.exact_max_application_price_hkd)) if valid_terms
                          else Decimal(str(row.ipo_price_hkd)))
            lock_amount = principal * cash_lock_multiplier + float(
                ipo_side_costs(int(row.minimum_application_shares), lock_price))
            term_blocked = bool(requested_action and require_verified_cash_terms and not valid_terms)
            accepted = bool(requested_action and not term_blocked
                            and cash + 1e-9 >= lock_amount + fixed_application_fee)
            before = _nav(cash, frozen, holdings)
            cash_before = cash
            if accepted:
                cash -= lock_amount + fixed_application_fee
                frozen += lock_amount
                total_fixed_fees += fixed_application_fee
                pending[event_id] = {"decision_date": current_date,
                                     "locked_cash_hkd": lock_amount,
                                     "application_principal_proxy_hkd": principal,
                                     "fixed_application_fee_hkd": fixed_application_fee}
            decision_rows.append({"date": current_date.isoformat(), "listing_event_id": event_id,
                                  "action_requested": requested_action,
                                  "action_executed": int(accepted),
                                  "skip_reason": ("missing_verified_cash_term" if term_blocked else
                                                  "insufficient_available_cash" if requested_action and not accepted else ""),
                                  "available_cash_before_hkd": cash_before,
                                  "cash_lock_hkd": lock_amount if accepted else 0.0,
                                  "fixed_application_fee_hkd": fixed_application_fee if accepted else 0.0,
                                  "ipo_price_hkd_for_environment_only": float(row.ipo_price_hkd),
                                  "max_application_price_hkd": (float(row.exact_max_application_price_hkd)
                                                                 if valid_terms else pd.NA),
                                  "cash_term_source": row.cash_term_source,
                                  "first_day_open_hkd_for_reward_only": float(row.first_day_open_hkd)})
            record(current_date, event_id, "application_decision", before,
                   cash_delta=-lock_amount - fixed_application_fee if accepted else 0.0,
                   frozen_delta=lock_amount if accepted else 0.0,
                   detail={"action_requested": requested_action, "action_executed": int(accepted),
                           "application_cash_proxy_hkd": lock_amount if accepted else 0.0,
                           "fixed_application_fee_hkd": fixed_application_fee if accepted else 0.0})

    decisions = pd.DataFrame(decision_rows)
    ledger = pd.DataFrame(ledger_rows)
    equity = pd.DataFrame(equity_rows).drop_duplicates(["date", "portfolio_nav_hkd"], keep="last")
    if len(equity):
        equity["peak_nav_hkd"] = equity.portfolio_nav_hkd.cummax()
        equity["drawdown_pct"] = equity.portfolio_nav_hkd / equity.peak_nav_hkd - 1.0
    applied_rows = pd.DataFrame(applied.values())
    final_nav = _nav(cash, frozen, holdings)
    summary = {"policy": policy, "initial_cash_hkd": initial_cash, "ending_cash_hkd": cash,
               "ending_frozen_cash_hkd": frozen, "ending_held_events": len(holdings),
               "ending_nav_hkd": final_nav, "net_profit_hkd": final_nav - initial_cash,
               "net_return": final_nav / initial_cash - 1.0,
               "decisions": int(len(decisions)), "applications": int(decisions.action_executed.sum()) if len(decisions) else 0,
               "skipped_for_insufficient_cash": int(decisions.skip_reason.eq("insufficient_available_cash").sum()) if len(decisions) else 0,
               "skipped_for_missing_cash_terms": int(decisions.skip_reason.eq("missing_verified_cash_term").sum()) if len(decisions) else 0,
               "expected_ipo_fees_hkd": total_ipo_fees, "expected_sale_fees_hkd": total_sale_fees,
               "fixed_application_fees_hkd": total_fixed_fees,
               "opportunity_cost_hkd": total_opportunity,
               "capital_days_locked_hkd_days": locked_cash_days,
               "max_drawdown_pct": float(equity.drawdown_pct.min()) if len(equity) else 0.0,
               "cash_lock_multiplier": cash_lock_multiplier,
               "fallback_lock_days": fallback_lock_days,
               "annual_opportunity_rate": annual_opportunity_rate,
               "sale_commission": sale_commission,
               "allocation_model": "deterministic expected value over official one-lot outcome groups",
               "application_amount_model": ("maximum application price times minimum application shares plus estimated IPO charges where terms exist"
                                             if cash_terms is not None else
                                             "final offer price times minimum application shares plus IPO charges; not historical max-price deposit"),
               "require_verified_cash_terms": require_verified_cash_terms,
               "events_with_cash_terms": int(events.term_price_valid.sum()),
               "official_application_close_dates": int(events.application_close_date_source.isin(
                   EXACT_DATE_SOURCES).sum())
                   if "application_close_date_source" in events else 0,
               "inferred_year_timetable_dates": int(events.application_close_date_source.isin(
                   INFERRED_DATE_SOURCES).sum())
                   if "application_close_date_source" in events else 0,
               "scenario_application_close_dates": int((~events.application_close_date_source.isin(
                   EXACT_DATE_SOURCES | INFERRED_DATE_SOURCES)).sum())
                   if "application_close_date_source" in events else int(len(events)),
               "sale_price_model": "Yahoo/yfinance listing-day open; not guaranteed execution",
               "warnings": (["unpublished offer-price ranges are approximated by final offer price"]
                            if cash_terms is None or not require_verified_cash_terms else [])
                           + (["missing application-close dates use a configurable calendar-day cash-lock scenario"]
                              if "application_close_date_source" in events and
                              (~events.application_close_date_source.isin(
                                  EXACT_DATE_SOURCES | INFERRED_DATE_SOURCES)).any() else [])
                           + (["two prospectus dates have year-inferred timetable entries"]
                              if "application_close_date_source" in events and
                              events.application_close_date_source.isin(INFERRED_DATE_SOURCES).any() else [])
                           + ["expected shares and proceeds can be fractional and are not a realizable single-account path"]}
    return decisions, ledger, equity, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("reports/hkipo_simulator_data_2021_2025/verified_events.csv"))
    parser.add_argument("--outcomes", type=Path, default=Path("reports/hkipo_simulator_data_2021_2025/one_lot_outcomes.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=["always", "skip", "action_file"], default="always")
    parser.add_argument("--actions", type=Path,
                        help="CSV with listing_event_id and action (0=skip, 1=apply) for action_file policy")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--cash-lock-multiplier", type=float, default=1.0)
    parser.add_argument("--fallback-lock-days", type=int, default=5)
    parser.add_argument("--sale-commission", type=float, default=0.001)
    parser.add_argument("--fixed-application-fee", type=float, default=0.0)
    parser.add_argument("--annual-opportunity-rate", type=float, default=0.0)
    parser.add_argument("--cash-terms", type=Path,
                        help="CSV produced by build_verified_cash_terms.py")
    parser.add_argument("--require-verified-cash-terms", action="store_true",
                        help="Skip applications whose maximum application price has not been recovered")
    args = parser.parse_args()
    actions = None
    if args.policy == "action_file":
        if args.actions is None:
            parser.error("--actions is required when --policy action_file is selected")
        action_frame = pd.read_csv(args.actions, dtype={"listing_event_id": str})
        if not {"listing_event_id", "action"}.issubset(action_frame.columns):
            parser.error("--actions CSV must contain listing_event_id and action columns")
        if action_frame.listing_event_id.duplicated().any():
            parser.error("--actions CSV must contain at most one action per listing_event_id")
        actions = dict(zip(action_frame.listing_event_id, pd.to_numeric(action_frame.action, errors="raise").astype(int)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(args.events, dtype={"listing_event_id": str, "stock_code": str})
    outcomes = pd.read_csv(args.outcomes, dtype={"listing_event_id": str})
    cash_terms = (pd.read_csv(args.cash_terms, dtype={"listing_event_id": str})
                  if args.cash_terms else None)
    decisions, ledger, equity, summary = simulate(
        events, outcomes, initial_cash=args.initial_cash, policy=args.policy,
        cash_lock_multiplier=args.cash_lock_multiplier, fallback_lock_days=args.fallback_lock_days,
        sale_commission=args.sale_commission, fixed_application_fee=args.fixed_application_fee,
        annual_opportunity_rate=args.annual_opportunity_rate, actions=actions,
        cash_terms=cash_terms, require_verified_cash_terms=args.require_verified_cash_terms)
    decisions.to_csv(args.output_dir / "decisions.csv", index=False)
    ledger.to_csv(args.output_dir / "ledger.csv", index=False)
    equity.to_csv(args.output_dir / "equity.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
