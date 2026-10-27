"""Estimate one-lot HK IPO subscription P&L from allotment outcomes.

This is an exploratory event study. It uses official 2025 allotment tables,
HKEX fee rates and Yahoo/yfinance listing-day OHLC, with explicit sell-broker
commission scenarios. It does not model financing or capital opportunity cost.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

CENT = Decimal("0.01")
IPO_BROKERAGE = Decimal("0.01")
SFC_LEVY = Decimal("0.000027")
AFRC_LEVY = Decimal("0.0000015")
TRADING_FEE = Decimal("0.0000565")
STAMP_DUTY = Decimal("0.001")
SELL_COMMISSION_SCENARIOS = (Decimal("0"), Decimal("0.001"), Decimal("0.0025"))
APPLICATION_FEE_SCENARIOS = (Decimal("0"), Decimal("50"), Decimal("100"))


def _money(value: float | int | str | Decimal) -> Decimal:
    return Decimal(str(value))


def _round_cent(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def ipo_side_costs(shares: int, offer_price: Decimal) -> Decimal:
    consideration = shares * offer_price
    # HKEX IPO subscription brokerage is 1%; SFC, AFRC and Exchange fees are
    # calculated separately and rounded to cents, as in HKEX's fee example.
    return sum((_round_cent(consideration * rate) for rate in
                (IPO_BROKERAGE, SFC_LEVY, AFRC_LEVY, TRADING_FEE)), Decimal("0"))


def sale_side_costs(shares: int, sale_price: Decimal, commission_rate: Decimal) -> Decimal:
    consideration = shares * sale_price
    # Stamp duty is rounded up to the nearest HKD; other ad valorem charges to cents.
    stamp = (consideration * STAMP_DUTY).quantize(Decimal("1"), rounding=ROUND_CEILING)
    other = sum((_round_cent(consideration * rate) for rate in
                 (commission_rate, SFC_LEVY, AFRC_LEVY, TRADING_FEE)), Decimal("0"))
    return stamp + other


def estimate_one_lot(events: pd.DataFrame, tiers: pd.DataFrame, outcomes: pd.DataFrame,
                     first_day: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    audit = pd.read_csv(events.attrs["audit_path"], dtype=str)
    eligible = set(audit.loc[
        audit.source_status.eq("parsed")
        & audit.matches_official_totals.eq("True")
        & audit.all_allocation_rates_reconciled.eq("True"), "listing_event_id"
    ])
    verified_event_count = len(eligible)
    events = events[events.listing_event_id.isin(eligible)].copy()
    first_day = first_day[first_day.ohlc_consistent.astype(str).str.lower().eq("true")].copy()
    price_fields = first_day.set_index("listing_event_id")
    verified_with_price_count = len(eligible.intersection(set(price_fields.index)))

    rows: list[dict] = []
    for event in events.itertuples(index=False):
        if event.listing_event_id not in price_fields.index:
            continue
        lot = int(event.lot_size)
        tier = tiers[(tiers.listing_event_id == event.listing_event_id)
                     & (pd.to_numeric(tiers.applied_shares, errors="coerce") == lot)]
        # One-lot subscription is the smallest standard application tier.
        if len(tier) != 1:
            continue
        valid = int(float(tier.iloc[0].total_valid_applications))
        og = outcomes[(outcomes.listing_event_id == event.listing_event_id)
                      & (pd.to_numeric(outcomes.applied_shares, errors="coerce") == lot)]
        if not len(og) or int(pd.to_numeric(og.applicants_in_outcome_group).sum()) != valid:
            continue
        px = price_fields.loc[event.listing_event_id]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[0]
        offer = _money(event.final_offer_price)
        capital = offer * lot
        allocations = [(int(r.applicants_in_outcome_group), int(r.allocated_shares_per_applicant))
                       for r in og.itertuples(index=False)]
        record = {
            "listing_event_id": event.listing_event_id,
            "stock_code": event.stock_code,
            "listing_date": event.listing_date,
            "lot_size": lot,
            "ipo_price_hkd": float(offer),
            "one_lot_subscription_hkd": float(capital),
            "valid_one_lot_applications": valid,
            "allocation_probability": sum(n for n, shares in allocations if shares > 0) / valid,
            "expected_allocated_shares": sum(n * shares for n, shares in allocations) / valid,
            "first_day_open_hkd": float(px.open),
            "first_day_close_hkd": float(px.close),
            "price_source": px.source,
        }
        for exit_name, price in (("open", _money(px.open)), ("close", _money(px.close))):
            record[f"gross_return_if_allocated_{exit_name}"] = float(price / offer - 1)
            for commission in SELL_COMMISSION_SCENARIOS:
                for application_fee in APPLICATION_FEE_SCENARIOS:
                    suffix = (f"{exit_name}_sell_commission_{commission * 100:g}pct"
                              f"_application_fee_{application_fee:g}hkd")
                    expected_pnl = Decimal("0")
                    for n_applicants, shares in allocations:
                        if shares == 0:
                            continue
                        pnl = (shares * price
                               - sale_side_costs(shares, price, commission)
                               - shares * offer
                               - ipo_side_costs(shares, offer))
                        expected_pnl += Decimal(n_applicants) / Decimal(valid) * pnl
                    expected_pnl -= application_fee
                    record[f"expected_net_hkd_per_application_{suffix}"] = float(expected_pnl)
                    record[f"expected_net_return_on_one_lot_deposit_{suffix}"] = float(expected_pnl / capital)
        rows.append(record)

    result = pd.DataFrame(rows).sort_values(["listing_date", "listing_event_id"]).reset_index(drop=True)
    summary: dict = {"verified_allotment_events": int(verified_event_count),
                     "verified_events_with_first_day_prices": int(verified_with_price_count),
                     "events_in_estimate": int(len(result)),
                     "price_source": "Yahoo Finance via yfinance",
                     "allocation_source": "HKEX official allotment announcements",
                     "cost_method": "HKEX IPO fees plus sell-side HKEX levies and stamp duty; sell brokerage scenarios",
                     "excluded_costs": ["subscription financing and cash opportunity cost", "broker-specific platform/handling/CCASS fees", "market impact and slippage"],
                     "by_scenario": {}}
    for exit_name in ("open", "close"):
        for commission in SELL_COMMISSION_SCENARIOS:
            for application_fee in APPLICATION_FEE_SCENARIOS:
                suffix = (f"{exit_name}_sell_commission_{commission * 100:g}pct"
                          f"_application_fee_{application_fee:g}hkd")
                pnl = result[f"expected_net_hkd_per_application_{suffix}"]
                roi = result[f"expected_net_return_on_one_lot_deposit_{suffix}"]
                summary["by_scenario"][suffix] = {
                    "mean_expected_net_hkd_per_application": float(pnl.mean()),
                    "median_expected_net_hkd_per_application": float(pnl.median()),
                    "positive_event_share": float(pnl.gt(0).mean()),
                    "mean_return_on_one_lot_deposit": float(roi.mean()),
                    "median_return_on_one_lot_deposit": float(roi.median()),
                    "negative_event_share": float(pnl.lt(0).mean()),
                    "p05_event_expected_net_hkd": float(pnl.quantile(0.05)),
                    "p95_event_expected_net_hkd": float(pnl.quantile(0.95)),
                    "worst_event_expected_net_hkd": float(pnl.min()),
                    "best_event_expected_net_hkd": float(pnl.max()),
                }
    summary["mean_allocation_probability"] = float(result.allocation_probability.mean())
    summary["mean_expected_allocated_shares"] = float(result.expected_allocated_shares.mean())
    summary["note"] = ("Each IPO contributes one hypothetical one-lot application. Event P&L is expected over the published "
                       "allotment outcome groups. Returns are not a capital-weighted or capacity-constrained portfolio result.")
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--allotment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    events = pd.read_csv(args.allotment_dir / "hkipo_allotment_events.csv", dtype=str)
    events.attrs["audit_path"] = str(args.allotment_dir / "parse_audit.csv")
    tiers = pd.read_csv(args.allotment_dir / "hkipo_allotment_tiers.csv", dtype=str)
    outcomes = pd.read_csv(args.allotment_dir / "hkipo_allotment_outcomes.csv", dtype=str)
    panel = pd.read_csv(args.panel)
    panel["listing_year"] = pd.to_datetime(panel.listing_date).dt.year
    first_day = panel[(panel.listing_year == 2025) & (pd.to_numeric(panel.listed_day) == 1)].copy()
    result, summary = estimate_one_lot(events, tiers, outcomes, first_day)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "one_lot_net_returns.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
