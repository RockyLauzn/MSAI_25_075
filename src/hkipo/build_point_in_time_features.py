"""Build candidate IPO decision features using only information available before each decision."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _lead_sponsor(value: object) -> str:
    if pd.isna(value):
        return ""
    # HKEX's annual report uses slash to separate co-sponsors and line breaks
    # within a sponsor's legal name. Keep the first listed sponsor only.
    name = str(value).split("/")[0]
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"\blimited\b", "ltd", name, flags=re.I)
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()


def build(*, root: Path, events_path: Path, pnl_path: Path,
          output_dir: Path) -> dict:
    events = pd.read_csv(events_path, dtype={"listing_event_id": str, "stock_code": str},
                         parse_dates=["decision_date", "listing_date", "result_date"])
    events["stock_code"] = events.stock_code.str.zfill(5)
    master = pd.read_csv(root / "data/processed/hkex_ipos_2021_2025.csv",
                         dtype={"stock_code": str},
                         parse_dates=["listing_date", "prospectus_date"])
    master["stock_code"] = master.stock_code.str.zfill(5)
    master["listing_event_id"] = (master.stock_code + "_" +
                                  master.listing_date.dt.strftime("%Y%m%d"))
    events = events.merge(master[["listing_event_id", "prospectus_date", "sponsors"]],
                          on="listing_event_id", how="left", validate="one_to_one")
    if events.prospectus_date.isna().any() or events.sponsors.isna().any():
        raise ValueError("Every simulator event must match a dated prospectus and sponsor record")
    if events.prospectus_date.gt(events.decision_date).any():
        raise ValueError("At least one prospectus date falls after its decision-date proxy")

    pnl = pd.read_csv(pnl_path, dtype={"listing_event_id": str})
    if pnl.listing_event_id.duplicated().any():
        raise ValueError("Expected one historical outcome per event")
    outcomes = pnl[["listing_event_id", "realized_expected_pnl_hkd"]].rename(
        columns={"realized_expected_pnl_hkd": "historical_one_lot_net_pnl_hkd"})
    events = events.merge(outcomes, on="listing_event_id", how="left", validate="one_to_one")
    if events.historical_one_lot_net_pnl_hkd.isna().any():
        raise ValueError("At least one event lacks an outcome for prior-history construction")
    events["historical_one_lot_net_return"] = (
        events.historical_one_lot_net_pnl_hkd / events.one_lot_subscription_cash_proxy_hkd)
    events["lead_sponsor_key"] = events.sponsors.map(_lead_sponsor)

    feature_rows: list[dict] = []
    for row in events.sort_values(["decision_date", "listing_event_id"]).itertuples(index=False):
        # A past event is eligible only after its listing date, when its realized
        # allocation and listing-price outcome would already have been public.
        prior = events.loc[events.listing_date.lt(row.decision_date)]
        prior_sponsor = prior.loc[prior.lead_sponsor_key.eq(row.lead_sponsor_key)]
        feature_rows.append({
            "listing_event_id": row.listing_event_id,
            "decision_date": row.decision_date.date().isoformat(),
            "decision_date_source": row.application_close_date_source,
            "prospectus_date": row.prospectus_date.date().isoformat(),
            "days_since_prospectus": int((row.decision_date - row.prospectus_date).days),
            "lead_sponsor_key": row.lead_sponsor_key,
            "prior_verified_ipo_count": int(len(prior)),
            "prior_verified_mean_net_return": float(prior.historical_one_lot_net_return.mean())
                if len(prior) else pd.NA,
            "prior_verified_positive_fraction": float(
                prior.historical_one_lot_net_pnl_hkd.gt(0).mean()) if len(prior) else pd.NA,
            "prior_sponsor_ipo_count": int(len(prior_sponsor)),
            "prior_sponsor_mean_net_pnl_hkd": float(
                prior_sponsor.historical_one_lot_net_pnl_hkd.mean()) if len(prior_sponsor) else pd.NA,
            "prior_sponsor_positive_fraction": float(
                prior_sponsor.historical_one_lot_net_pnl_hkd.gt(0).mean()) if len(prior_sponsor) else pd.NA,
        })

    features = pd.DataFrame(feature_rows)
    labels = events[["listing_event_id", "listing_date", "historical_one_lot_net_pnl_hkd"]].copy()
    labels["listing_date"] = labels.listing_date.dt.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "candidate_features.csv", index=False)
    labels.to_csv(output_dir / "candidate_labels_separate.csv", index=False)

    coverage = features.assign(decision_year=pd.to_datetime(features.decision_date).dt.year)
    exact_sources = {"official_announcement", "official_prospectus_timetable",
                     "official_prospectus_pdf_timetable"}
    inferred_sources = {"official_prospectus_timetable_year_inferred",
                        "official_prospectus_pdf_timetable_year_inferred"}
    inferred_year_dates = int(features.decision_date_source.isin(inferred_sources).sum())
    scenario_dates = int((~features.decision_date_source.isin(
        exact_sources | inferred_sources)).sum())
    summary = {
        "events": int(len(features)),
        "features_exclude_current_event_outcome": True,
        "past_outcome_eligibility": "listing_date strictly before decision_date",
        "official_decision_dates": int(features.decision_date_source.isin(exact_sources).sum()),
        "inferred_year_timetable_dates": inferred_year_dates,
        "scenario_decision_dates": scenario_dates,
        "events_with_any_prior_verified_history": int(features.prior_verified_ipo_count.gt(0).sum()),
        "events_with_prior_sponsor_history": int(features.prior_sponsor_ipo_count.gt(0).sum()),
        "sponsor_history_count_ge_3": int(features.prior_sponsor_ipo_count.ge(3).sum()),
        "by_decision_year": coverage.groupby("decision_year").agg(
            events=("listing_event_id", "size"),
            with_prior_sponsor=("prior_sponsor_ipo_count", lambda s: int(s.gt(0).sum())),
            with_three_prior_sponsor_events=("prior_sponsor_ipo_count", lambda s: int(s.ge(3).sum()))
        ).reset_index().to_dict(orient="records"),
        "limitations": [
            f"{scenario_dates} of {len(features)} decision dates are fallback scenarios, not recovered application-close dates",
            "sponsor identity uses first listed sponsor and legal-name normalization; aliases may remain",
            "historical outcome features use only verified events with listing date before the decision",
            "candidate features omit account state and cash requirement; recoverable maximum-price terms must be added before training",
            "strict simulator runs can enforce verified maximum application prices, but point-in-time states are not complete for DRL",
        ],
    }
    (output_dir / "feature_coverage.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--events", type=Path,
                        default=Path("reports/hkipo_simulator_data_2021_2025/verified_events.csv"))
    parser.add_argument("--pnl", type=Path,
                        default=Path("reports/hkipo_simulator_2021_2025_always/ledger.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/hkipo_point_in_time_features_2021_2025"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pnl = pd.read_csv(args.pnl, dtype={"listing_event_id": str})
    pnl = pnl.loc[pnl.entry_type.eq("listing_day_sale"),
                  ["listing_event_id", "realized_expected_pnl_hkd"]]
    pnl.to_csv(args.output_dir / "_temporary_event_pnl.csv", index=False)
    try:
        print(json.dumps(build(root=args.root.resolve(), events_path=args.events,
                               pnl_path=args.output_dir / "_temporary_event_pnl.csv",
                               output_dir=args.output_dir), indent=2))
    finally:
        (args.output_dir / "_temporary_event_pnl.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
