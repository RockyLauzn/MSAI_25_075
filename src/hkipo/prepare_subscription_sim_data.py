"""Prepare strictly audited multi-year one-lot IPO simulation inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from hkipo.parse_allotment_pdf import _application_close_date, extract_pdf_text


def _read_bundle(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = pd.read_csv(folder / "parse_audit.csv", dtype={"listing_event_id": str})
    events = pd.read_csv(folder / "hkipo_allotment_events.csv", dtype={"listing_event_id": str})
    tiers = pd.read_csv(folder / "hkipo_allotment_tiers.csv", dtype={"listing_event_id": str})
    outcomes = pd.read_csv(folder / "hkipo_allotment_outcomes.csv", dtype={"listing_event_id": str})
    audit["fully_verified"] = (
        audit.source_status.eq("parsed")
        & audit.matches_official_totals.fillna(False).astype(bool)
        & audit.all_allocation_rates_reconciled.fillna(False).astype(bool)
    )
    return audit, events, tiers, outcomes


def _extract_close_date(root: Path, relative_path: str) -> tuple[str | None, str]:
    path = (root / Path(relative_path)).resolve()
    try:
        text, _ = extract_pdf_text(path)
        return _application_close_date(text), ""
    except Exception as exc:  # record failures; never infer a date silently here
        return None, f"{type(exc).__name__}: {exc}"


def prepare(*, root: Path, output_dir: Path, fallback_lock_days: int = 5) -> dict:
    folders = [root / "data/processed/hkipo_allotments_2021_2024",
               root / "data/processed/hkipo_allotments_2025"]
    bundles = [_read_bundle(folder) for folder in folders]
    audits = pd.concat([bundle[0] for bundle in bundles], ignore_index=True)
    events = pd.concat([bundle[1] for bundle in bundles], ignore_index=True)
    tiers = pd.concat([bundle[2] for bundle in bundles], ignore_index=True)
    outcomes = pd.concat([bundle[3] for bundle in bundles], ignore_index=True)
    clean_ids = set(audits.loc[audits.fully_verified, "listing_event_id"])
    events = events[events.listing_event_id.isin(clean_ids)].copy()
    tiers = tiers[tiers.listing_event_id.isin(clean_ids)].copy()
    outcomes = outcomes[outcomes.listing_event_id.isin(clean_ids)].copy()

    ipo_master = pd.read_csv(root / "data/processed/hkex_ipos_2021_2025.csv",
                             dtype={"stock_code": str}, parse_dates=["listing_date"])
    ipo_master["stock_code"] = ipo_master.stock_code.str.zfill(5)
    ipo_master["listing_event_id"] = (
        ipo_master.stock_code + "_" + ipo_master.listing_date.dt.strftime("%Y%m%d"))
    if ipo_master.listing_event_id.duplicated().any():
        raise ValueError("IPO master has duplicate listing_event_id values")

    bars = pd.read_csv(root / "data/processed/hkipo_daily_bars_2021_2025.csv",
                       dtype={"stock_code": str}, parse_dates=["date"])
    bars["stock_code"] = bars.stock_code.str.zfill(5)
    if bars.duplicated(["stock_code", "date"]).any():
        raise ValueError("Daily bars contain duplicate stock_code/date rows")
    listing_bars = bars.merge(
        ipo_master[["listing_event_id", "stock_code", "listing_date"]],
        left_on=["stock_code", "date"], right_on=["stock_code", "listing_date"], how="inner")
    listing_bars = listing_bars[["listing_event_id", "open", "close", "source", "turnover_is_estimate"]]

    audit_paths = audits.loc[audits.fully_verified, ["listing_event_id", "pdf_path"]].copy()
    with ThreadPoolExecutor(max_workers=4) as pool:
        close_results = list(pool.map(
            lambda path: _extract_close_date(root, path), audit_paths.pdf_path.astype(str)))
    audit_paths["application_close_date"] = [date for date, _ in close_results]
    audit_paths["date_extract_error"] = [error for _, error in close_results]

    events["stock_code"] = events.stock_code.astype(str).str.zfill(5)
    events["listing_date"] = pd.to_datetime(events.listing_date)
    events["result_date"] = pd.to_datetime(events.announcement_release_time_hkt).dt.normalize()
    events = events.merge(
        ipo_master[["listing_event_id", "ipo_price", "company_name"]],
        on="listing_event_id", how="left", suffixes=("", "_master"))
    events["ipo_price"] = pd.to_numeric(events.ipo_price, errors="coerce")
    events["final_offer_price"] = pd.to_numeric(events.final_offer_price, errors="coerce")
    price_diff = (events.ipo_price - events.final_offer_price).abs()
    events["offer_price_matches_master"] = price_diff.le(0.011)

    events = events.merge(audit_paths[["listing_event_id", "application_close_date", "date_extract_error"]],
                          on="listing_event_id", how="left")
    events = events.merge(listing_bars, on="listing_event_id", how="left")
    events["lot_size"] = pd.to_numeric(events.lot_size, errors="coerce")

    one_lot_tiers = tiers[pd.to_numeric(tiers.applied_shares, errors="coerce").eq(
        tiers.listing_event_id.map(events.set_index("listing_event_id").lot_size))].copy()
    duplicate_one_lot = set(one_lot_tiers.listing_event_id[one_lot_tiers.listing_event_id.duplicated(keep=False)])
    one_lot_outcomes = outcomes.merge(
        events[["listing_event_id", "lot_size"]], on="listing_event_id", how="inner")
    one_lot_outcomes = one_lot_outcomes[
        pd.to_numeric(one_lot_outcomes.applied_shares, errors="coerce").eq(one_lot_outcomes.lot_size)
    ].copy()
    outcome_counts = one_lot_outcomes.groupby("listing_event_id").applicants_in_outcome_group.sum()
    tier_valid = one_lot_tiers.set_index("listing_event_id").total_valid_applications
    invalid_outcomes = {event_id for event_id, count in outcome_counts.items()
                        if event_id not in tier_valid.index or int(count) != int(tier_valid.loc[event_id])}
    one_lot_outcomes = one_lot_outcomes[~one_lot_outcomes.listing_event_id.isin(invalid_outcomes)].copy()

    exclusion_rows: list[dict] = []
    usable_rows: list[dict] = []
    for row in events.itertuples(index=False):
        reasons = []
        event_id = row.listing_event_id
        if not bool(row.offer_price_matches_master):
            reasons.append("final offer price does not match IPO master")
        if pd.isna(row.lot_size) or row.lot_size <= 0:
            reasons.append("minimum application shares missing")
        if event_id not in set(one_lot_tiers.listing_event_id):
            reasons.append("no unique one-lot allocation tier")
        if event_id in duplicate_one_lot:
            reasons.append("multiple allocation pools have the same minimum application size")
        if event_id in invalid_outcomes or event_id not in set(one_lot_outcomes.listing_event_id):
            reasons.append("one-lot outcome groups do not reconcile to tier applicant count")
        if pd.isna(row.open) or pd.isna(row.close):
            reasons.append("listing-day open/close price unavailable")
        close_date = pd.to_datetime(row.application_close_date, errors="coerce")
        if not pd.isna(close_date) and close_date > row.result_date:
            reasons.append("application close date occurs after allotment announcement")
        if reasons:
            exclusion_rows.append({"listing_event_id": event_id, "reasons": "; ".join(reasons)})
        else:
            usable_rows.append({"listing_event_id": event_id, "result_date": row.result_date,
                                "application_close_date": "" if pd.isna(close_date) else close_date.date().isoformat(),
                                "application_close_date_source": "official_announcement" if not pd.isna(close_date) else "missing_use_simulator_scenario",
                                "cash_lock_proxy_days": fallback_lock_days if pd.isna(close_date) else (row.result_date - close_date).days,
                                "stock_code": row.stock_code, "company_name": row.company_name,
                                "listing_date": row.listing_date.date().isoformat(),
                                "ipo_price_hkd": row.final_offer_price,
                                "minimum_application_shares": int(row.lot_size),
                                "one_lot_subscription_cash_proxy_hkd": row.final_offer_price * row.lot_size,
                                "first_day_open_hkd": row.open, "first_day_close_hkd": row.close,
                                "price_source": row.source,
                                "turnover_is_estimate": row.turnover_is_estimate})

    event_input = pd.DataFrame(usable_rows).sort_values(["result_date", "stock_code"]).reset_index(drop=True)
    if len(event_input):
        decision_dates = pd.to_datetime(event_input.application_close_date, errors="coerce")
        fallback_dates = pd.to_datetime(event_input.result_date) - pd.to_timedelta(fallback_lock_days, unit="D")
        event_input["decision_date"] = decision_dates.fillna(fallback_dates).dt.strftime("%Y-%m-%d")
        event_input["result_date"] = pd.to_datetime(event_input.result_date).dt.strftime("%Y-%m-%d")
    else:
        event_input["decision_date"] = pd.Series(dtype=str)
    event_input = event_input.merge(
        audits[["listing_event_id", "pdf_path", "source_status", "matches_official_totals",
                "all_allocation_rates_reconciled"]], on="listing_event_id", how="left")
    one_lot_outcomes = one_lot_outcomes[
        one_lot_outcomes.listing_event_id.isin(set(event_input.listing_event_id))
    ].copy()
    one_lot_outcomes.to_csv(output_dir / "one_lot_outcomes.csv", index=False)
    event_input.to_csv(output_dir / "verified_events.csv", index=False)
    pd.DataFrame(exclusion_rows).to_csv(output_dir / "excluded_verified_events.csv", index=False)

    coverage_rows = []
    for year in range(2021, 2026):
        manifest = pd.read_csv(root / f"data/processed/hkex_allotment_manifest_{year}.csv",
                               dtype={"stock_code": str})
        manifest["listing_event_id"] = manifest.stock_code.str.zfill(5) + "_" + pd.to_datetime(
            manifest.listing_date).dt.strftime("%Y%m%d")
        audit_year = audits[audits.listing_event_id.isin(set(manifest.listing_event_id))]
        parsed = audit_year.source_status.eq("parsed")
        total_ok = audit_year.matches_official_totals.fillna(False).astype(bool)
        rates_ok = audit_year.all_allocation_rates_reconciled.fillna(False).astype(bool)
        coverage_rows.append({"year": year, "ipo_events": len(manifest),
                              "official_announcement_found": int(manifest.source_status.eq("found").sum()),
                              "pdf_parsed": int(parsed.sum()),
                              "summary_totals_reconciled": int((parsed & total_ok).sum()),
                              "all_tier_rates_reconciled": int((parsed & rates_ok).sum()),
                              "fully_verified": int((parsed & total_ok & rates_ok).sum()),
                              "simulator_events_with_listing_prices": int(
                                  event_input.listing_event_id.isin(set(manifest.listing_event_id)).sum())})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_dir / "coverage_by_year.csv", index=False)
    return {"allotment_fully_verified": int(audits.fully_verified.sum()),
            "event_rows_in_simulator": int(len(event_input)),
            "one_lot_outcome_rows": int(len(one_lot_outcomes)),
            "events_with_official_close_date": int(event_input.application_close_date.ne("").sum()),
            "events_using_cash_lock_scenario": int(event_input.application_close_date.eq("").sum()),
            "excluded_from_simulator": int(len(exclusion_rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hkipo_simulator_data_2021_2025"))
    parser.add_argument("--fallback-lock-days", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(prepare(root=args.root.resolve(), output_dir=args.output_dir, fallback_lock_days=args.fallback_lock_days))


if __name__ == "__main__":
    main()
