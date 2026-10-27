"""Add directly extracted HKEX prospectus timetable dates to simulator events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def apply_dates(*, events_path: Path, timetable_path: Path, output_path: Path,
                pdf_timetable_path: Path | None = None) -> dict:
    events = pd.read_csv(events_path, dtype={"listing_event_id": str})
    initial_official_sources = {"official_announcement", "official_prospectus_timetable",
                                "official_prospectus_pdf_timetable"}
    official_before = events.application_close_date_source.isin(initial_official_sources)
    records = json.loads(timetable_path.read_text(encoding="utf-8"))
    timetable = pd.DataFrame(records)
    if timetable.empty:
        raise ValueError("Timetable extraction file is empty")
    if pdf_timetable_path and pdf_timetable_path.exists():
        pdf_records = json.loads(pdf_timetable_path.read_text(encoding="utf-8"))
        pdf = pd.DataFrame(pdf_records)
        pdf = pdf.loc[pdf.status.eq("date_verified_in_pdf_body")].copy()
        if not pdf.empty:
            pdf = pdf.rename(columns={"event_id": "listing_event_id"})
            pdf["timetable_pdf_url"] = pdf.prospectus_url
            timetable = pd.concat([timetable, pdf], ignore_index=True, sort=False)
    required = ["listing_event_id", "application_close_date",
                "application_close_date_source", "timetable_pdf_url",
                "prospectus_release_date"]
    for col in required:
        if col not in timetable:
            timetable[col] = ""
    timetable = timetable[required].copy()
    timetable = timetable.rename(columns={
        "application_close_date": "timetable_application_close_date",
        "application_close_date_source": "timetable_date_source",
    })
    timetable["listing_event_id"] = timetable.listing_event_id.astype(str)
    timetable = timetable[timetable.timetable_application_close_date.fillna("").ne("")]
    source_counts = timetable.groupby("listing_event_id").timetable_application_close_date.nunique()
    conflicts = source_counts[source_counts.gt(1)]
    if not conflicts.empty:
        raise ValueError(f"Timetable sources disagree for event ids: {conflicts.index.tolist()}")
    timetable["_source_rank"] = timetable.timetable_date_source.map({
        "official_prospectus_pdf_timetable": 3,
        "official_prospectus_timetable": 2,
        "official_prospectus_pdf_timetable_year_inferred": 1,
        "official_prospectus_timetable_year_inferred": 1,
    }).fillna(0)
    timetable = (timetable.sort_values("_source_rank", ascending=False)
                 .drop_duplicates("listing_event_id", keep="first").drop(columns="_source_rank"))
    events = events.merge(timetable, on="listing_event_id", how="left", validate="one_to_one")
    proposed = pd.to_datetime(events.timetable_application_close_date, errors="coerce")
    result_date = pd.to_datetime(events.result_date, errors="coerce")
    listing_date = pd.to_datetime(events.listing_date, errors="coerce")
    release_date = pd.to_datetime(events.prospectus_release_date, errors="coerce")
    valid = (proposed.notna() & result_date.notna() & listing_date.notna()
             & release_date.notna() & proposed.ge(release_date)
             & proposed.le(result_date) & proposed.lt(listing_date)
             & (listing_date.sub(proposed).dt.days.le(45)))
    invalid = proposed.notna() & ~valid
    if invalid.any():
        bad = events.loc[invalid, "listing_event_id"].tolist()
        raise ValueError(f"Timetable dates failed chronology checks: {bad}")

    current = pd.to_datetime(events.application_close_date, errors="coerce")
    overlap = proposed.notna() & current.notna() & proposed.ne(current)
    if overlap.any():
        bad = events.loc[overlap, "listing_event_id"].tolist()
        raise ValueError(f"Prospectus timetable conflicts with existing official dates: {bad}")

    fill = proposed.notna() & current.isna()
    events.loc[fill, "application_close_date"] = proposed[fill].dt.strftime("%Y-%m-%d")
    events.loc[fill, "application_close_date_source"] = events.loc[fill, "timetable_date_source"]
    exact_sources = {"official_prospectus_timetable", "official_prospectus_pdf_timetable"}
    inferred_sources = {"official_prospectus_timetable_year_inferred",
                        "official_prospectus_pdf_timetable_year_inferred"}
    promote_inferred = (proposed.notna() & current.notna() & proposed.eq(current)
                        & events.application_close_date_source.isin(inferred_sources)
                        & events.timetable_date_source.isin(exact_sources))
    exact_fill = (fill & events.timetable_date_source.isin(exact_sources)) | promote_inferred
    fill = fill | promote_inferred
    result_for_fill = pd.to_datetime(events.loc[exact_fill, "result_date"], errors="coerce")
    close_for_fill = pd.to_datetime(events.loc[exact_fill, "application_close_date"], errors="coerce")
    events.loc[exact_fill, "cash_lock_proxy_days"] = (result_for_fill - close_for_fill).dt.days
    events.loc[exact_fill, "decision_date"] = close_for_fill.dt.strftime("%Y-%m-%d")
    official_sources = {"official_announcement", "official_prospectus_timetable",
                        "official_prospectus_pdf_timetable"}
    inferred_sources = {"official_prospectus_timetable_year_inferred",
                        "official_prospectus_pdf_timetable_year_inferred"}
    after_sources = events.application_close_date_source
    official_after = after_sources.isin(official_sources)
    inferred_after = after_sources.isin(inferred_sources)
    events = events.drop(columns=["timetable_application_close_date", "timetable_date_source",
                                  "timetable_pdf_url", "prospectus_release_date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False)
    return {
        "events": int(len(events)),
        "official_close_dates_before": int(official_before.sum()),
        "prospectus_timetable_dates_parsed": int(proposed.notna().sum()),
        "official_close_dates_added_or_promoted": int(exact_fill.sum()),
        "inferred_year_dates_retained": int(events.application_close_date_source.isin(inferred_sources).sum()),
        "official_close_dates_after": int(official_after.sum()),
        "remaining_scenario_dates": int((~(official_after | inferred_after)).sum()),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path,
                        default=Path("reports/hkipo_simulator_data_2021_2025/verified_events.csv"))
    parser.add_argument("--timetable-dates", type=Path,
                        default=Path("reports/hkipo_prospectus_timetable_dates_2021_2025.json"))
    parser.add_argument("--pdf-timetable-dates", type=Path,
                        default=Path("reports/hkipo_prospectus_timetable_pdf_extraction_2021_2025.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/hkipo_simulator_data_2021_2025/verified_events_timetable_augmented.csv"))
    args = parser.parse_args()
    print(json.dumps(apply_dates(events_path=args.events, timetable_path=args.timetable_dates,
                                 pdf_timetable_path=args.pdf_timetable_dates,
                                 output_path=args.output), indent=2))


if __name__ == "__main__":
    main()
