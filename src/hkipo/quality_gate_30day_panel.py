"""Create a predeclared, OHLC-clean 30-session HK IPO panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ipos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    args = parser.parse_args()

    master = pd.read_csv(args.ipos, dtype={"stock_code": str}, parse_dates=["listing_date"])
    panel = pd.read_csv(args.panel, dtype={"stock_code": str}, parse_dates=["date", "listing_date"])
    if master.duplicated(["stock_code", "listing_date"]).any():
        raise ValueError("IPO master contains duplicated stock_code/listing_date events")
    if not panel.empty and not pd.to_datetime(panel.listing_date).dt.year.between(2021, 2026).all():
        raise ValueError("Primary panel may contain only 2021–2026 IPO listings")

    master["stock_code"] = master.stock_code.astype(str).str.replace(r"\.HK$", "", regex=True).str.zfill(5)
    master["listing_event_id"] = master.stock_code + "_" + master.listing_date.dt.strftime("%Y%m%d")
    if panel.empty:
        raise ValueError("Input panel is empty")
    if panel.duplicated(["listing_event_id", "date"]).any():
        raise ValueError("Panel contains duplicate event/date rows")

    grouped = {str(event): g.sort_values("date").reset_index(drop=True)
               for event, g in panel.groupby("listing_event_id", sort=False)}
    eligible, excluded = [], []
    per_year = {}
    bad_first30_rows = 0
    for row in master.itertuples(index=False):
        event_id = row.listing_event_id
        year = int(row.listing_date.year)
        g = grouped.get(event_id)
        reason = ""
        if g is None or g.empty:
            reason = "no_daily_bars"
        elif len(g) < 30:
            reason = "fewer_than_30_sessions"
        else:
            first30 = g.iloc[:30].copy()
            listed_days = pd.to_numeric(first30.listed_day, errors="coerce").tolist()
            if listed_days != list(range(1, 31)):
                raise ValueError(f"Non-consecutive listed_day values in {event_id}: {listed_days}")
            valid = first30.ohlc_consistent.astype(str).str.lower().eq("true")
            if not valid.all():
                bad_first30_rows += int((~valid).sum())
                reason = "ohlc_inconsistent_within_first_30_sessions"
            else:
                eligible.append(first30)

        stats = per_year.setdefault(str(year), {"master_events": 0, "no_daily_bars": 0,
                                                "fewer_than_30_sessions": 0,
                                                "bad_ohlc_first30": 0, "usable_events": 0})
        stats["master_events"] += 1
        if reason:
            reason_key = ("bad_ohlc_first30" if reason == "ohlc_inconsistent_within_first_30_sessions"
                          else reason)
            stats[reason_key] += 1
            excluded.append({"listing_event_id": event_id, "stock_code": row.stock_code,
                             "listing_date": row.listing_date.date().isoformat(),
                             "company_name": getattr(row, "company_name", ""),
                             "exclusion_reason": reason})
        else:
            stats["usable_events"] += 1

    clean = pd.concat(eligible, ignore_index=True) if eligible else panel.iloc[0:0].copy()
    excluded_frame = pd.DataFrame(excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.excluded.parent.mkdir(parents=True, exist_ok=True)
    args.quality_report.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    excluded_frame.to_csv(args.excluded, index=False)

    panel_event_count = int(panel.listing_event_id.nunique())
    bad_raw = ~panel.ohlc_consistent.astype(str).str.lower().eq("true")
    bad_raw_days = pd.to_numeric(panel.listed_day, errors="coerce")
    no_bars = int((excluded_frame.exclusion_reason == "no_daily_bars").sum())
    incomplete = int((excluded_frame.exclusion_reason == "fewer_than_30_sessions").sum())
    bad_events = int((excluded_frame.exclusion_reason == "ohlc_inconsistent_within_first_30_sessions").sum())
    quality = {
        "rule_version": "2026-10-27-primary-30-session-v1",
        "universe_years": [2021, 2026],
        "primary_horizon_sessions": 30,
        "rule": "Use the first 30 listed sessions only when all 30 rows have ohlc_consistent=true; exclude the whole IPO event otherwise. Do not repair or impute OHLC prices. Missing and incomplete events remain in the master-universe coverage denominator but do not enter returns.",
        "master_events": int(len(master)),
        "events_with_any_bars": panel_event_count,
        "raw_panel_rows": int(len(panel)),
        "raw_ohlc_inconsistent_rows_all_sessions": int(bad_raw.sum()),
        "raw_events_with_any_ohlc_inconsistency": int(panel.loc[bad_raw, "listing_event_id"].nunique()),
        "raw_ohlc_inconsistent_rows_sessions_1_to_30": int((bad_raw & bad_raw_days.le(30)).sum()),
        "raw_ohlc_inconsistent_rows_sessions_31_to_60": int((bad_raw & bad_raw_days.gt(30)).sum()),
        "no_daily_bars": no_bars,
        "fewer_than_30_sessions": incomplete,
        "complete_30_events_with_bad_ohlc_in_first30": bad_events,
        "bad_ohlc_rows_in_first30_among_excluded_events": bad_first30_rows,
        "usable_events": int(clean.listing_event_id.nunique()),
        "usable_rows": int(len(clean)),
        "per_year": per_year,
        "excluded_event_file": str(args.excluded),
        "clean_panel_file": str(args.output),
    }
    args.quality_report.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
