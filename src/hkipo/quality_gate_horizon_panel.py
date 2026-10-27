"""Build an event-complete, OHLC-consistent panel for one declared IPO horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ipos", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=[20, 40, 60], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    args = parser.parse_args()
    h = args.horizon

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
    per_year: dict[str, dict[str, int]] = {}
    bad_rows_in_excluded_events = 0
    for row in master.itertuples(index=False):
        event_id = row.listing_event_id
        year = str(int(row.listing_date.year))
        g = grouped.get(event_id)
        reason = ""
        if g is None or g.empty:
            reason = "no_daily_bars"
        elif len(g) < h:
            reason = f"fewer_than_{h}_sessions"
        else:
            episode = g.iloc[:h].copy()
            listed_days = pd.to_numeric(episode.listed_day, errors="coerce").tolist()
            if listed_days != list(range(1, h + 1)):
                raise ValueError(f"Non-consecutive listed_day values in {event_id}: {listed_days}")
            valid = episode.ohlc_consistent.astype(str).str.lower().eq("true")
            if not valid.all():
                bad_rows_in_excluded_events += int((~valid).sum())
                reason = f"ohlc_inconsistent_within_first_{h}_sessions"
            else:
                eligible.append(episode)

        stats = per_year.setdefault(year, {"master_events": 0, "no_daily_bars": 0,
                                            f"fewer_than_{h}_sessions": 0,
                                            f"bad_ohlc_first_{h}": 0, "usable_events": 0})
        stats["master_events"] += 1
        if reason:
            if reason.startswith("ohlc_inconsistent"):
                key = f"bad_ohlc_first_{h}"
            else:
                key = reason
            stats[key] += 1
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

    bad_raw = ~panel.ohlc_consistent.astype(str).str.lower().eq("true")
    listed_day = pd.to_numeric(panel.listed_day, errors="coerce")
    no_bars = int((excluded_frame.exclusion_reason == "no_daily_bars").sum())
    incomplete = int((excluded_frame.exclusion_reason == f"fewer_than_{h}_sessions").sum())
    bad_event_count = int(excluded_frame.exclusion_reason.str.startswith("ohlc_inconsistent").sum())
    quality = {
        "rule_version": f"2026-10-27-horizon-{h}-session-v1",
        "universe_years": [2021, 2026],
        "horizon_sessions": h,
        "rule": f"Use exactly the first {h} listed sessions only when all {h} rows exist consecutively and have ohlc_consistent=true; exclude the entire IPO event otherwise. Never repair OHLC values or impute missing prices.",
        "master_events": int(len(master)),
        "raw_events_with_any_bars": int(panel.listing_event_id.nunique()),
        "raw_panel_rows": int(len(panel)),
        "raw_ohlc_inconsistent_rows_all_sessions": int(bad_raw.sum()),
        "raw_ohlc_inconsistent_rows_within_horizon": int((bad_raw & listed_day.le(h)).sum()),
        "raw_ohlc_inconsistent_rows_after_horizon": int((bad_raw & listed_day.gt(h)).sum()),
        "no_daily_bars": no_bars,
        f"fewer_than_{h}_sessions": incomplete,
        f"events_with_bad_ohlc_in_first_{h}": bad_event_count,
        f"bad_ohlc_rows_in_excluded_events_first_{h}": bad_rows_in_excluded_events,
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
