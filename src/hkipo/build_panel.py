"""Build the HK IPO post-listing event panel from normalized CSV inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import build_event_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipos", required=True, type=Path)
    parser.add_argument("--bars", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-sessions", type=int, default=30)
    args = parser.parse_args()
    ipos = pd.read_csv(args.ipos, dtype={"stock_code": str}, parse_dates=["listing_date"])
    bars = pd.read_csv(args.bars, dtype={"stock_code": str}, parse_dates=["date"])
    panel = build_event_panel(ipos, bars, max_sessions=args.max_sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    report = {
        "ipo_records": int(len(ipos)), "events_with_prices": int(panel.listing_event_id.nunique()),
        "panel_rows": int(len(panel)), "date_start": str(panel.date.min().date()),
        "date_end": str(panel.date.max().date()),
        "ohlc_inconsistent_rows": int((~panel.ohlc_consistent).sum()),
        "ohlc_inconsistent_fraction": float((~panel.ohlc_consistent).mean()),
    }
    args.output.with_name(f"{args.output.stem}_quality.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
