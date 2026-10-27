"""Download short post-listing daily-bar windows for HKEX IPOs using yfinance."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def yahoo_symbol(stock_code: object) -> str:
    digits = "".join(ch for ch in str(stock_code) if ch.isdigit())
    if not digits:
        raise ValueError(f"Invalid HKEX stock code: {stock_code!r}")
    return f"{int(digits):04d}.HK"


def download_bars(ipos: pd.DataFrame, max_sessions: int = 30,
                  pause_seconds: float = 0.25, retries: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install the optional dependency with: pip install -e ".[marketdata]"') from exc
    needed = {"stock_code", "listing_date"}
    if not needed.issubset(ipos.columns):
        raise ValueError(f"IPO file must contain {', '.join(sorted(needed))}")
    records, failures = [], []
    unique = ipos.loc[:, ["stock_code", "listing_date"]].drop_duplicates(["stock_code", "listing_date"])
    for item in unique.itertuples(index=False):
        code = str(item.stock_code).zfill(5)
        listing_date = pd.to_datetime(item.listing_date, errors="raise").normalize()
        event_id = f"{code}_{listing_date:%Y%m%d}"
        symbol = yahoo_symbol(code)
        start = listing_date.strftime("%Y-%m-%d")
        end = (listing_date + pd.offsets.BDay(max_sessions + 14) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        error_text = ""
        history = pd.DataFrame()
        for attempt in range(retries + 1):
            try:
                history = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False,
                                                   actions=False)
                if not history.empty:
                    break
                error_text = "empty history"
            except Exception as exc:  # keep batch collection resilient to ticker-level failures
                error_text = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
        if history.empty:
            failures.append({"stock_code": code, "listing_event_id": event_id, "yahoo_symbol": symbol,
                             "listing_date": listing_date, "error": error_text})
        else:
            history = history.reset_index()
            date_col = "Date" if "Date" in history else "Datetime"
            for row in history.itertuples(index=False):
                values = row._asdict()
                close = float(values["Close"])
                volume = float(values.get("Volume", 0) or 0)
                records.append({
                    "stock_code": code,
                    "listing_event_id": event_id,
                    "date": pd.Timestamp(values[date_col]).tz_localize(None).normalize()
                    if pd.Timestamp(values[date_col]).tzinfo else pd.Timestamp(values[date_col]).normalize(),
                    "open": float(values["Open"]), "high": float(values["High"]),
                    "low": float(values["Low"]), "close": close,
                    "volume": volume, "turnover": close * volume,
                    "turnover_is_estimate": True, "source": "Yahoo Finance via yfinance",
                })
        time.sleep(pause_seconds)
    bars = pd.DataFrame(records)
    if not bars.empty:
        bars = bars.sort_values(["listing_event_id", "date"]).drop_duplicates(
            ["listing_event_id", "date"], keep="last").reset_index(drop=True)
    return bars, pd.DataFrame(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipos", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--max-sessions", type=int, default=30)
    parser.add_argument("--pause-seconds", type=float, default=.25)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    bars, failures = download_bars(pd.read_csv(args.ipos, dtype={"stock_code": str}),
                                   args.max_sessions, args.pause_seconds, args.retries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    failure_path = args.failures or args.output.with_name(f"{args.output.stem}_failures.csv")
    failures.to_csv(failure_path, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(bars):,} bars for {bars.stock_code.nunique() if not bars.empty else 0:,} IPOs")
    print(f"No-data/failed tickers: {len(failures):,}; details: {failure_path}")


if __name__ == "__main__":
    main()
