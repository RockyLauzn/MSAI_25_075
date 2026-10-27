"""Download SPY daily adjusted-close history for research dataset construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def download(start: str, output: Path, end: str | None = None) -> int:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("Install the optional Yahoo downloader with: pip install -e '.[yahoo]'") from exc

    prices = yf.download(
        "SPY",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        multi_level_index=False,
    )
    if prices is None or prices.empty or "Adj Close" not in prices.columns:
        raise RuntimeError("Yahoo Finance returned no usable SPY adjusted-close history")

    output_frame = pd.DataFrame({
        "date": pd.to_datetime(prices.index).date,
        "adj_close": pd.to_numeric(prices["Adj Close"], errors="raise").to_numpy(),
    }).dropna()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output, index=False)
    return len(output_frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument("--end", help="Exclusive end date, YYYY-MM-DD (defaults to latest)")
    parser.add_argument("--output", type=Path, default=Path("data/raw/spy_daily.csv"))
    args = parser.parse_args()
    row_count = download(args.start, args.output, args.end)
    print(f"Wrote {row_count:,} daily rows to {args.output}")


if __name__ == "__main__":
    main()
