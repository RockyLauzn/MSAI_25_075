"""Download yearly SPY EOD option Parquet files from the public mirror."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://raw.githubusercontent.com/anahatsingh-ui/options-dataset-hist/main/spy"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must be no later than --end-year")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for year in range(args.start_year, args.end_year + 1):
        target = args.output_dir / f"spy_options_{year}.parquet"
        if not target.exists():
            download(f"{BASE_URL}/options_{year}.parquet", target)
        print(f"{year}: {target.stat().st_size:,} bytes")
    underlying = args.output_dir / "spy_underlying_prices.parquet"
    if not underlying.exists():
        download(f"{BASE_URL}/underlying_prices.parquet", underlying)
    print(f"Underlying prices: {underlying}")


if __name__ == "__main__":
    main()
