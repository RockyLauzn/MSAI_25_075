"""Download and normalize HKEX annual New Listing Reports (Main Board)."""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

URL_TEMPLATE = (
    "https://www2.hkexnews.hk/-/media/HKEXnews/Homepage/New-Listings/"
    "New-Listing-Information/New-Listing-Report/Main/NLR{year}_Eng.xlsx"
)


def _code(value: object) -> str | None:
    text = str(value).strip().replace('"', "")
    match = re.search(r"\b(\d{1,5})\b", text)
    return match.group(1).zfill(5) if match else None


def _number(value: object) -> float:
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else float("nan")


def parse_report(content: bytes, year: int, source_url: str) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(content), sheet_name=0, header=None)
    if raw.shape[1] < 10 or len(raw) < 3:
        raise ValueError(f"Unexpected HKEX report layout for {year}: {raw.shape}")
    rows = raw.iloc[2:, :10].copy()
    rows.columns = ["sequence", "stock_code", "company_name", "prospectus_date",
                    "listing_date", "sponsors", "reporting_accountants", "valuers",
                    "funds_raised_part_hkd", "ipo_subscription_price"]
    rows["stock_code"] = rows.stock_code.map(_code)
    rows = rows.dropna(subset=["stock_code", "listing_date"])
    rows["listing_date"] = pd.to_datetime(rows.listing_date, errors="coerce", dayfirst=True)
    rows["prospectus_date"] = pd.to_datetime(rows.prospectus_date, errors="coerce", dayfirst=True)
    rows["funds_raised_part_hkd"] = rows.funds_raised_part_hkd.map(_number)
    rows["ipo_price"] = rows.ipo_subscription_price.map(_number)
    rows = rows.dropna(subset=["listing_date", "ipo_price"])
    rows = rows.loc[rows.ipo_price.gt(0)]
    rows["company_name"] = rows.company_name.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # HKEX reports commonly have separate (a)/(b) rows per IPO. Collapse them,
    # retain the single subscription price, and sum the two funds-raised tranches.
    result = rows.groupby(["stock_code", "listing_date"], as_index=False, sort=False).agg(
        company_name=("company_name", "first"),
        prospectus_date=("prospectus_date", "first"),
        sponsors=("sponsors", "first"),
        ipo_price=("ipo_price", "first"),
        funds_raised_hkd=("funds_raised_part_hkd", "sum"),
    )
    result["lot_size"] = pd.NA  # Not provided by this annual report.
    result["report_year"] = year
    result["source_url"] = source_url
    return result


def download_reports(start_year: int, end_year: int, raw_dir: Path) -> pd.DataFrame:
    frames = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        url = URL_TEMPLATE.format(year=year)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-data-import"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        archive = raw_dir / f"hkex_new_listing_report_{year}.xlsx"
        archive.write_bytes(content)
        frame = parse_report(content, year, url)
        frames.append(frame)
        print(f"{year}: {len(frame):,} listing rows")
    if not frames:
        return pd.DataFrame()
    return (pd.concat(frames, ignore_index=True)
            .sort_values(["listing_date", "stock_code"])
            .drop_duplicates(["stock_code", "listing_date"], keep="last")
            .reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    result = download_reports(args.start_year, args.end_year, args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(result):,} IPOs to {args.output}")


if __name__ == "__main__":
    main()
