"""Extract maximum application prices from prospectus PDFs for missing IPO terms."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)
MAX_PRICE = re.compile(
    r"m\s*a\s*x\s*i\s*m\s*u\s*m\s*(?:p\s*u\s*b\s*l\s*i\s*c\s*)?"
    r"o\s*f\s*f\s*e\s*r\s*p\s*r\s*i\s*c\s*e"
    r"[\s\S]{0,500}?(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)", re.I)
NOT_MORE_THAN = re.compile(
    r"(?:offer|public\s+offer)\s+price[\s\S]{0,150}?not\s+more\s+than"
    r"[\s\S]{0,80}?(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)", re.I)
PRICE_RANGE = re.compile(
    r"offer\s+price\s*:?\s*(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)\s*"
    r"t\s*o\s*(?:HKD|HK\$)?\s*([\d,]+(?:\.\d+)?)", re.I)
FIXED_PRICE = re.compile(
    r"offer\s+price(?:\s+will\s+be)?\s*:?\s*(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)", re.I)


def extract_one(row: dict, raw_dir: Path, pages: int) -> dict:
    event_id = str(row["listing_event_id"])
    url = str(row["prospectus_url"])
    path = raw_dir / f"{event_id}_prospectus.pdf"
    result = {"event_id": event_id, "company": row.get("company_name", ""),
              "final_price_hkd": float(row["ipo_price_hkd"]),
              "expected_max_price_hkd": None, "source_url": url,
              "local_pdf": str(path), "http_status": None, "content_type": "",
              "file_size_bytes": None, "pdf_text_max_prices_hkd": [],
              "pdf_body_match": False, "error": ""}
    try:
        if not url.startswith("https://www1.hkexnews.hk/listedco/listconews/sehk/") or not url.lower().endswith(".pdf"):
            raise ValueError("Prospectus URL is not a direct HKEX PDF")
        signature = b""
        if path.exists() and path.stat().st_size >= 8:
            with path.open("rb") as stream:
                signature = stream.read(5)
        if signature != b"%PDF-":
            last_error = None
            for attempt in range(4):
                try:
                    response = requests.get(url, timeout=(25, 180), stream=True,
                                            headers={"User-Agent": "Mozilla/5.0"})
                    result["http_status"] = response.status_code
                    result["content_type"] = response.headers.get("content-type", "")
                    response.raise_for_status()
                    if "pdf" not in result["content_type"].lower():
                        raise ValueError(f"Unexpected content type: {result['content_type']}")
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix(".pdf.part")
                    with temp.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                    with temp.open("rb") as stream:
                        signature = stream.read(5)
                    if signature != b"%PDF-":
                        temp.unlink(missing_ok=True)
                        raise ValueError("Response body does not have a PDF signature")
                    temp.replace(path)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        time.sleep((attempt + 1) * 2)
            if last_error:
                raise last_error
        else:
            result["http_status"] = 200
            result["content_type"] = "application/pdf (cached)"
        result["file_size_bytes"] = path.stat().st_size
        reader = PdfReader(str(path), strict=False)
        text = re.sub(r"\s+", " ", " ".join((page.extract_text() or "") for page in reader.pages[:pages]))
        values = []
        for match in MAX_PRICE.finditer(text):
            value = float(match.group(1).replace(",", ""))
            if value not in values:
                values.append(value)
        method = "maximum_offer_price_text"
        if not values:
            for match in NOT_MORE_THAN.finditer(text):
                value = float(match.group(1).replace(",", ""))
                if value not in values:
                    values.append(value)
            if values:
                method = "offer_price_not_more_than"
        if not values:
            for match in PRICE_RANGE.finditer(text):
                value = float(match.group(2).replace(",", ""))
                if value not in values:
                    values.append(value)
            if values:
                method = "offer_price_range_upper_bound"
        if not values:
            for match in FIXED_PRICE.finditer(text):
                value = float(match.group(1).replace(",", ""))
                if value not in values:
                    values.append(value)
            if values:
                method = "fixed_offer_price"
        result["pdf_text_max_prices_hkd"] = values
        # When a price range is disclosed, the stated maximum is the highest listed value.
        eligible = [value for value in values if value >= result["final_price_hkd"] - 0.011]
        if eligible:
            result["expected_max_price_hkd"] = max(eligible)
            result["pdf_body_match"] = True
            result["extraction_method"] = method
            result["page_count"] = len(reader.pages)
        else:
            result["error"] = "No extracted maximum offer price at or above final issue price"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=Path, default=Path("data/processed/hkex_ipo_cash_terms_2021_2025.csv"))
    parser.add_argument("--events", type=Path, default=Path("reports/hkipo_simulator_data_2021_2025/verified_events.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/hkex_ipo_prospectus_manifest_2021_2025.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hkex_ipo_missing_cash_term_prospectuses_2021_2025"))
    parser.add_argument("--output", type=Path, default=Path("reports/hkipo_missing_cash_term_prospectus_extraction_2021_2025.json"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pages", type=int, default=12)
    args = parser.parse_args()

    terms = pd.read_csv(args.terms, dtype={"listing_event_id": str})
    missing = set(terms.loc[~terms.term_price_valid, "listing_event_id"].astype(str))
    events = pd.read_csv(args.events, dtype={"listing_event_id": str})
    manifest = pd.read_csv(args.manifest, dtype={"listing_event_id": str})
    rows = events[events.listing_event_id.isin(missing)].merge(
        manifest[["listing_event_id", "prospectus_url", "prospectus_release_date"]],
        on="listing_event_id", how="left", validate="one_to_one")
    rows = rows[rows.prospectus_url.fillna("").str.lower().str.endswith(".pdf")]
    payload = rows.to_dict("records")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_one, row, args.raw_dir, args.pages) for row in payload]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(f"processed {index}/{len(futures)}", flush=True)
    results.sort(key=lambda row: row["event_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"candidate_prospectuses": len(results),
               "body_matches": sum(row["pdf_body_match"] for row in results),
               "missing_or_unparsed": sum(not row["pdf_body_match"] for row in results),
               "total_bytes": sum(row["file_size_bytes"] or 0 for row in results),
               "output": str(args.output), "raw_directory": str(args.raw_dir)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
