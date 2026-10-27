"""Follow HKEX HTML prospectus indexes to their summary PDFs and extract offer prices."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)
MAX_PRICE = re.compile(
    r"m\s*a\s*x\s*i\s*m\s*u\s*m\s*(?:p\s*u\s*b\s*l\s*i\s*c\s*)?"
    r"o\s*f\s*f\s*e\s*r\s*p\s*r\s*i\s*c\s*e[\s\S]{0,500}?(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)", re.I)
NOT_MORE_THAN = re.compile(
    r"(?:offer|public\s+offer)\s+price[\s\S]{0,150}?not\s+more\s+than"
    r"[\s\S]{0,80}?(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)", re.I)
PRICE_RANGE = re.compile(
    r"offer\s+price\s*:?\s*(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)\s*"
    r"t\s*o\s*(?:HKD|HK\$)?\s*([\d,]+(?:\.\d+)?)", re.I)
FIXED_PRICE = re.compile(
    r"offer\s+price(?:\s+will\s+be)?\s*:?\s*(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)", re.I)
COMPACT_MAX = re.compile(
    r"maximum(?:public|indicative)?offerprice.{0,500}?HK\$([\d,]+(?:\.\d+)?)", re.I)
COMPACT_NOT_MORE = re.compile(
    r"(?:offer|publicoffer)price.{0,150}?notmorethan.{0,80}?HK\$([\d,]+(?:\.\d+)?)", re.I)
COMPACT_RANGE = re.compile(
    r"offerprice(?:rangeof|of)?HK\$([\d,]+(?:\.\d+)?)toHK\$?([\d,]+(?:\.\d+)?)", re.I)
COMPACT_FIXED = re.compile(
    r"offerprice(?:willbe|of)?HK\$([\d,]+(?:\.\d+)?)", re.I)
ANCHOR = re.compile(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)


def _read_pdf(url: str, target: Path) -> bytes:
    if target.exists() and target.stat().st_size > 8:
        with target.open("rb") as stream:
            if stream.read(5) == b"%PDF-":
                return target.read_bytes()
    last_error = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(25, 120), headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            if "pdf" not in response.headers.get("content-type", "").lower() or not response.content.startswith(b"%PDF-"):
                raise ValueError("The HKEX link did not return a PDF")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            return response.content
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
    raise last_error or RuntimeError("Failed to retrieve PDF")


def _prices(text: str) -> tuple[list[float], str]:
    compact = re.sub(r"(?<=[A-Za-z])\s+(?=[A-Za-z])", "", text)
    patterns = ((COMPACT_MAX, 1, "maximum_offer_price_text"),
                (COMPACT_NOT_MORE, 1, "offer_price_not_more_than"),
                (COMPACT_RANGE, 2, "offer_price_range_upper_bound"),
                (COMPACT_FIXED, 1, "fixed_offer_price"))
    for pattern, group, method in patterns:
        values = []
        for match in pattern.finditer(compact):
            value = float(match.group(group).replace(",", ""))
            if value not in values:
                values.append(value)
        if values:
            return values, method
    return [], ""


def extract_one(row: dict, raw_dir: Path) -> dict:
    event_id = str(row["listing_event_id"])
    prospectus_url = str(row["prospectus_url"])
    result = {"event_id": event_id, "company": row.get("company_name", ""),
              "final_price_hkd": float(row["ipo_price_hkd"]), "expected_max_price_hkd": None,
              "source_url": prospectus_url, "local_pdf": "", "pdf_body_match": False,
              "pdf_text_max_prices_hkd": [], "extraction_method": "", "error": ""}
    try:
        html_response = requests.get(prospectus_url, timeout=(20, 45),
                                     headers={"User-Agent": "Mozilla/5.0"})
        html_response.raise_for_status()
        page_text = html_response.content.decode("utf-8", errors="replace")
        links = []
        for href, label in ANCHOR.findall(page_text):
            label = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(label))).strip().upper()
            if label == "SUMMARY":
                links.append((0, urljoin(prospectus_url, href), label))
            elif label == "HOW TO APPLY FOR HONG KONG OFFER SHARES":
                links.append((1, urljoin(prospectus_url, href), label))
        if not links:
            raise ValueError("No Summary or How-to-Apply PDF found on the HKEX index page")
        links.sort()
        values = []
        for _rank, pdf_url, label in links:
            target = raw_dir / f"{event_id}_{label.lower().replace(' ', '_')}.pdf"
            content = _read_pdf(pdf_url, target)
            result["local_pdf"] = str(target)
            reader = PdfReader(str(target), strict=False)
            text = re.sub(r"\s+", " ", " ".join(page.extract_text() or "" for page in reader.pages))
            values, method = _prices(text)
            if values:
                result["source_url"] = pdf_url
                result["extraction_method"] = method
                result["pdf_text_max_prices_hkd"] = values
                break
        eligible = [value for value in values if value >= result["final_price_hkd"] - 0.011]
        if eligible:
            result["expected_max_price_hkd"] = max(eligible)
            result["pdf_body_match"] = True
        else:
            result["error"] = "No maximum, range upper bound or fixed offer price at or above final price in section PDFs"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=Path, default=Path("data/processed/hkex_ipo_cash_terms_2021_2025.csv"))
    parser.add_argument("--events", type=Path, default=Path("reports/hkipo_simulator_data_2021_2025/verified_events.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/hkex_ipo_prospectus_manifest_2021_2025.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hkex_ipo_html_prospectus_sections_2021_2025"))
    parser.add_argument("--output", type=Path, default=Path("reports/hkipo_html_prospectus_price_extraction_2021_2025.json"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    terms = pd.read_csv(args.terms, dtype={"listing_event_id": str})
    missing = set(terms.loc[~terms.term_price_valid, "listing_event_id"].astype(str))
    events = pd.read_csv(args.events, dtype={"listing_event_id": str})
    manifest = pd.read_csv(args.manifest, dtype={"listing_event_id": str})
    rows = events[events.listing_event_id.isin(missing)].merge(
        manifest[["listing_event_id", "prospectus_url"]], on="listing_event_id", how="left")
    rows = rows[rows.prospectus_url.fillna("").str.lower().str.endswith(".htm")]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_one, row, args.raw_dir) for row in rows.to_dict("records")]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print(f"processed {index}/{len(futures)}", flush=True)
    results.sort(key=lambda row: row["event_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"html_prospectuses": len(results),
                      "body_matches": sum(row["pdf_body_match"] for row in results),
                      "unresolved": sum(not row["pdf_body_match"] for row in results),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
