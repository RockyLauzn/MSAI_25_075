"""Download HKEX search-index PDF sources and verify extracted maximum prices."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)

MAX_PRICE = re.compile(
    r"m\s*a\s*x\s*i\s*m\s*u\s*m\s*o\s*f\s*f\s*e\s*r\s*p\s*r\s*i\s*c\s*e[\s\S]{0,400}?(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)",
    re.I,
)
ALT_PRICE = re.compile(
    r"(?:not\s+more\s+than|offer\s+price\s*:)\s*(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)",
    re.I,
)


def verify_one(item: dict, raw_dir: Path, pages_to_read: int) -> dict:
    event_id = str(item["event_id"])
    url = str(item.get("url", ""))
    expected = float(item["max_price"])
    target = raw_dir / f"{event_id}.pdf"
    result = {"event_id": event_id, "expected_max_price_hkd": expected,
              "source_url": url, "local_pdf": str(target), "http_status": None,
              "content_type": "", "file_size_bytes": None, "sha256": None,
              "pdf_text_max_prices_hkd": [], "pdf_body_match": False, "error": ""}
    try:
        if not url.startswith("https://www1.hkexnews.hk/listedco/listconews/sehk/"):
            raise ValueError("Source URL is not an HKEX listedco PDF URL")
        signature = b""
        if target.exists() and target.stat().st_size >= 8:
            with target.open("rb") as stream:
                signature = stream.read(5)
        if signature != b"%PDF-":
            response = requests.get(url, timeout=(15, 120), stream=True,
                                    headers={"User-Agent": "Mozilla/5.0"})
            result["http_status"] = response.status_code
            result["content_type"] = response.headers.get("content-type", "")
            response.raise_for_status()
            if "pdf" not in result["content_type"].lower():
                raise ValueError(f"Unexpected content type: {result['content_type']}")
            raw_dir.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".pdf.part")
            digest = hashlib.sha256()
            with temp.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        digest.update(chunk)
            with temp.open("rb") as stream:
                temp_signature = stream.read(5)
            if temp_signature != b"%PDF-":
                temp.unlink(missing_ok=True)
                raise ValueError("Response body does not have a PDF signature")
            temp.replace(target)
            result["sha256"] = digest.hexdigest()
        else:
            result["http_status"] = 200
            result["content_type"] = "application/pdf (cached)"
        result["file_size_bytes"] = target.stat().st_size
        if result["sha256"] is None:
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result["sha256"] = digest.hexdigest()

        reader = PdfReader(str(target), strict=False)
        text = " ".join((page.extract_text() or "") for page in reader.pages[:pages_to_read])
        text = re.sub(r"\s+", " ", text)
        values = []
        for pattern in (MAX_PRICE, ALT_PRICE):
            for match in pattern.finditer(text):
                value = float(match.group(1).replace(",", ""))
                if value not in values:
                    values.append(value)
        result["pdf_text_max_prices_hkd"] = values
        result["pdf_body_match"] = any(abs(value - expected) <= 0.005 for value in values)
        result["page_count"] = len(reader.pages)
        if not result["pdf_body_match"]:
            result["error"] = "Expected maximum price not found in extracted PDF text"
    except Exception as exc:  # record per-file problems and continue the audit
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("reports/hkipo_web_search_cash_terms_2021_2025.json"))
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("data/raw/hkex_ipo_indexed_cash_term_pdfs_2021_2025"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/hkipo_indexed_cash_term_pdf_verification_2021_2025.json"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pages-to-read", type=int, default=12)
    args = parser.parse_args()
    items = [row for row in json.loads(args.input.read_text(encoding="utf-8"))
             if row.get("valid") and row.get("max_price") is not None and row.get("url")]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(verify_one, item, args.raw_dir, args.pages_to_read) for item in items]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(f"verified {index}/{len(futures)}", flush=True)
    results.sort(key=lambda row: row["event_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"pdfs": len(results),
               "body_matches": sum(row["pdf_body_match"] for row in results),
               "body_mismatches_or_unparsed": sum(not row["pdf_body_match"] for row in results),
               "total_downloaded_bytes": sum(row["file_size_bytes"] or 0 for row in results),
               "errors": sum(bool(row["error"]) for row in results),
               "output": str(args.output), "raw_directory": str(args.raw_dir)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
