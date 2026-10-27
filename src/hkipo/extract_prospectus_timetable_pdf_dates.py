"""Extract application-close dates from the opening pages of HKEX IPO prospectuses."""

from __future__ import annotations

import argparse
import hashlib
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
MONTH = (r"January|February|March|April|May|June|July|August|September|"
         r"October|November|December")
WEEKDAY = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
DATE = rf"(?:\d{{1,2}}\s+(?:{MONTH})(?:\s+\d{{4}})?|(?:{MONTH})\s+\d{{1,2}},?(?:\s+\d{{4}})?)"
CLOSE_LABEL_DIRECT = re.compile(
    r"application\s+lists?(?:\s+(?:of|for)\s+(?:the\s+)?(?:Hong\s+Kong\s+)?Public\s+Offer(?:ing)?)?\s+close(?:s)?|"
    r"close of the application lists", re.I)
CLOSE_LABEL_FALLBACK = re.compile(r"last day for (?:lodging|submitting) applications", re.I)
DATE_AFTER_LABEL = re.compile(rf"(?:{WEEKDAY})?\s*,?\s*(?P<date>{DATE})", re.I)


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b",
                  lambda m: re.sub(r"\s+", "", m.group(0)), text)
    text = re.sub(r"(?<!\d)(?:\d\s+){1,}\d(?!\d)",
                  lambda m: re.sub(r"\s+", "", m.group(0)), text)
    return re.sub(r"\s+", " ", text)


def _candidate_dates(text: str, listing_date: pd.Timestamp,
                     release_date: pd.Timestamp, result_date: pd.Timestamp) -> list[dict]:
    normalized = _normalize_spacing(text)
    found: list[dict] = []
    labels = list(CLOSE_LABEL_DIRECT.finditer(normalized))
    if not labels:
        labels = list(CLOSE_LABEL_FALLBACK.finditer(normalized))
    for label in labels:
        snippet = normalized[label.start():label.start() + 320]
        match = DATE_AFTER_LABEL.search(snippet)
        if not match:
            continue
        raw = match.group("date").strip()
        inferred = not bool(re.search(r"\b\d{4}\b", raw))
        candidates: list[pd.Timestamp] = []
        if not inferred:
            parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
            if not pd.isna(parsed):
                candidates.append(parsed.normalize())
        else:
            for year in sorted({listing_date.year - 1, listing_date.year, listing_date.year + 1}):
                parsed = pd.to_datetime(f"{raw} {year}", errors="coerce", dayfirst=True)
                if (not pd.isna(parsed) and parsed < listing_date
                        and (listing_date - parsed).days <= 45):
                    candidates.append(parsed.normalize())
        for parsed in candidates:
            explicit_match = None
            if inferred:
                month_name = parsed.strftime("%B")
                day = parsed.day
                year = parsed.year
                explicit_patterns = (
                    rf"\b{month_name}\s+{day}(?:st|nd|rd|th)?[,]?\s+{year}\b",
                    rf"\b{day}\s+{month_name}\s+{year}\b",
                )
                for pattern in explicit_patterns:
                    explicit_match = re.search(pattern, normalized, flags=re.I)
                    if explicit_match:
                        break
            chronology_valid = (parsed >= release_date and parsed <= result_date
                                and parsed < listing_date
                                and (listing_date - parsed).days <= 45)
            found.append({"raw_date": raw, "application_close_date": parsed.date().isoformat(),
                          "year_inferred": inferred and explicit_match is None,
                          "year_confirmed_elsewhere_in_timetable": bool(explicit_match),
                          "year_confirmation_snippet": (normalized[max(0, explicit_match.start() - 80):
                                                                       explicit_match.end() + 80]
                                                        if explicit_match else ""),
                          "chronology_valid": bool(chronology_valid),
                          "evidence_snippet": snippet[:240]})
    # Keep unique date/evidence pairs while retaining repeated mentions for review.
    unique: dict[tuple[str, bool], dict] = {}
    for item in found:
        unique[(item["application_close_date"], item["year_inferred"])] = item
    return list(unique.values())


def _download_pdf(url: str, path: Path, *, retries: int = 3) -> tuple[int, str]:
    if path.exists() and path.stat().st_size > 8:
        with path.open("rb") as f:
            if f.read(5) == b"%PDF-":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                return path.stat().st_size, digest
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=(20, 180), stream=True,
                                    headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type:
                raise ValueError(f"Unexpected content type: {content_type}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".part")
            digest = hashlib.sha256()
            size = 0
            with temp.open("wb") as out:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            with temp.open("rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("Response body is not a PDF")
            temp.replace(path)
            return size, digest.hexdigest()
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    assert last_error is not None
    raise last_error


def extract_one(row: dict, *, raw_dir: Path, pages: int) -> dict:
    event_id = str(row["listing_event_id"])
    url = str(row["prospectus_url"])
    record = {"event_id": event_id, "company": row.get("company_name", ""),
              "listing_date": str(row["listing_date"])[:10],
              "prospectus_release_date": str(row.get("prospectus_release_date", ""))[:10],
              "prospectus_url": url, "local_pdf": str(raw_dir / f"{event_id}_prospectus.pdf"),
              "download_bytes": None, "sha256": "", "pages_scanned": 0,
              "date_candidates": [], "application_close_date": "",
              "application_close_date_source": "", "status": "not_parsed",
              "manual_review_required": False, "error": ""}
    try:
        allowed_prefix = "https://www1.hkexnews.hk/listedco/listconews/sehk/"
        if not url.startswith(allowed_prefix) or not url.lower().endswith(".pdf"):
            raise ValueError("Source is not a direct official HKEX PDF")
        path = Path(record["local_pdf"])
        record["download_bytes"], record["sha256"] = _download_pdf(url, path)
        reader = PdfReader(str(path))
        record["pages_scanned"] = min(pages, len(reader.pages))
        text = "\n".join((reader.pages[i].extract_text() or "")
                         for i in range(record["pages_scanned"]))
        listing_date = pd.Timestamp(row["listing_date"])
        release_date = pd.Timestamp(row["prospectus_release_date"])
        result_date = pd.Timestamp(row["result_date"])
        candidates = _candidate_dates(text, listing_date, release_date, result_date)
        record["date_candidates"] = candidates
        valid = [c for c in candidates if c["chronology_valid"]]
        valid_dates = sorted({c["application_close_date"] for c in valid})
        if len(valid_dates) == 1:
            selected = valid_dates[0]
            selected_rows = [c for c in valid if c["application_close_date"] == selected]
            inferred = all(c["year_inferred"] for c in selected_rows)
            record["application_close_date"] = selected
            record["application_close_date_source"] = (
                "official_prospectus_pdf_timetable_year_inferred" if inferred
                else "official_prospectus_pdf_timetable")
            record["status"] = "date_verified_in_pdf_body"
        elif len(valid_dates) > 1:
            record["status"] = "multiple_valid_dates"
            record["manual_review_required"] = True
            record["error"] = "More than one distinct valid application-close date found"
        elif candidates:
            record["status"] = "dates_failed_chronology"
            record["manual_review_required"] = True
            record["error"] = "Extracted dates did not pass prospectus/result/listing chronology checks"
        else:
            record["status"] = "no_close_date_found"
    except Exception as exc:
        record["status"] = "download_or_parse_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path,
                        default=Path("reports/hkipo_simulator_data_2021_2025/verified_events_timetable_augmented.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/processed/hkex_ipo_prospectus_manifest_2021_2025.csv"))
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("data/raw/hkex_ipo_prospectus_timetable_pdfs_2021_2025"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/hkipo_prospectus_timetable_pdf_extraction_2021_2025.json"))
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    events = pd.read_csv(args.events, dtype={"listing_event_id": str})
    missing = set(events.loc[~events.application_close_date_source.isin(
        ["official_announcement", "official_prospectus_timetable"]), "listing_event_id"].astype(str))
    manifest = pd.read_csv(args.manifest, dtype={"listing_event_id": str})
    rows = events[events.listing_event_id.isin(missing)].merge(
        manifest[["listing_event_id", "prospectus_release_date", "prospectus_url"]],
        on="listing_event_id", how="inner", validate="one_to_one")
    rows = rows[rows.prospectus_url.fillna("").str.lower().str.endswith(".pdf")]
    payload = rows.to_dict("records")
    if args.limit is not None:
        payload = payload[:args.limit]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_one, row, raw_dir=args.raw_dir, pages=args.pages)
                   for row in payload]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(f"processed {index}/{len(futures)}", flush=True)
    results.sort(key=lambda row: row["event_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"candidate_pdf_prospectuses": len(results),
               "verified_close_dates": sum(bool(row["application_close_date"]) for row in results),
               "manual_review_required": sum(bool(row["manual_review_required"]) for row in results),
               "no_close_date_found": sum(row["status"] == "no_close_date_found" for row in results),
               "download_or_parse_errors": sum(row["status"] == "download_or_parse_error" for row in results),
               "download_bytes": sum(row["download_bytes"] or 0 for row in results),
               "output": str(args.output), "raw_dir": str(args.raw_dir)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
