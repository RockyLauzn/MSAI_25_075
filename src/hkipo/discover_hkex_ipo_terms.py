"""Discover official HKEX prospectus summary/timetable PDFs for IPO terms."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
from datetime import timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse

import pandas as pd
from pypdf import PdfReader

from hkipo.discover_hkex_allotments import HKEX, _RowsParser, _get, _normalize_url, _stock_ids


class _AttachmentLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.anchor: tuple[str, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a" and attr.get("href"):
            self.anchor = (attr["href"] or "", [])

    def handle_data(self, data: str) -> None:
        if self.anchor:
            self.anchor[1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.anchor:
            href, label = self.anchor
            self.links.append((href, " ".join(" ".join(label).split())))
            self.anchor = None


def _find_prospectus(event: dict, stock_id: str) -> dict:
    listing_date = pd.Timestamp(event["listing_date"]).date()
    matches: list[dict] = []
    for days in (45, 90, 180):
        start = (listing_date - timedelta(days=days)).strftime("%Y%m%d")
        end = listing_date.strftime("%Y%m%d")
        query = urlencode({"category": "0", "lang": "EN", "market": "SEHK",
                           "stockId": stock_id, "from": start, "to": end})
        url = f"{HKEX}/search/titlesearch.xhtml?{query}"
        try:
            page = _get(url).decode("utf-8", errors="replace")
        except Exception as exc:
            return {**event, "terms_status": "search_error", "terms_error": f"{type(exc).__name__}: {exc}"}
        parser = _RowsParser()
        parser.feed(page)
        for row in parser.rows:
            text = row["text"].upper()
            if "LISTING DOCUMENTS" not in text or "GLOBAL OFFERING" not in text:
                continue
            release_match = re.search(r"Release Time:\s*(\d{2}/\d{2}/\d{4})", row["text"], re.I)
            if not release_match:
                continue
            release_date = pd.to_datetime(release_match.group(1), dayfirst=True)
            if release_date.date() > listing_date:
                continue
            links = [(href, label) for href, label in row["links"]
                     if re.search(r"\.(?:htm|pdf)(?:$|\?)", href, re.I)]
            if not links:
                continue
            # The multi-file HTML page yields compact official summary and
            # timetable PDFs; the standalone PDF is the fallback.
            links.sort(key=lambda item: (".htm" not in item[0].lower(),
                                         "multi" not in item[1].lower()))
            matches.append({"release_date": release_date, "delta": abs((listing_date-release_date.date()).days),
                            "prospectus_url": _normalize_url(links[0][0]),
                            "prospectus_title": links[0][1]})
        if matches:
            break
    if not matches:
        return {**event, "terms_status": "not_found", "terms_error": "No official Global Offering listing document found"}
    best = min(matches, key=lambda item: item["delta"])
    return {**event, "prospectus_release_date": best["release_date"].strftime("%Y-%m-%d"),
            "prospectus_url": best["prospectus_url"], "prospectus_title": best["prospectus_title"],
            "terms_status": "found", "terms_error": ""}


def _extract_terms(row: dict, raw_dir: Path) -> dict:
    event_id = str(row["listing_event_id"])
    url = str(row["prospectus_url"])
    result = {**row, "summary_pdf_url": "", "timetable_pdf_url": "",
              "max_application_price_hkd": pd.NA, "application_close_date": "",
              "terms_status": "not_parsed", "terms_error": ""}
    try:
        page = _get(url)
        if page.startswith(b"%PDF"):
            return {**result, "terms_status": "pdf_only", "terms_error": "Standalone prospectus needs attachment discovery"}
        attachments = _AttachmentLinks()
        attachments.feed(page.decode("utf-8", errors="replace"))
        found: dict[str, tuple[str, str]] = {}
        for href, label in attachments.links:
            label_lower = label.casefold()
            if "summary" in label_lower:
                found["summary"] = (urljoin(url, href), label)
            elif "timetable" in label_lower:
                found["timetable"] = (urljoin(url, href), label)
        summary_text = ""
        if "summary" in found:
            summary_url = found["summary"][0]
            result["summary_pdf_url"] = summary_url
            content = _get(summary_url)
            path = raw_dir / f"{event_id}_summary.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            summary_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        if "timetable" in found:
            timetable_url = found["timetable"][0]
            result["timetable_pdf_url"] = timetable_url
            content = _get(timetable_url)
            path = raw_dir / f"{event_id}_timetable.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            timetable_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        else:
            timetable_text = ""

        maximum_match = re.search(
            r"Maximum\s+Offer\s+Price.{0,100}?(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)",
            re.sub(r"\s+", " ", summary_text), re.I)
        if maximum_match:
            result["max_application_price_hkd"] = float(maximum_match.group(1).replace(",", ""))
        normalized_timetable = re.sub(
            r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b",
            lambda match: re.sub(r"\s+", "", match.group(0)), timetable_text)
        normalized_timetable = re.sub(
            r"(?<!\d)(?:\d\s+){1,}\d(?!\d)",
            lambda match: re.sub(r"\s+", "", match.group(0)), normalized_timetable)
        month_name = (r"January|February|March|April|May|June|July|August|September|"
                      r"October|November|December")
        close_date_match = re.search(
            r"(?:application\s+lists?(?:\s+of\s+the\s+Hong\s+Kong\s+Public\s+Offering)?\s+close|close of the application lists|"
            r"last day for (?:lodging|submitting) applications)"
            r".{0,180}?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\s*,?\s*"
            rf"(?P<date>\d{{1,2}}\s+(?:{month_name})(?:\s+\d{{4}})?|"
            rf"(?:{month_name})\s+\d{{1,2}},?(?:\s+\d{{4}})?)",
            re.sub(r"\s+", " ", normalized_timetable), re.I)
        if close_date_match:
            raw_date = close_date_match.group("date").strip()
            parsed = pd.to_datetime(raw_date, errors="coerce", dayfirst=True)
            if pd.isna(parsed) and not re.search(r"\b\d{4}\b", raw_date):
                listing_date = pd.to_datetime(row.get("listing_date"), errors="coerce")
                if not pd.isna(listing_date):
                    candidates = []
                    for year in (listing_date.year - 1, listing_date.year, listing_date.year + 1):
                        candidate = pd.to_datetime(f"{raw_date} {year}", errors="coerce", dayfirst=True)
                        if (not pd.isna(candidate) and candidate < listing_date
                                and (listing_date - candidate).days <= 45):
                            candidates.append(candidate)
                    if candidates:
                        parsed = max(candidates)
            if not pd.isna(parsed):
                result["application_close_date"] = parsed.date().isoformat()
                result["application_close_date_source"] = (
                    "official_prospectus_timetable_year_inferred"
                    if not re.search(r"\b\d{4}\b", raw_date)
                    else "official_prospectus_timetable")
        result["terms_status"] = "parsed" if pd.notna(result["max_application_price_hkd"]) else "price_missing"
        if not result["application_close_date"]:
            result["terms_error"] = "Maximum price parsed; application close date not recognized"
        elif result["terms_status"] != "parsed":
            result["terms_error"] = "Application close date parsed; maximum price not recognized"
        return result
    except Exception as exc:
        return {**result, "terms_status": "download_or_parse_error",
                "terms_error": f"{type(exc).__name__}: {exc}"}


def discover(events: pd.DataFrame, *, raw_dir: Path, workers: int = 4,
             limit: int | None = None, download_terms: bool = True) -> pd.DataFrame:
    stock_ids = _stock_ids()
    selected = events.copy()
    if limit is not None:
        selected = selected.head(limit)
    jobs = []
    for event in selected.to_dict("records"):
        code = str(event["stock_code"]).zfill(5)
        for stock_id in stock_ids.get(code, []):
            jobs.append((event, stock_id))
    found: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_find_prospectus, event, stock_id) for event, stock_id in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            if row.get("terms_status") == "found":
                found.append(row)
    if not found:
        return pd.DataFrame()
    # Multiple historical stock IDs can point to the same event. Keep the most
    # recently released offering document before downloading attachments.
    found_frame = pd.DataFrame(found).sort_values("prospectus_release_date")
    found_frame = found_frame.drop_duplicates("listing_event_id", keep="last")
    if not download_terms:
        return found_frame.sort_values("listing_event_id").reset_index(drop=True)
    rows = found_frame.to_dict("records")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        parsed = list(pool.map(lambda row: _extract_terms(row, raw_dir), rows))
    return pd.DataFrame(parsed).sort_values("listing_event_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path,
                        default=Path("reports/hkipo_simulator_data_2021_2025/verified_events.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hkex_ipo_terms_2021_2025"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/hkex_ipo_terms_2021_2025.csv"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--discover-only", action="store_true",
                        help="Save prospectus search results without downloading attachments")
    args = parser.parse_args()
    events = pd.read_csv(args.events, dtype={"listing_event_id": str, "stock_code": str},
                         parse_dates=["listing_date"])
    result = discover(events, raw_dir=args.raw_dir, workers=args.workers, limit=args.limit,
                      download_terms=not args.discover_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.terms_status.value_counts(dropna=False).to_dict() if len(result) and "terms_status" in result
          else {"found_prospectus": len(result)})


if __name__ == "__main__":
    main()
