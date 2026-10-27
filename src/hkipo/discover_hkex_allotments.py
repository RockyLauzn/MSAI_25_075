"""Discover official HKEX allotment-result PDFs for IPO events by date/code."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
from datetime import timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

HKEX = "https://www1.hkexnews.hk"
EDS = f"{HKEX}/ncms/script/eds"
USER_AGENT = "Mozilla/5.0 (compatible; HKIPOResearch/0.1)"


def _get(url: str, timeout: int = 30) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


class _RowsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self._row_depth = 0
        self._row_text: list[str] = []
        self._links: list[tuple[str, str]] = []
        self._anchor: tuple[str, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "tr":
            self._row_depth += 1
            if self._row_depth == 1:
                self._row_text, self._links = [], []
        elif self._row_depth and tag == "a" and attr.get("href"):
            self._anchor = (attr["href"] or "", [])

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor:
            href, label = self._anchor
            self._links.append((href, " ".join(label).strip()))
            self._anchor = None
        elif tag == "tr" and self._row_depth:
            self._row_depth -= 1
            if self._row_depth == 0:
                self.rows.append({"text": " ".join(" ".join(self._row_text).split()),
                                  "links": self._links})

    def handle_data(self, data: str) -> None:
        if self._row_depth:
            self._row_text.append(data)
            if self._anchor:
                self._anchor[1].append(data)


def _stock_ids() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for filename in ("activestock_sehk_e.json", "inactivestock_sehk_e.json"):
        rows = json.loads(_get(f"{EDS}/{filename}"))
        for row in rows:
            code, internal_id = str(row.get("c", "")).zfill(5), str(row.get("i", ""))
            if code.isdigit() and internal_id:
                mapping.setdefault(code, [])
                if internal_id not in mapping[code]:
                    mapping[code].append(internal_id)
    return mapping


def _normalize_url(href: str) -> str:
    if href.startswith("http"):
        return href
    return f"{HKEX}/{href.lstrip('/')}"


def _search_one(event: dict, stock_ids: list[str]) -> dict:
    listing_date = pd.Timestamp(event["listing_date"]).date()
    start = (listing_date - timedelta(days=14)).strftime("%Y%m%d")
    end = (listing_date + timedelta(days=2)).strftime("%Y%m%d")
    matches = []
    for internal_id in stock_ids:
        query = urlencode({"category": "0", "lang": "EN", "market": "SEHK",
                           "stockId": internal_id, "from": start, "to": end})
        url = f"{HKEX}/search/titlesearch.xhtml?{query}"
        try:
            html = _get(url).decode("utf-8", errors="replace")
        except Exception as exc:  # retain failures for audit/retry
            return {**event, "source_status": "request_error", "error": str(exc)}
        parser = _RowsParser()
        parser.feed(html)
        for row in parser.rows:
            if "ALLOTMENT RESULTS" not in row["text"].upper():
                continue
            date_match = re.search(r"Release Time:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})", row["text"], re.I)
            document_link = next((href for href, _ in row["links"]
                                  if re.search(r"\.(?:pdf|htm)(?:$|\?)", href, re.I)), None)
            if not document_link:
                continue
            release = pd.to_datetime(date_match.group(1), dayfirst=True) if date_match else pd.NaT
            days_delta = (release.date() - listing_date).days if not pd.isna(release) else 999
            matches.append({"release": release, "delta": abs(days_delta),
                            "url": _normalize_url(document_link),
                            "title": next((label for href, label in row["links"]
                                           if href == document_link), "")})
    if not matches:
        return {**event, "source_status": "not_found"}
    best = min(matches, key=lambda item: item["delta"])
    return {**event,
            "announcement_release_time_hkt": best["release"].strftime("%Y-%m-%d %H:%M") if not pd.isna(best["release"]) else "",
            "announcement_url": best["url"], "announcement_title": best["title"],
            "release_listing_day_delta": (best["release"].date() - listing_date).days if not pd.isna(best["release"]) else None,
            "source_status": "found" if best["delta"] <= 3 else "found_outside_window"}


def discover(ipos: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
    required = {"stock_code", "listing_date"}
    missing = required.difference(ipos.columns)
    if missing:
        raise ValueError(f"IPO master missing columns: {', '.join(sorted(missing))}")
    mapping = _stock_ids()
    events = []
    for row in ipos.to_dict("records"):
        code = re.sub(r"\.0$", "", str(row["stock_code"]).strip()).zfill(5)
        event = {**row, "stock_code": code}
        events.append((event, mapping.get(code, [])))
    out: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_search_one, event, ids) for event, ids in events]
        for future in concurrent.futures.as_completed(futures):
            out.append(future.result())
    return pd.DataFrame(out).sort_values(["listing_date", "stock_code"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    ipos = pd.read_csv(args.ipos, dtype={"stock_code": str}, parse_dates=["listing_date"])
    if args.year is not None:
        ipos = ipos.loc[ipos.listing_date.dt.year.eq(args.year)].copy()
    result = discover(ipos, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.source_status.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
