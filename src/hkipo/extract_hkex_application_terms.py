"""Find HKEX white application forms and extract the maximum application price."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from pypdf import PdfReader

from hkipo.discover_hkex_allotments import EDS, HKEX, _RowsParser, _get, _normalize_url

logging.getLogger("pypdf").setLevel(logging.ERROR)


def _fetch(url: str, *, timeout: int = 45) -> bytes:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0 HKIPOResearch/0.1"})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _stock_id_map(cache_path: Path) -> dict[str, list[str]]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    mapping: dict[str, list[str]] = {}
    for filename in ("activestock_sehk_e.json", "inactivestock_sehk_e.json"):
        rows = json.loads(_fetch(f"{EDS}/{filename}", timeout=60))
        for row in rows:
            code, internal_id = str(row.get("c", "")).zfill(5), str(row.get("i", ""))
            if code.isdigit() and internal_id:
                mapping.setdefault(code, [])
                if internal_id not in mapping[code]:
                    mapping[code].append(internal_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def _find_form(row: dict, ids: list[str]) -> dict:
    target_date = pd.Timestamp(row["prospectus_release_date"]).strftime("%Y%m%d")
    for stock_id in ids:
        query = urlencode({"category": "0", "lang": "EN", "market": "SEHK",
                           "stockId": stock_id, "from": target_date, "to": target_date})
        try:
            page = _fetch(f"{HKEX}/search/titlesearch.xhtml?{query}").decode("utf-8", errors="replace")
        except Exception as exc:
            return {**row, "application_form_url": "", "form_status": "search_error",
                    "form_error": f"{type(exc).__name__}: {exc}"}
        parser = _RowsParser()
        parser.feed(page)
        candidates = []
        for result in parser.rows:
            text = result["text"].upper()
            if "LISTING DOCUMENTS" not in text or "APPLICATION FORM" not in text:
                continue
            for href, label in result["links"]:
                if re.search(r"\.pdf(?:$|\?)", href, re.I):
                    candidates.append(("WHITE APPLICATION FORM" not in text, href, label))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return {**row, "application_form_url": _normalize_url(candidates[0][1]),
                    "application_form_title": candidates[0][2], "form_status": "found", "form_error": ""}
    return {**row, "application_form_url": "", "form_status": "not_found",
            "form_error": "No application form found on prospectus release date"}


def _parse_form(row: dict, raw_dir: Path) -> dict:
    if row.get("form_status") != "found":
        return row
    try:
        content = _fetch(row["application_form_url"], timeout=60)
        if not content.startswith(b"%PDF"):
            raise ValueError("Application form link did not return a PDF")
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{row['listing_event_id']}_white_application_form.pdf"
        path.write_bytes(content)
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages[:6])
        flat = re.sub(r"\s+", " ", text)
        max_match = re.search(
            r"(?:Maximum\s+)?Offer\s+Price\s*:\s*(?:Not\s+more\s+than\s*)?"
            r"(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)", flat, re.I)
        fee_matches = {
            "brokerage_rate": re.search(r"brokerage of\s*([\d.]+)\s*%", flat, re.I),
            "sfc_levy_rate": re.search(r"SFC transaction levy of\s*([\d.]+)\s*%", flat, re.I),
            "exchange_trading_fee_rate": re.search(r"(?:Stock Exchange )?trading fee of\s*([\d.]+)\s*%", flat, re.I),
            "afrc_levy_rate": re.search(r"AFRC transaction levy of\s*([\d.]+)\s*%", flat, re.I),
        }
        parsed = {key: (float(match.group(1)) / 100 if match else pd.NA)
                  for key, match in fee_matches.items()}
        return {**row,
                "max_application_price_hkd": float(max_match.group(1).replace(",", "")) if max_match else pd.NA,
                **parsed,
                "application_form_pdf": str(path),
                "form_status": "parsed" if max_match else "price_missing",
                "form_error": "" if max_match else "Maximum Offer Price not found in first six pages"}
    except Exception as exc:
        return {**row, "form_status": "download_or_parse_error",
                "form_error": f"{type(exc).__name__}: {exc}"}


def extract(manifest: pd.DataFrame, *, raw_dir: Path, workers: int = 6,
            stock_id_cache: Path = Path("data/raw/hkex_stock_id_map.json")) -> pd.DataFrame:
    ids = _stock_id_map(stock_id_cache)
    rows = manifest.to_dict("records")
    found: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_find_form, row, ids.get(str(row["stock_code"]).zfill(5), []))
                   for row in rows]
        for future in concurrent.futures.as_completed(futures):
            found.append(future.result())
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        parsed = list(pool.map(lambda row: _parse_form(row, raw_dir), found))
    return pd.DataFrame(parsed).sort_values("listing_event_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/processed/hkex_ipo_prospectus_manifest_2021_2025.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hkex_ipo_application_forms_2021_2025"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/hkex_ipo_application_forms_2021_2025.csv"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--stock-id-cache", type=Path, default=Path("data/raw/hkex_stock_id_map.json"))
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"listing_event_id": str, "stock_code": str})
    result = extract(manifest, raw_dir=args.raw_dir, workers=args.workers,
                     stock_id_cache=args.stock_id_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.form_status.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
