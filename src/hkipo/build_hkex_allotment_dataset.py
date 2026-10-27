"""Download and parse allotment PDFs listed in an HKEX announcement manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from hkipo.parse_allotment_pdf import parse_allotment_pdf


def _get(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "www1.hkexnews.hk":
        raise ValueError(f"Refusing unexpected HKEX URL host: {url}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 HKIPOResearch/0.1"})
    with urlopen(request, timeout=45) as response:
        return response.read()


class _PdfLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._anchor: tuple[str, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a" and attr.get("href"):
            self._anchor = (attr["href"] or "", [])

    def handle_data(self, data: str) -> None:
        if self._anchor:
            self._anchor[1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor:
            href, label = self._anchor
            self.links.append((href, " ".join(" ".join(label).split())))
            self._anchor = None


def _pdf_urls(announcement_url: str, content: bytes) -> list[str]:
    if content.startswith(b"%PDF"):
        return [announcement_url]
    parser = _PdfLinks()
    parser.feed(content.decode("utf-8", errors="replace"))
    links = [(urljoin(announcement_url, href), label) for href, label in parser.links
             if ".pdf" in href.lower()
             and any(tag in label.lower() for tag in ("cover", "summary"))]
    links.sort(key=lambda item: ("summary" not in item[1].lower(), "cover" not in item[1].lower()))
    return [url for url, _ in links]


def _one(row: dict, raw_dir: Path) -> tuple[dict | None, pd.DataFrame | None, pd.DataFrame | None, dict]:
    code = str(row["stock_code"]).zfill(5)
    event_id = row.get("listing_event_id") or f"{code}_{pd.Timestamp(row['listing_date']).strftime('%Y%m%d')}"
    release_date = pd.Timestamp(row['announcement_release_time_hkt']).strftime('%Y%m%d')
    pdf_path = raw_dir / f"{event_id}_{release_date}.pdf"
    try:
        announcement_url = str(row["announcement_url"])
        content = _get(announcement_url)
        candidates = _pdf_urls(announcement_url, content)
        if not candidates:
            raise ValueError("No PDF attachment was linked from the allotment announcement")
        parse_error = None
        for pdf_url in candidates:
            attachment_name = Path(urlparse(pdf_url).path).name
            pdf_path = raw_dir / f"{event_id}_{release_date}_{attachment_name}"
            if not pdf_path.exists():
                pdf_content = content if pdf_url == announcement_url else _get(pdf_url)
                if not pdf_content.startswith(b"%PDF"):
                    parse_error = ValueError(f"Announcement attachment is not a PDF: {pdf_url}")
                    continue
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                pdf_path.write_bytes(pdf_content)
            try:
                summary, tiers, outcomes = parse_allotment_pdf(
                    pdf_path, listing_event_id=event_id, stock_code=code,
                    listing_date=pd.Timestamp(row["listing_date"]).date().isoformat(),
                    release_time_hkt=str(row["announcement_release_time_hkt"]),
                    source_url=pdf_url)
                break
            except Exception as exc:
                parse_error = exc
        else:
            raise ValueError(f"No parseable allocation PDF among {len(candidates)} attachments: {parse_error}")
        calculated = pd.to_numeric(tiers["calculated_allocation_rate_pct"], errors="coerce")
        published = pd.to_numeric(tiers["official_allocation_rate_pct"], errors="coerce")
        rate_differences = (calculated - published).abs()
        rate_mismatches = int((rate_differences.gt(0.051) & rate_differences.notna()).sum())
        rates_observed = int(published.notna().sum())
        rates_total = len(tiers)
        return summary, tiers, outcomes, {"listing_event_id": event_id, "source_status": "parsed",
                                         "pdf_path": str(pdf_path),
                                         "source_pdf_url": pdf_url,
                                         "matches_official_totals": summary["reconciliation"]["matches_official_totals"],
                                         "allocation_rate_tiers_total": rates_total,
                                         "allocation_rate_tiers_observed": rates_observed,
                                         "allocation_rate_tiers_missing": rates_total - rates_observed,
                                         "allocation_rate_mismatches": rate_mismatches,
                                         "all_allocation_rates_reconciled": rates_observed == rates_total and rate_mismatches == 0,
                                         "error": ""}
    except Exception as exc:
        return None, None, None, {"listing_event_id": event_id, "source_status": "parse_or_download_error",
                                  "pdf_path": str(pdf_path), "matches_official_totals": False,
                                  "error": f"{type(exc).__name__}: {exc}"}


def build_dataset(manifest: pd.DataFrame, *, raw_dir: Path, workers: int = 4,
                  limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = manifest.loc[manifest.source_status.eq("found")].copy()
    if limit is not None:
        selected = selected.head(limit)
    jobs = [(row, raw_dir) for row in selected.to_dict("records")]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, row, raw_dir) for row, raw_dir in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    event_rows, tier_frames, outcome_frames, audit_rows = [], [], [], []
    for summary, tiers, outcomes, audit in results:
        audit_rows.append(audit)
        if summary is not None:
            event_rows.append(summary)
            tier_frames.append(tiers)
            outcome_frames.append(outcomes)
    return (pd.DataFrame(event_rows),
            pd.concat(tier_frames, ignore_index=True) if tier_frames else pd.DataFrame(),
            pd.concat(outcome_frames, ignore_index=True) if outcome_frames else pd.DataFrame(),
            pd.DataFrame(audit_rows).sort_values("listing_event_id").reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hkex_allotment_results"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"stock_code": str}, parse_dates=["listing_date"])
    events, tiers, outcomes, audit = build_dataset(manifest, raw_dir=args.raw_dir,
                                                    workers=args.workers, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_dir / "hkipo_allotment_events.csv", index=False)
    tiers.to_csv(args.output_dir / "hkipo_allotment_tiers.csv", index=False)
    outcomes.to_csv(args.output_dir / "hkipo_allotment_outcomes.csv", index=False)
    audit.to_csv(args.output_dir / "parse_audit.csv", index=False)
    print(audit.source_status.value_counts(dropna=False).to_dict())
    print(f"Summary totals reconciled: {int(audit.matches_official_totals.sum())}/{len(audit)}")


if __name__ == "__main__":
    main()
