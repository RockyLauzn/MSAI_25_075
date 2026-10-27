"""Parse a text-based HKEX IPO allotment-results PDF into normalized CSV rows.

The parser intentionally supports a narrow, auditable table layout first. If
the PDF layout differs or totals do not reconcile, it fails or marks the output
for review instead of guessing.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

NUMBER = r"[\d,]+"
ROW_START = re.compile(rf"^\s*({NUMBER})\s+({NUMBER})(?:\s+(.*))?$", re.I)
BALLOT_DETAIL = re.compile(
    rf"(?:(?P<base>{NUMBER})\s+(?:H\s+)?shares?\s+plus\s+)?"
    rf"(?P<winners>{NUMBER})\s+out\s+of\s+(?P<total>{NUMBER})(?:\s+applicants?)?\s+to\s+receive\s+"
    rf"(?:(?:an\s+)?additional\s+)?(?P<increment>{NUMBER})\s+(?:H\s+)?shares?", re.I
)


def _int(value: str) -> int:
    return int(value.replace(",", ""))


def extract_pdf_text(path: Path) -> tuple[str, dict[int, str]]:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    pages = {i + 1: (page.extract_text() or "") for i, page in enumerate(reader.pages)}
    return "\n".join(pages.values()), pages


def _summary_value(text: str, label: str, pattern: str) -> int | float:
    text = re.sub(r"\s+", " ", text)
    match = re.search(label + r"(?:\s*\([^)]*\))*\s*(" + pattern + r")", text, flags=re.I | re.S)
    if not match:
        raise ValueError(f"Could not extract announcement summary field: {label}")
    raw = match.group(1).replace(",", "")
    return float(raw) if "." in raw else int(raw)


def _application_close_date(text: str) -> str | None:
    date_pattern = (r"At the close of the application lists.{0,180}?"
                    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\s*,?\s*"
                    r"(?P<date>\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})")
    match = re.search(date_pattern, re.sub(r"\s+", " ", text), re.I)
    if not match:
        return None
    raw = match.group("date").strip()
    parsed = pd.to_datetime(raw, errors="coerce", dayfirst=bool(re.match(r"\d", raw)))
    return None if pd.isna(parsed) else parsed.date().isoformat()


def parse_allotment_pdf(path: Path, *, listing_event_id: str, stock_code: str,
                        listing_date: str, release_time_hkt: str, source_url: str,
                        language: str = "en") -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    text, pages = extract_pdf_text(path)
    # Keep only the allocation table region; the same numeric/share pattern can
    # appear elsewhere in a long IPO announcement.
    heading_pattern = r"BASIS OF (?:ALLOCATION|ALLOTMENT) (?:UNDER|OF) THE (?:HONG KONG )?PUBLIC\s+OFFER(?:ING)?"
    heading = re.search(heading_pattern, text, re.I)
    if not heading:
        raise ValueError("Allocation table heading was not found")
    raw_rows: list[tuple[int, str, int, int, int]] = []  # page, pool, applied, count, shares
    published_rate_by_tier: dict[tuple[str, int], float] = {}
    current_pool = "not_stated_in_table"
    in_table = False
    blocks: list[tuple[int, str, int, int, str]] = []
    active_block: tuple[int, str, int, int, list[str]] | None = None
    pending_rate_suffix: str | None = None

    def flush_block() -> None:
        nonlocal active_block, pending_rate_suffix
        if active_block is not None:
            page, pool, applied, applicants, lines = active_block
            if pending_rate_suffix:
                lines.append(pending_rate_suffix)
            blocks.append((page, pool, applied, applicants, " ".join(lines)))
            active_block = None
            pending_rate_suffix = None

    # Earlier pages may mention the heading in prose. Start at the last all-caps
    # occurrence, which is the printed allocation-table heading in HKEX PDFs.
    table_start_pages = [page_no for page_no, page_text in pages.items()
                         if re.search(heading_pattern, page_text)]
    if not table_start_pages:
        table_start_pages = [page_no for page_no, page_text in pages.items()
                             if re.search(heading_pattern, page_text, re.I)]
    table_start_page = max(table_start_pages)
    for page_no, page_text in pages.items():
        if page_no < table_start_page:
            continue
        in_table = True
        page_lines: list[str] = []
        for raw_line in page_text.splitlines():
            # Some HKEX PDFs extract adjacent table cells on one line, e.g.
            # "6.01%500 739 500 H Shares". Start the next tier after the rate.
            split_lines = re.split(r"(?<=%)(?=\s*\d[\d,]*\s+\d[\d,]*\s)", raw_line)
            page_lines.extend(line.strip() for line in split_lines if line.strip())
        normalized_lines: list[str] = []
        line_index = 0
        while line_index < len(page_lines):
            line = page_lines[line_index]
            if (re.fullmatch(NUMBER, line) and line_index + 1 < len(page_lines)
                    and re.fullmatch(NUMBER, page_lines[line_index + 1])):
                normalized_lines.append(f"{line} {page_lines[line_index + 1]}")
                line_index += 2
            else:
                normalized_lines.append(line)
                line_index += 1
        for line in normalized_lines:
            if re.search(heading_pattern, line, re.I):
                flush_block()
                continue
            # Plain page numbers sometimes follow the final table row and must
            # not become part of that row's allocation description.
            if (re.fullmatch(NUMBER, line)
                    or re.fullmatch(r"[-–—−]\s*(?:\d\s*)+[-–—−]", line)):
                continue
            if re.match(r"\s*(?:[\d,]+\s+)?Total(?:\s|:|$)", line, re.I):
                flush_block()
                continue
            pool_match = re.match(r"\s*POOL\s+([A-Z0-9]+)(?:\s|$)", line, re.I)
            if pool_match:
                flush_block()
                current_pool = f"pool_{pool_match.group(1).lower()}"
                continue
            row_start = ROW_START.match(line)
            if row_start:
                flush_block()
                applied, applicants, remainder = row_start.groups()
                active_block = (page_no, current_pool, _int(applied), _int(applicants), [remainder or ""])
                continue
            if active_block is not None:
                active_block[4].append(line.strip())
        if re.search(r"As of the date of this announcement", page_text, re.I):
            flush_block()
            break
        # A row can be split across pages when the PDF breaks after "plus";
        # some PDFs place its percentage in the right-hand column before the
        # allocation-text continuation on the next page. Hold that trailing
        # percentage aside, then restore it after the continuation is joined.
        desc = " ".join(active_block[4]) if active_block is not None else ""
        trailing_rate = re.search(r"(\d+(?:\.\d+)?%)\s*$", desc)
        split_row = active_block is not None and re.search(r"\bplus\s*$", desc, re.I)
        split_row = split_row or (active_block is not None and trailing_rate is not None
                                  and re.search(r"\bplus\s+\d+(?:\.\d+)?%\s*$", desc, re.I))
        if split_row and trailing_rate is not None:
            pending_rate_suffix = trailing_rate.group(1)
            active_block[4][-1] = re.sub(r"\s*\d+(?:\.\d+)?%\s*$", "", active_block[4][-1])
        if not split_row:
            flush_block()
    flush_block()

    for page_no, pool, applied, applicants, description in blocks:
        description = re.sub(r"\s*,\s*", ",", description)
        description = re.sub(r"\bre\s+ceive\b", "receive", description, flags=re.I)
        rate_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*$", description)
        if rate_match is None:
            # A small set of HKEX tables omits the percent sign even though
            # the column header explicitly identifies the value as a percent.
            rate_match = re.search(r"(\d+\.\d+)\s*$", description)
        if rate_match:
            published_rate_by_tier[(pool, applied)] = float(rate_match.group(1))
        detail = BALLOT_DETAIL.search(description)
        if detail:
            base = _int(detail.group("base") or "0")
            winners = _int(detail.group("winners"))
            total = _int(detail.group("total"))
            increment = _int(detail.group("increment"))
            if total != applicants or winners > total:
                raise ValueError(f"Ballot row applicant counts disagree on page {page_no}: {description}")
            if total > winners:
                raw_rows.append((page_no, pool, applied, total - winners, base))
            raw_rows.append((page_no, pool, applied, winners, base + increment))
            continue
        simple = re.search(rf"^\s*({NUMBER})\s+(?:H\s+)?Shares?\b", description, re.I)
        if simple:
            raw_rows.append((page_no, pool, applied, applicants, _int(simple.group(1))))
    if not raw_rows:
        raise ValueError("No supported application/allocation rows were found")

    outcomes = []
    for page_no, pool, applied, applicants, allocated_shares in raw_rows:
        outcomes.append({
            "listing_event_id": listing_event_id,
            "stock_code": str(stock_code).zfill(5),
            "listing_date": listing_date,
            "announcement_release_time_hkt": release_time_hkt,
            "announcement_url": source_url,
            "language": language,
            "pool": pool,
            "applied_shares": applied,
            "applicants_in_outcome_group": applicants,
            "allocated_shares_per_applicant": allocated_shares,
            "source_page": page_no,
            "verification_status": "parsed_needs_review",
            "notes": "Parsed from official HKEX announcement PDF; compare against source table before research use.",
        })
    outcomes_frame = pd.DataFrame(outcomes)
    group_columns = ["listing_event_id", "stock_code", "listing_date", "announcement_release_time_hkt",
                     "announcement_url", "language", "pool", "applied_shares"]
    tier_rows = []
    for _, group in outcomes_frame.groupby(group_columns, sort=False, dropna=False):
        total_apps = int(group.applicants_in_outcome_group.sum())
        successful = int(group.loc[group.allocated_shares_per_applicant.gt(0),
                                   "applicants_in_outcome_group"].sum())
        expected_shares = float(
            (group.applicants_in_outcome_group * group.allocated_shares_per_applicant).sum() / total_apps)
        row = group.iloc[0][group_columns].to_dict()
        row.update({
            "source_page": int(group.source_page.min()),
            "total_valid_applications": total_apps,
            "successful_applicants": successful,
            "expected_allocated_shares_per_application": expected_shares,
            "implied_success_probability": successful / total_apps,
            "calculated_allocation_rate_pct": 100 * expected_shares / int(group.applied_shares.iloc[0]),
            "official_allocation_rate_pct": pd.NA,
            "verification_status": "parsed_needs_review",
            "notes": "Aggregated from allocation outcome groups; verify against the published percentage.",
        })
        tier_rows.append(row)
    tiers_frame = pd.DataFrame(tier_rows)
    tiers_frame = tiers_frame.sort_values("applied_shares").reset_index(drop=True)
    tiers_frame["official_allocation_rate_pct"] = tiers_frame.apply(
        lambda row: published_rate_by_tier.get((row.pool, int(row.applied_shares))), axis=1)
    parsed_valid = int(outcomes_frame.applicants_in_outcome_group.sum())
    parsed_successful = int(outcomes_frame.loc[
        outcomes_frame.allocated_shares_per_applicant.gt(0), "applicants_in_outcome_group"].sum())
    try:
        total_valid = int(_summary_value(text, r"(?:No\.|Number) of valid applications", NUMBER))
    except ValueError:
        total_valid = int(_summary_value(text, r"(?:a total of|total of)\s*", NUMBER))
    try:
        total_successful = int(_summary_value(text, r"(?:No\.|Number) of successful applications", NUMBER))
        successful_source = "announcement_summary"
    except ValueError:
        # Some older announcements publish successful counts only in the allocation table.
        total_successful = parsed_successful
        successful_source = "sum_of_official_allocation_groups"
    reconciled = parsed_valid == total_valid and parsed_successful == total_successful
    try:
        final_public_shares = int(_summary_value(
            text, r"Final (?:no\.|number) of Offer Shares under (?:the )?(?:Hong Kong )?Public Offer(?:ing)?\s*(?:is|:)?", NUMBER))
    except ValueError:
        final_public_shares = pd.NA
    try:
        oversubscription = _summary_value(text, r"Subscription level", NUMBER + r"(?:\.\d+)?")
    except ValueError:
        oversubscription = pd.NA
    tiers_frame.loc[:, "verification_status"] = "summary_reconciled" if reconciled else "needs_manual_reconciliation"
    outcomes_frame.loc[:, "verification_status"] = "summary_reconciled" if reconciled else "needs_manual_reconciliation"
    summary = {
        "listing_event_id": listing_event_id,
        "stock_code": str(stock_code).zfill(5),
        "listing_date": listing_date,
        "announcement_release_time_hkt": release_time_hkt,
        "application_close_date": _application_close_date(text),
        "application_close_date_source": "official_allotment_announcement_text",
        "announcement_url": source_url,
        "final_offer_price": _summary_value(
            text, r"(?:Final )?Offer Price(?: has been determined at)?\s*[:：]?\s*HK\$", NUMBER + r"(?:\.\d+)?"),
        "lot_size": int(tiers_frame.applied_shares.min()),
        "final_hk_public_offer_shares": final_public_shares,
        "total_valid_applications": total_valid,
        "total_successful_applicants": total_successful,
        "successful_applicants_source": successful_source,
        "public_offer_oversubscription_multiple": oversubscription,
        "reconciliation": {
            "parsed_valid_applications": parsed_valid,
            "official_valid_applications": total_valid,
            "parsed_successful_applicants": parsed_successful,
            "official_successful_applicants": total_successful,
            "matches_official_totals": reconciled,
        },
    }
    return summary, tiers_frame, outcomes_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--stock-code", required=True)
    parser.add_argument("--listing-date", required=True)
    parser.add_argument("--release-time-hkt", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, tiers, outcomes = parse_allotment_pdf(
        args.pdf, listing_event_id=args.event_id, stock_code=args.stock_code,
        listing_date=args.listing_date, release_time_hkt=args.release_time_hkt,
        source_url=args.source_url)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tiers.to_csv(args.out_dir / "hkipo_allotment_tiers.csv", index=False)
    outcomes.to_csv(args.out_dir / "hkipo_allotment_outcomes.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.out_dir / "hkipo_allotment_events.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
