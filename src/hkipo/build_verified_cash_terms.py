"""Combine HKEX application-form and allotment-announcement price terms."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)

ANNOUNCEMENT_MAX = re.compile(
    r"maximum\s+offer\s+price.{0,180}?(?:HKD|HK\$)\s*([\d,]+(?:\.\d+)?)", re.I | re.S)


def _announcement_price(row: object) -> dict:
    try:
        reader = PdfReader(str(Path(row.pdf_path)))
        text = " ".join((page.extract_text() or "") for page in reader.pages[:12])
        match = ANNOUNCEMENT_MAX.search(re.sub(r"\s+", " ", text))
        value = float(match.group(1).replace(",", "")) if match else pd.NA
        return {"listing_event_id": row.listing_event_id,
                "announcement_max_application_price_hkd": value}
    except Exception as exc:
        return {"listing_event_id": row.listing_event_id,
                "announcement_max_application_price_hkd": pd.NA,
                "announcement_price_error": f"{type(exc).__name__}: {exc}"}


def build(*, events_path: Path, forms_path: Path, output: Path,
          coverage_output: Path, search_terms_path: Path | None = None,
          conflict_review_path: Path | None = None,
          workers: int = 8) -> dict:
    events = pd.read_csv(events_path, dtype={"listing_event_id": str}, parse_dates=["listing_date"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        announcement = pd.DataFrame(pool.map(_announcement_price, events.itertuples(index=False)))
    forms = pd.read_csv(forms_path, dtype={"listing_event_id": str})
    form_columns = ["listing_event_id", "max_application_price_hkd", "form_status",
                    "application_form_url", "brokerage_rate", "sfc_levy_rate",
                    "exchange_trading_fee_rate", "afrc_levy_rate"]
    forms = forms[[c for c in form_columns if c in forms.columns]]
    frame = events.merge(announcement, on="listing_event_id", how="left", validate="one_to_one")
    frame = frame.merge(forms, on="listing_event_id", how="left", validate="one_to_one")
    if search_terms_path and search_terms_path.exists():
        search = pd.DataFrame(json.loads(search_terms_path.read_text(encoding="utf-8")))
        if "pdf_body_match" in search.columns:
            search = search.rename(columns={"event_id": "listing_event_id",
                                            "expected_max_price_hkd": "web_search_max_application_price_hkd",
                                            "source_url": "web_search_source_url",
                                            "local_pdf": "web_search_source_pdf_path",
                                            "pdf_body_match": "web_search_valid"})
            search["web_search_pdf_body_match"] = search.web_search_valid
            search = search[["listing_event_id", "web_search_max_application_price_hkd",
                             "web_search_source_url", "web_search_source_pdf_path",
                             "web_search_valid", "web_search_pdf_body_match"]]
        else:
            search = search.rename(columns={"event_id": "listing_event_id",
                                            "max_price": "web_search_max_application_price_hkd",
                                            "url": "web_search_source_url"})
            search["web_search_source_pdf_path"] = pd.NA
            search["web_search_pdf_body_match"] = False
            search = search[["listing_event_id", "web_search_max_application_price_hkd",
                             "web_search_source_url", "web_search_source_pdf_path",
                             "valid", "web_search_pdf_body_match"]].rename(
                                 columns={"valid": "web_search_valid"})
        search = search.drop_duplicates("listing_event_id", keep="last")
        frame = frame.merge(search, on="listing_event_id", how="left", validate="one_to_one")
    else:
        frame["web_search_max_application_price_hkd"] = pd.NA
        frame["web_search_source_url"] = pd.NA
        frame["web_search_source_pdf_path"] = pd.NA
        frame["web_search_valid"] = False
        frame["web_search_pdf_body_match"] = False
    frame["ipo_price_hkd"] = pd.to_numeric(frame.ipo_price_hkd, errors="coerce")
    frame["max_application_price_hkd"] = pd.to_numeric(frame.max_application_price_hkd, errors="coerce")
    frame["announcement_max_application_price_hkd"] = pd.to_numeric(
        frame.announcement_max_application_price_hkd, errors="coerce")
    frame["web_search_max_application_price_hkd"] = pd.to_numeric(
        frame.web_search_max_application_price_hkd, errors="coerce")
    frame["price_sources_agree"] = (
        frame.max_application_price_hkd.notna()
        & frame.announcement_max_application_price_hkd.notna()
        & (frame.max_application_price_hkd - frame.announcement_max_application_price_hkd).abs().le(0.011))

    form_valid = (frame.max_application_price_hkd.notna()
                  & frame.max_application_price_hkd.ge(frame.ipo_price_hkd - 0.011))
    announcement_valid = (frame.announcement_max_application_price_hkd.notna()
                          & frame.announcement_max_application_price_hkd.ge(frame.ipo_price_hkd - 0.011))
    web_search_valid = (frame.web_search_valid.fillna(False).astype(bool)
                        & frame.web_search_source_url.fillna("").str.startswith(
                            "https://www1.hkexnews.hk/listedco/listconews/sehk/")
                        & frame.web_search_max_application_price_hkd.notna()
                        & frame.web_search_max_application_price_hkd.ge(frame.ipo_price_hkd - 0.011))
    frame["exact_max_application_price_hkd"] = frame.max_application_price_hkd.where(
        form_valid, frame.announcement_max_application_price_hkd.where(
            announcement_valid, frame.web_search_max_application_price_hkd.where(web_search_valid)))
    frame["cash_term_source"] = "missing"
    frame.loc[form_valid, "cash_term_source"] = "official_application_form"
    frame.loc[~form_valid & announcement_valid, "cash_term_source"] = "official_allotment_announcement_text"
    frame.loc[~form_valid & ~announcement_valid & web_search_valid,
              "cash_term_source"] = "official_hkex_search_index"
    if "web_search_pdf_body_match" in frame:
        frame.loc[~form_valid & ~announcement_valid & web_search_valid & frame.web_search_pdf_body_match,
                  "cash_term_source"] = "official_hkex_pdf_text"
    frame["term_price_valid"] = frame.exact_max_application_price_hkd.notna()
    frame["max_to_final_price_ratio"] = frame.exact_max_application_price_hkd / frame.ipo_price_hkd
    frame["source_values_differ"] = (
        frame.max_application_price_hkd.notna()
        & frame.announcement_max_application_price_hkd.notna()
        & ~frame.price_sources_agree)
    frame["source_conflict_review_status"] = "not_applicable"
    frame.loc[frame.source_values_differ, "source_conflict_review_status"] = "pending"
    if conflict_review_path and conflict_review_path.exists():
        review = pd.read_csv(conflict_review_path, dtype={"listing_event_id": str})
        review = review[["listing_event_id", "manual_review_status"]].rename(
            columns={"manual_review_status": "source_conflict_review_status"})
        frame = frame.drop(columns=["source_conflict_review_status"]).merge(
            review, on="listing_event_id", how="left", validate="one_to_one")
        frame["source_conflict_review_status"] = frame.source_conflict_review_status.fillna(
            "not_applicable")
        frame.loc[frame.source_values_differ & frame.source_conflict_review_status.eq("not_applicable"),
                  "source_conflict_review_status"] = "pending"
    frame["needs_source_conflict_review"] = (
        frame.source_values_differ & ~frame.source_conflict_review_status.eq("resolved"))

    cols = ["listing_event_id", "listing_date", "ipo_price_hkd", "exact_max_application_price_hkd",
            "cash_term_source", "max_to_final_price_ratio", "term_price_valid", "price_sources_agree",
            "source_values_differ", "source_conflict_review_status", "needs_source_conflict_review",
            "application_form_url", "announcement_max_application_price_hkd",
            "web_search_max_application_price_hkd", "web_search_source_url", "web_search_source_pdf_path",
            "web_search_pdf_body_match",
            "brokerage_rate", "sfc_levy_rate", "exchange_trading_fee_rate", "afrc_levy_rate"]
    frame[cols].to_csv(output, index=False)
    frame["year"] = frame.listing_date.dt.year
    coverage = frame.groupby("year").agg(
        events=("listing_event_id", "size"),
        official_terms=("term_price_valid", "sum"),
        application_form_terms=("cash_term_source", lambda s: int(s.eq("official_application_form").sum())),
        announcement_text_terms=("cash_term_source", lambda s: int(s.eq("official_allotment_announcement_text").sum())),
        official_search_terms=("cash_term_source", lambda s: int(s.eq("official_hkex_search_index").sum())),
        official_pdf_text_terms=("cash_term_source", lambda s: int(s.eq("official_hkex_pdf_text").sum())),
        source_conflicts=("needs_source_conflict_review", "sum"),
        source_value_discrepancies=("source_values_differ", "sum"),
    ).reset_index()
    coverage.to_csv(coverage_output, index=False)
    return {"events": int(len(frame)),
            "events_with_valid_max_application_price": int(frame.term_price_valid.sum()),
            "application_form_terms": int(frame.cash_term_source.eq("official_application_form").sum()),
            "allotment_announcement_terms": int(frame.cash_term_source.eq("official_allotment_announcement_text").sum()),
            "official_search_index_terms": int(frame.cash_term_source.eq("official_hkex_search_index").sum()),
            "official_pdf_text_terms": int(frame.cash_term_source.eq("official_hkex_pdf_text").sum()),
            "source_value_discrepancies": int(frame.source_values_differ.sum()),
            "unresolved_source_conflicts": int(frame.needs_source_conflict_review.sum()),
            "coverage_by_year": coverage.to_dict(orient="records"),
            "price_ratio_quantiles": frame.loc[frame.term_price_valid, "max_to_final_price_ratio"]
                .quantile([0.5, 0.95, 1.0]).to_dict(),
            "limitations": [
                "application-form terms are preferred when they conflict with allotment-announcement text",
                "allotment-announcement matches use text extraction and need manual review when sources conflict",
                "events without a recovered official maximum price remain missing; do not fill from final price silently",
                "this module does not establish historical broker-specific ancillary fees or exact funding timestamps",
            ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path,
                        default=Path("reports/hkipo_simulator_data_2021_2025/verified_events.csv"))
    parser.add_argument("--forms", type=Path,
                        default=Path("data/processed/hkex_ipo_application_forms_2021_2025.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/hkex_ipo_cash_terms_2021_2025.csv"))
    parser.add_argument("--coverage-output", type=Path,
                        default=Path("reports/hkipo_cash_terms_coverage_2021_2025.csv"))
    parser.add_argument("--search-terms", type=Path,
                        default=Path("reports/hkipo_max_price_pdf_source_audit_2021_2025.json"))
    parser.add_argument("--conflict-review", type=Path,
                        default=Path("reports/hkipo_cash_term_source_conflict_review_2021_2025.csv"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
    result = build(events_path=args.events, forms_path=args.forms, output=args.output,
                   coverage_output=args.coverage_output, search_terms_path=args.search_terms,
                   conflict_review_path=args.conflict_review,
                   workers=args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
