"""Normalize a licensed/vendor daily-bar CSV into the HK IPO bar contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REQUIRED = ("stock_code", "date", "open", "high", "low", "close", "volume", "turnover")


def _clean_number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype("string").str.replace(",", "", regex=False), errors="coerce")


def normalize_csv(path: Path, source_name: str, mapping: dict[str, str],
                  price_basis: str = "raw") -> pd.DataFrame:
    """Read daily bars using explicit source-column -> canonical-column mapping."""
    if price_basis != "raw":
        raise ValueError("IPO aftermarket panel requires raw, unadjusted OHLC prices")
    source_bytes = path.read_bytes()
    raw = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    missing_map = [key for key in REQUIRED if key not in mapping]
    if missing_map:
        raise ValueError(f"Column mapping missing canonical fields: {', '.join(missing_map)}")
    absent = [mapping[key] for key in REQUIRED if mapping[key] not in raw.columns]
    if absent:
        raise ValueError(f"Source CSV columns not found: {', '.join(absent)}")

    bars = pd.DataFrame({key: raw[mapping[key]] for key in REQUIRED})
    bars["stock_code"] = bars.stock_code.str.replace(r"\.HK$", "", regex=True).str.strip().str.zfill(5)
    bars["date"] = pd.to_datetime(bars.date, errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "volume", "turnover"):
        bars[column] = _clean_number(bars[column])

    if bars.isna().any().any():
        bad = bars.isna().sum()
        raise ValueError(f"Missing/unparseable values by field: {bad[bad.gt(0)].to_dict()}")
    if bars.duplicated(["stock_code", "date"]).any():
        examples = bars.loc[bars.duplicated(["stock_code", "date"], keep=False),
                            ["stock_code", "date"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate stock_code/date rows; examples: {examples}")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (bars[["volume", "turnover"]] < 0).any().any():
        raise ValueError("Volume and turnover cannot be negative")

    bars["ohlc_consistent"] = (
        bars.high.ge(bars.low)
        & bars.high.ge(bars[["open", "close"]].max(axis=1))
        & bars.low.le(bars[["open", "close"]].min(axis=1))
    )
    bars["turnover_is_estimate"] = False
    bars["source"] = source_name
    bars["source_file"] = path.name
    bars["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    bars["price_basis"] = price_basis
    return bars.sort_values(["stock_code", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-name", required=True,
                        help="Provider/product name as shown on the data license or file")
    parser.add_argument("--column-map", required=True, type=Path,
                        help="JSON mapping from canonical fields to source CSV headers")
    parser.add_argument("--price-basis", choices=["raw", "adjusted"], default="raw")
    args = parser.parse_args()
    mapping = json.loads(args.column_map.read_text(encoding="utf-8"))
    bars = normalize_csv(args.input, args.source_name, mapping, args.price_basis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(args.output, index=False)
    report = {
        "source": args.source_name,
        "source_file": args.input.name,
        "source_sha256": bars.source_sha256.iloc[0] if len(bars) else hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "rows": int(len(bars)),
        "unique_symbols": int(bars.stock_code.nunique()),
        "date_start": bars.date.min() if len(bars) else None,
        "date_end": bars.date.max() if len(bars) else None,
        "ohlc_inconsistent_rows": int((~bars.ohlc_consistent).sum()),
        "ohlc_inconsistent_examples": bars.loc[~bars.ohlc_consistent,
            ["stock_code", "date", "open", "high", "low", "close"]].head(20).to_dict("records"),
    }
    report_path = args.output.with_name(f"{args.output.stem}_quality.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
