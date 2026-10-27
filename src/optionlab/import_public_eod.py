"""Normalize the public SPY EOD Parquet mirror into the project CSV contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_options(options: pd.DataFrame, underlying: pd.DataFrame, min_dte: int = 0,
                      max_dte: int = 40, spot_band: float = 0.50) -> pd.DataFrame:
    spot = underlying.loc[underlying["symbol"].eq("SPY"), ["date", "close"]].copy()
    spot["date"] = pd.to_datetime(spot["date"]).dt.date
    frame = options.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame.merge(spot.rename(columns={"close": "underlying_price"}), on="date", how="left")
    frame["expiration"] = pd.to_datetime(frame["expiration"]).dt.date
    dte = (pd.to_datetime(frame["expiration"]) - pd.to_datetime(frame["date"].astype(str))).dt.days
    frame = frame.loc[
        frame["underlying_price"].notna() & frame["implied_volatility"].notna()
        & (frame["implied_volatility"] > 0) & (frame["bid"] > 0)
        & (frame["ask"] >= frame["bid"]) & (frame["ask_size"] >= 1)
        & dte.between(min_dte, max_dte)
        & (frame["strike"].div(frame["underlying_price"]).between(1 - spot_band, 1 + spot_band))
        & frame["contract_id"].str.match(r"^SPY\d{6}[CP]\d{8}$")
    ]
    local = pd.to_datetime(frame["date"].astype(str)).dt.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    return pd.DataFrame({
        "quote_time": local.dt.tz_convert("UTC").astype(str),
        "contract_symbol": frame["contract_id"].astype(str),
        "underlying": "SPY",
        "expiration": frame["expiration"],
        "right": frame["type"].str.lower().map({"call": "C", "put": "P"}),
        "strike": frame["strike"], "bid": frame["bid"], "ask": frame["ask"],
        "bid_size": frame["bid_size"], "ask_size": frame["ask_size"],
        "underlying_price": frame["underlying_price"], "contract_multiplier": 100,
        "deliverable_shares": 100, "deliverable_cash": 0,
        "implied_volatility": frame["implied_volatility"],
        "volume": frame["volume"], "open_interest": frame["open_interest"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", required=True, type=Path)
    parser.add_argument("--additional-options", nargs="*", type=Path, default=[])
    parser.add_argument("--underlying", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-dte", type=int, default=0,
                        help="Keep expiry-day quotes needed for open-position exits")
    parser.add_argument("--max-dte", type=int, default=40)
    parser.add_argument("--spot-band", type=float, default=0.50,
                        help="Retain a broad strike band so large moves still have exit quotes")
    args = parser.parse_args()
    sources = [args.options, *args.additional_options]
    underlying = pd.read_parquet(args.underlying)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    if args.output.suffix.lower() == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        try:
            for source in sources:
                normalized = normalize_options(pd.read_parquet(source), underlying,
                                               args.min_dte, args.max_dte, args.spot_band)
                table = pa.Table.from_pandas(normalized, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
                writer.write_table(table)
                total += len(normalized)
                print(f"{source.name}: kept {len(normalized):,} rows")
        finally:
            if writer is not None:
                writer.close()
    else:
        frames = [normalize_options(pd.read_parquet(source), underlying, args.min_dte,
                                    args.max_dte, args.spot_band) for source in sources]
        normalized = pd.concat(frames, ignore_index=True)
        normalized.to_csv(args.output, index=False)
        total = len(normalized)
    print(f"Wrote {total:,} rows to {args.output}")


if __name__ == "__main__":
    main()
