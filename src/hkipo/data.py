"""Validation and point-in-time feature construction for HK IPO daily bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

IPO_REQUIRED = {"stock_code", "listing_date", "ipo_price"}
BAR_REQUIRED = {"stock_code", "date", "open", "high", "low", "close", "volume", "turnover"}


def _normalize_code(values: pd.Series) -> pd.Series:
    code = values.astype("string").str.upper().str.replace(r"\.HK$", "", regex=True)
    return code.str.zfill(5)


def build_event_panel(ipo_master: pd.DataFrame, daily_bars: pd.DataFrame,
                      max_sessions: int = 30) -> pd.DataFrame:
    """Join IPO reference data to bars and compute features observable by each close.

    ``ipo_master`` has one row per listing event. ``daily_bars`` must contain raw,
    unadjusted tradable OHLCV/turnover. No forward labels are created here.
    """
    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    missing_ipo = IPO_REQUIRED.difference(ipo_master.columns)
    missing_bars = BAR_REQUIRED.difference(daily_bars.columns)
    if missing_ipo:
        raise ValueError(f"IPO master missing columns: {', '.join(sorted(missing_ipo))}")
    if missing_bars:
        raise ValueError(f"Daily bars missing columns: {', '.join(sorted(missing_bars))}")

    master = ipo_master.copy()
    bars = daily_bars.copy()
    master["stock_code"] = _normalize_code(master.stock_code)
    bars["stock_code"] = _normalize_code(bars.stock_code)
    master["listing_date"] = pd.to_datetime(master.listing_date, errors="raise").dt.normalize()
    bars["date"] = pd.to_datetime(bars.date, errors="raise").dt.normalize()
    if master.duplicated(["stock_code", "listing_date"]).any():
        raise ValueError("IPO master must have one row per stock_code/listing_date event")
    master = master.sort_values(["stock_code", "listing_date"])
    master["listing_event_id"] = (master.stock_code + "_" + master.listing_date.dt.strftime("%Y%m%d"))
    master["next_listing_date"] = master.groupby("stock_code").listing_date.shift(-1)
    if bars.duplicated(["stock_code", "date"]).any():
        raise ValueError("Daily bars must have at most one row per stock_code/date")

    for col in ("ipo_price",):
        master[col] = pd.to_numeric(master[col], errors="raise")
        if not np.isfinite(master[col]).all() or (master[col] <= 0).any():
            raise ValueError(f"{col} must contain finite positive values")
    if "lot_size" in master:
        master["lot_size"] = pd.to_numeric(master["lot_size"], errors="coerce")
        if (master.lot_size.dropna() <= 0).any():
            raise ValueError("Non-missing lot_size values must be positive")
    numeric_bar_cols = ["open", "high", "low", "close", "volume", "turnover"]
    for col in numeric_bar_cols:
        bars[col] = pd.to_numeric(bars[col], errors="raise")
        if not np.isfinite(bars[col]).all():
            raise ValueError(f"{col} must contain finite values")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (bars[["volume", "turnover"]] < 0).any().any():
        raise ValueError("Volume and turnover cannot be negative")
    bars["ohlc_consistent"] = (
        bars.high.ge(bars.low)
        & bars.high.ge(bars[["open", "close"]].max(axis=1))
        & bars.low.le(bars[["open", "close"]].min(axis=1))
    )

    # Support a combined file containing both legacy bars (no event ID) and
    # newer event-keyed bars. Missing IDs must never be cast to the literal "nan".
    if "listing_event_id" in bars:
        explicit = bars.loc[bars.listing_event_id.notna()].copy()
        legacy = bars.loc[bars.listing_event_id.isna()].drop(columns="listing_event_id")
        parts = []
        if not explicit.empty:
            explicit["listing_event_id"] = explicit.listing_event_id.astype(str)
            # The event key determines the canonical code from the IPO master.
            explicit = explicit.drop(columns="stock_code")
            parts.append(explicit.merge(master.drop(columns="next_listing_date"), on="listing_event_id",
                                        how="inner", validate="many_to_one"))
        if not legacy.empty:
            joined = legacy.merge(master, on="stock_code", how="inner", validate="many_to_many")
            joined = joined.loc[joined.date.ge(joined.listing_date)
                                & (joined.next_listing_date.isna() | joined.date.lt(joined.next_listing_date))]
            parts.append(joined.drop(columns="next_listing_date"))
        panel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        panel = bars.merge(master, on="stock_code", how="inner", validate="many_to_many")
        panel = panel.loc[panel.date.ge(panel.listing_date)
                          & (panel.next_listing_date.isna() | panel.date.lt(panel.next_listing_date))]
        panel = panel.drop(columns="next_listing_date")
    panel = panel.sort_values(["listing_event_id", "date"])
    panel["listed_day"] = panel.groupby("listing_event_id").cumcount() + 1
    panel = panel.loc[panel.listed_day.le(max_sessions)].copy()
    grouped = panel.groupby("listing_event_id", sort=False)
    panel["return_1d"] = grouped.close.pct_change()
    panel["premium_to_ipo"] = panel.close / panel.ipo_price - 1.0
    panel["return_5d"] = grouped.close.pct_change(5)
    panel["rv_5d"] = grouped.return_1d.transform(
        lambda s: s.rolling(5, min_periods=3).std(ddof=1) * np.sqrt(252)
    )
    if "free_float_shares" in panel:
        shares = pd.to_numeric(panel.free_float_shares, errors="coerce")
        panel["turnover_rate"] = panel.volume / shares.where(shares > 0)
    return panel.reset_index(drop=True)
