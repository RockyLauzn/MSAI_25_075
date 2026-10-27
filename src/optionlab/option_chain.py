"""Validation for normalized historical U.S. listed-options quote snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "quote_time",
    "contract_symbol",
    "underlying",
    "expiration",
    "right",
    "strike",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "underlying_price",
    "contract_multiplier",
    "deliverable_shares",
    "deliverable_cash",
}
OPTIONAL_COLUMNS = {"volume", "open_interest", "implied_volatility"}


def load_quote_snapshots(path: str) -> pd.DataFrame:
    """Load a provider-exported CSV and validate its normalized quote fields.

    Timestamps are normalized to UTC. The source adapter is responsible for
    mapping provider-specific contract identifiers and exchange timestamps.
    """
    return validate_quote_snapshots(pd.read_csv(path))


def validate_quote_snapshots(quotes: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a quote snapshot table already loaded in memory."""
    quotes = quotes.copy()
    missing = REQUIRED_COLUMNS.difference(quotes.columns)
    if missing:
        raise ValueError(f"Option quote file is missing columns: {', '.join(sorted(missing))}")

    quotes = quotes.copy()
    quotes["quote_time"] = pd.to_datetime(quotes["quote_time"], utc=True, errors="raise")
    quotes["expiration"] = pd.to_datetime(quotes["expiration"], errors="raise").dt.date
    quotes["right"] = quotes["right"].astype(str).str.upper()
    for column in ("contract_symbol", "underlying"):
        if quotes[column].isna().any():
            raise ValueError(f"{column} contains missing values")
        quotes[column] = quotes[column].astype(str).str.strip()
        if quotes[column].eq("").any():
            raise ValueError(f"{column} must be nonempty")
    if quotes["quote_time"].isna().any() or quotes["expiration"].isna().any():
        raise ValueError("quote_time and expiration must not be missing")
    if not quotes["right"].isin({"C", "P"}).all():
        raise ValueError("right must use C for call or P for put")

    numeric = [
        "strike", "bid", "ask", "bid_size", "ask_size", "underlying_price",
        "contract_multiplier", "deliverable_shares", "deliverable_cash",
    ]
    numeric += sorted(OPTIONAL_COLUMNS.intersection(quotes.columns))
    for column in numeric:
        quotes[column] = pd.to_numeric(quotes[column], errors="raise")
        if quotes[column].isna().any():
            raise ValueError(f"{column} contains missing values")
        if not np.isfinite(quotes[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values")
    if (quotes[["strike", "bid", "ask"]] < 0).any().any() or (quotes["strike"] == 0).any():
        raise ValueError("strike must be positive; bid and ask must be nonnegative")
    if (quotes["ask"] < quotes["bid"]).any():
        raise ValueError("Found crossed quotes where ask is below bid")
    if (quotes[["bid_size", "ask_size"]] < 0).any().any():
        raise ValueError("Displayed quote sizes must be nonnegative")
    if (quotes["underlying_price"] <= 0).any():
        raise ValueError("underlying_price must be positive")
    if (quotes["contract_multiplier"] <= 0).any():
        raise ValueError("contract_multiplier must be positive")
    if (quotes["deliverable_shares"] < 0).any():
        raise ValueError("deliverable_shares must be nonnegative")
    local_quote_date = quotes["quote_time"].dt.tz_convert("America/New_York").dt.date
    if (quotes["expiration"] < local_quote_date).any():
        raise ValueError("Found expired contracts quoted after expiration")
    return quotes
