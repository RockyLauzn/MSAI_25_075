"""Exploratory Friday-expiry OI concentration and next-week move study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bs_gamma(spot: np.ndarray, strike: np.ndarray, iv: np.ndarray,
              days: np.ndarray) -> np.ndarray:
    t = np.maximum(days, 0.5) / 365.0
    vol_t = np.maximum(iv, 1e-6) * np.sqrt(t)
    d1 = (np.log(spot / strike) + 0.5 * iv**2 * t) / vol_t
    return np.exp(-0.5 * d1**2) / (spot * vol_t * np.sqrt(2 * np.pi))


def analyze_friday_oi(quotes: pd.DataFrame, prices: pd.DataFrame,
                      moneyness_band: float = 0.05) -> tuple[dict, pd.DataFrame]:
    required = {"quote_time", "expiration", "right", "strike", "underlying_price",
                "implied_volatility", "open_interest"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Missing option fields: {', '.join(sorted(missing))}")
    q = quotes.loc[:, list(required)].copy()
    q["date"] = pd.to_datetime(q.quote_time, utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    q["expiration"] = pd.to_datetime(q.expiration, errors="coerce").dt.normalize()
    q["dte"] = (q.expiration - q.date).dt.days
    for col in ("strike", "underlying_price", "implied_volatility", "open_interest"):
        q[col] = pd.to_numeric(q[col], errors="coerce")
    q = q.dropna(subset=["date", "expiration", "strike", "underlying_price", "implied_volatility", "open_interest"])
    q = q.loc[(q.strike > 0) & (q.underlying_price > 0) & (q.implied_volatility > 0)
              & (q.open_interest >= 0)]
    q["right"] = q.right.astype(str).str.upper()
    q["moneyness"] = q.strike / q.underlying_price - 1

    p = prices.loc[:, ["date", "adj_close"]].copy()
    p["date"] = pd.to_datetime(p.date).dt.normalize()
    p["adj_close"] = pd.to_numeric(p.adj_close, errors="raise")
    p = p.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    p["next5_abs_return"] = (p.adj_close.shift(-5) / p.adj_close - 1).abs()
    daily_logret = np.log(p.adj_close).diff()
    p["next5_rv"] = [float(np.sqrt(np.mean(daily_logret.iloc[i+1:i+6]**2) * 252))
                      if i + 6 < len(p) else np.nan for i in range(len(p))]
    p["next5_max_abs_move"] = [float(np.max(np.abs(p.adj_close.iloc[i+1:i+6].to_numpy() /
                                                        p.adj_close.iloc[i] - 1)))
                               if i + 6 < len(p) else np.nan for i in range(len(p))]
    p["next5_max_up"] = [float(np.max(p.adj_close.iloc[i+1:i+6].to_numpy() / p.adj_close.iloc[i] - 1))
                         if i + 6 < len(p) else np.nan for i in range(len(p))]
    p["next5_max_down"] = [float(np.min(p.adj_close.iloc[i+1:i+6].to_numpy() / p.adj_close.iloc[i] - 1))
                           if i + 6 < len(p) else np.nan for i in range(len(p))]

    by_date = {date: frame for date, frame in q.groupby("date", sort=False)}
    dates = pd.DatetimeIndex(sorted(by_date))
    records = []
    # Friday close is the decision point. Use the previous available option-chain
    # session so the feature is conservative even if source OI's timestamp is unclear.
    for i, friday in enumerate(p.date):
        if friday.weekday() != 4 or i + 5 >= len(p):
            continue
        prior = dates[dates < friday]
        if not len(prior):
            continue
        quote_day = prior[-1]
        # Friday expiration is the expiry on the signal date. The prior session
        # generally has one calendar day to expiration; the 0-2d range covers holidays.
        chain = by_date[quote_day]
        chain = chain.loc[chain.expiration.eq(friday) & chain.dte.between(0, 2)
                          & chain.moneyness.abs().le(moneyness_band)].copy()
        if chain.empty:
            continue
        spot = float(p.loc[i, "adj_close"])
        chain["gamma"] = _bs_gamma(chain.underlying_price.to_numpy(), chain.strike.to_numpy(),
                                    chain.implied_volatility.to_numpy(), chain.dte.to_numpy())
        chain["gamma_notional_1pct"] = (chain.open_interest * chain.gamma * 100
                                        * chain.underlying_price**2 * 0.01)
        call_oi = float(chain.loc[chain.right.eq("C"), "open_interest"].sum())
        put_oi = float(chain.loc[chain.right.eq("P"), "open_interest"].sum())
        # This unsigned proxy is not dealer gamma: the public chain does not identify
        # which side is customer/dealer or whether positions are long/short.
        records.append({
            "date": friday, "oi_observation_date": quote_day,
            "weekly_call_oi": call_oi, "weekly_put_oi": put_oi,
            "weekly_total_oi": call_oi + put_oi,
            "weekly_pc_oi_ratio": put_oi / call_oi if call_oi else np.nan,
            "unsigned_gamma_notional_1pct": float(chain.gamma_notional_1pct.sum()),
            "strike_count": int(chain.strike.nunique()),
            "next5_abs_return": p.loc[i, "next5_abs_return"],
            "next5_rv": p.loc[i, "next5_rv"],
            "next5_max_abs_move": p.loc[i, "next5_max_abs_move"],
            "next5_max_up": p.loc[i, "next5_max_up"],
            "next5_max_down": p.loc[i, "next5_max_down"],
        })
    events = pd.DataFrame(records).dropna().reset_index(drop=True)
    if len(events) < 50:
        raise ValueError(f"Too few Friday-expiry observations: {len(events)}")
    score = "unsigned_gamma_notional_1pct"
    cutoff = float(events[score].quantile(0.90))
    events["top_decile"] = events[score].ge(cutoff)
    events["year"] = events.date.dt.year
    year_cutoff = events.groupby("year")[score].transform(lambda s: s.quantile(0.75))
    events["top_year_quartile"] = events[score].ge(year_cutoff)
    high, rest = events.loc[events.top_decile], events.loc[~events.top_decile]
    year_high, year_rest = events.loc[events.top_year_quartile], events.loc[~events.top_year_quartile]
    fields = ["next5_abs_return", "next5_rv", "next5_max_abs_move"]
    summary = {
        "observation_count": int(len(events)),
        "date_start": str(events.date.min().date()),
        "date_end": str(events.date.max().date()),
        "moneyness_band": moneyness_band,
        "high_gamma_proxy_cutoff_full_sample": cutoff,
        "top_decile_count": int(len(high)), "other_count": int(len(rest)),
        "top_decile_mean": {f: float(high[f].mean()) for f in fields},
        "other_mean": {f: float(rest[f].mean()) for f in fields},
        "top_decile_median": {f: float(high[f].median()) for f in fields},
        "other_median": {f: float(rest[f].median()) for f in fields},
        "top_decile_fraction_next5_abs_return_over_3pct": float(high.next5_abs_return.gt(.03).mean()),
        "other_fraction_next5_abs_return_over_3pct": float(rest.next5_abs_return.gt(.03).mean()),
        "top_year_quartile_count": int(len(year_high)),
        "other_within_year_count": int(len(year_rest)),
        "top_year_quartile_mean": {f: float(year_high[f].mean()) for f in fields},
        "other_within_year_mean": {f: float(year_rest[f].mean()) for f in fields},
        "top_year_quartile_fraction_next5_abs_return_over_3pct": float(year_high.next5_abs_return.gt(.03).mean()),
        "other_within_year_fraction_next5_abs_return_over_3pct": float(year_rest.next5_abs_return.gt(.03).mean()),
        "interpretation": "Unsigned OI-weighted gamma concentration proxy only; not dealer net gamma or a causal squeeze label.",
    }
    return summary, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", required=True, type=Path)
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--moneyness-band", type=float, default=.05)
    args = parser.parse_args()
    quotes = pd.read_parquet(args.quotes, columns=["quote_time", "expiration", "right", "strike",
                                                   "underlying_price", "implied_volatility", "open_interest"])
    prices = pd.read_csv(args.prices)
    summary, events = analyze_friday_oi(quotes, prices, args.moneyness_band)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "friday_oi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    events.to_csv(args.output_dir / "friday_oi_events.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
