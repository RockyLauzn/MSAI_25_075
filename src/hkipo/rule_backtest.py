"""Exploratory long-only rules backtest on HK IPO event bars.

All signals use the listing-day close. Orders are modeled at the next available
session open and exits at the close of the fifth observed holding session.
This is a daily-bar approximation, not a broker-executable simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import _statutory_fee_rate

HOLD_SESSIONS = 5
MAX_POSITIONS = 10
POSITION_FRACTION = 0.10
INITIAL_CASH = 1_000_000.0
RULES = {
    "unconditional_next_open": lambda r: True,
    "close_above_ipo": lambda r: r.close > r.ipo_price,
    "positive_premium_momentum": lambda r: r.close > r.ipo_price and r.close > r.open,
}


def _fee_rate(date: pd.Timestamp, brokerage: float, slippage: float) -> float:
    statutory = float(_statutory_fee_rate(pd.Series([date])).iloc[0])
    return statutory + brokerage + slippage


def prepare_opportunities(panel: pd.DataFrame) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    required = {"listing_event_id", "stock_code", "listing_date", "date", "listed_day",
                "open", "high", "low", "close", "ipo_price", "ohlc_consistent"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Event panel missing columns: {', '.join(sorted(missing))}")
    data = panel.copy()
    data["date"] = pd.to_datetime(data.date, errors="raise").dt.normalize()
    data["listing_date"] = pd.to_datetime(data.listing_date, errors="raise").dt.normalize()
    data["listed_day"] = pd.to_numeric(data.listed_day, errors="raise").astype(int)
    data["ohlc_consistent"] = data.ohlc_consistent.astype(bool)
    rule_candidates: dict[str, list[dict]] = {name: [] for name in RULES}
    event_audit: dict[str, dict] = {}

    for event_id, g in data.groupby("listing_event_id", sort=False):
        g = g.sort_values("listed_day").set_index("listed_day", drop=False)
        base = {"listing_event_id": event_id, "stock_code": str(g.iloc[0].stock_code).zfill(5),
                "listing_date": g.iloc[0].listing_date}
        if 1 not in g.index:
            event_audit[event_id] = {**base, "status": "missing_listing_day_1"}
            continue
        d1 = g.loc[1]
        if not bool(d1.ohlc_consistent):
            event_audit[event_id] = {**base, "status": "invalid_signal_bar"}
            continue
        if 2 not in g.index:
            event_audit[event_id] = {**base, "status": "missing_entry_bar"}
            continue
        if HOLD_SESSIONS + 1 not in g.index:
            event_audit[event_id] = {**base, "status": "missing_exit_bar"}
            continue
        holding = g.loc[2:HOLD_SESSIONS + 1]
        if len(holding) != HOLD_SESSIONS or not holding.ohlc_consistent.all():
            event_audit[event_id] = {**base, "status": "invalid_ohlc_in_holding_window"}
            continue
        entry, exit_row = g.loc[2], g.loc[HOLD_SESSIONS + 1]
        if not (np.isfinite(entry.open) and np.isfinite(exit_row.close)):
            event_audit[event_id] = {**base, "status": "missing_entry_or_exit_price"}
            continue
        event_audit[event_id] = {**base, "status": "valid_bars", "signal_date": d1.date,
                                 "signal_premium": float(d1.close / d1.ipo_price - 1),
                                 "signal_intraday_return": float(d1.close / d1.open - 1)}
        for rule_name, predicate in RULES.items():
            if predicate(d1):
                rule_candidates[rule_name].append({
                    **base, "signal_date": d1.date, "entry_date": entry.date,
                    "entry_open": float(entry.open), "exit_date": exit_row.date,
                    "exit_close": float(exit_row.close),
                    "signal_premium": float(d1.close / d1.ipo_price - 1),
                    "signal_intraday_return": float(d1.close / d1.open - 1),
                })
    return rule_candidates, event_audit


def _simulate_portfolio(candidates: list[dict], market_dates: list[pd.Timestamp], panel: pd.DataFrame,
                        brokerage: float, slippage: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cash = INITIAL_CASH
    positions: dict[str, dict] = {}
    pending: dict[pd.Timestamp, list[dict]] = {}
    exit_schedule: dict[pd.Timestamp, list[str]] = {}
    for trade in candidates:
        pending.setdefault(trade["entry_date"], []).append(trade)
        exit_schedule.setdefault(trade["exit_date"], []).append(trade["listing_event_id"])
    close_lookup = panel.set_index(["listing_event_id", "date"]).close.to_dict()

    daily_rows, closed = [], []
    rejected_capacity = 0
    previous_equity = INITIAL_CASH
    for date in market_dates:
        # New positions enter at the open using only the prior close's portfolio value.
        entries = pending.get(date, [])
        entries = sorted(entries, key=lambda t: (
            -t["signal_premium"], -t["signal_intraday_return"], t["listing_event_id"]))
        for trade in entries:
            if len(positions) >= MAX_POSITIONS:
                rejected_capacity += 1
                closed.append({**trade, "accepted": False, "status": "capacity_limit"})
                continue
            target = min(previous_equity * POSITION_FRACTION, cash / (1 + _fee_rate(date, brokerage, slippage)))
            if target <= 0:
                rejected_capacity += 1
                closed.append({**trade, "accepted": False, "status": "insufficient_cash"})
                continue
            fee = target * _fee_rate(date, brokerage, slippage)
            units = target / trade["entry_open"]
            cash -= target + fee
            positions[trade["listing_event_id"]] = {**trade, "units": units,
                "entry_notional": target, "entry_fee": fee, "last_mark": trade["entry_open"]}

        # Mark open holdings to today's observed close before calculating exits/equity.
        stale_marks = 0
        for event_id, pos in positions.items():
            mark = close_lookup.get((event_id, date))
            if mark is not None and np.isfinite(mark):
                pos["last_mark"] = float(mark)
            else:
                stale_marks += 1

        # Exit at the requested close. Any position without an exit bar remains marked stale.
        for event_id in exit_schedule.get(date, []):
            pos = positions.pop(event_id, None)
            if pos is None:
                continue
            gross_proceeds = pos["units"] * pos["exit_close"]
            exit_fee = gross_proceeds * _fee_rate(date, brokerage, slippage)
            cash += gross_proceeds - exit_fee
            net_return = (gross_proceeds - exit_fee) / (pos["entry_notional"] + pos["entry_fee"]) - 1
            closed.append({**pos, "exit_fee": exit_fee, "gross_return": pos["exit_close"] / pos["entry_open"] - 1,
                           "net_return": net_return, "accepted": True, "status": "closed"})

        # Daily bars from unrelated IPOs supply the market calendar. Missing bars
        # for an open position are marked at its last observed close, never fabricated.
        held_value = sum(p["units"] * p["last_mark"] for p in positions.values())
        equity = cash + held_value
        exposure = held_value / equity if equity > 0 else 0.0
        daily_rows.append({"date": date, "equity": equity, "cash": cash,
                           "open_positions": len(positions), "gross_exposure": exposure,
                           "daily_return": equity / previous_equity - 1 if previous_equity else 0.0,
                           "stale_mark_positions": stale_marks})
        previous_equity = equity

    # Positions still open at data end are valued on their last modeled mark and
    # reported separately; eligible candidates should normally have a day-6 exit.
    for event_id, pos in positions.items():
        closed.append({**pos, "exit_fee": np.nan, "gross_return": np.nan,
                       "net_return": np.nan, "accepted": True, "status": "open_at_sample_end"})
    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(closed)
    if not daily.empty:
        daily["peak"] = daily.equity.cummax()
        daily["drawdown"] = daily.equity / daily.peak - 1
        std = daily.daily_return.std(ddof=1)
        summary = {
            "initial_cash_hkd": INITIAL_CASH,
            "final_equity_hkd": float(daily.equity.iloc[-1]),
            "total_return": float(daily.equity.iloc[-1] / INITIAL_CASH - 1),
            "cagr_252_session_annualized": float(
                (daily.equity.iloc[-1] / INITIAL_CASH) ** (252 / len(daily)) - 1
            ),
            "max_drawdown": float(daily.drawdown.min()),
            "daily_sharpe_zero_rate": float(daily.daily_return.mean() / std * np.sqrt(252)) if std else None,
            "market_sessions": int(len(daily)),
            "position_fraction": POSITION_FRACTION,
            "max_positions": MAX_POSITIONS,
            "capacity_rejections": int(rejected_capacity),
            "mean_gross_exposure": float(daily.gross_exposure.mean()),
            "stale_marks_note": "Daily portfolio marks use last available close when an open event has no bar that date.",
        }
        daily["calendar_year"] = pd.to_datetime(daily.date).dt.year
        annual_rows = []
        last_equity = INITIAL_CASH
        for year, year_data in daily.groupby("calendar_year", sort=True):
            year_end = float(year_data.equity.iloc[-1])
            annual_rows.append({"year": int(year), "return": year_end / last_equity - 1,
                                "sessions": int(len(year_data))})
            last_equity = year_end
        summary["portfolio_returns_by_calendar_year"] = annual_rows
    else:
        summary = {}
    return daily, trades, summary


def run_backtest(panel: pd.DataFrame, ipo_master: pd.DataFrame,
                 brokerage: float = .0025, slippage: float = .0025) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates, audit = prepare_opportunities(panel)
    data = panel.copy()
    data["date"] = pd.to_datetime(data.date).dt.normalize()
    market_dates = sorted(pd.Timestamp(d) for d in data.date.unique())
    ipo_master = ipo_master.copy()
    ipo_master["stock_code"] = ipo_master.stock_code.astype("string").str.zfill(5)
    ipo_master["listing_date"] = pd.to_datetime(ipo_master.listing_date).dt.normalize()
    ipo_master["listing_event_id"] = (ipo_master.stock_code + "_"
                                       + ipo_master.listing_date.dt.strftime("%Y%m%d"))
    universe = set(ipo_master.listing_event_id)
    priced = set(data.listing_event_id.astype(str))
    total_events = len(universe)
    for row in ipo_master.itertuples(index=False):
        event_id = row.listing_event_id
        if event_id not in audit:
            audit[event_id] = {"listing_event_id": event_id, "stock_code": row.stock_code,
                               "listing_date": row.listing_date, "status": "no_price_data"}

    all_trade_rows, daily_frames, report = [], [], {
        "data_source_status": "exploratory Yahoo/yfinance; not exchange-verified",
        "source_events": total_events,
        "events_with_any_price": len(universe & priced),
        "events_without_any_price": total_events - len(universe & priced),
        "brokerage_per_side": brokerage,
        "slippage_per_side_assumption": slippage,
        "holding_sessions": HOLD_SESSIONS,
        "entry_rule_timing": "evaluate listing-day close; enter next observed session open",
        "exit_rule_timing": "exit at close of the fifth observed session after entry (listed_day 6)",
        "fractional_shares": True,
        "portfolio_assumptions": {"initial_cash_hkd": INITIAL_CASH,
                                   "position_fraction": POSITION_FRACTION,
                                   "max_concurrent_positions": MAX_POSITIONS,
                                   "risk_free_rate": 0.0},
        "event_quality_audit": pd.Series([r["status"] for r in audit.values()]).value_counts().to_dict(),
        "strategies": {},
    }
    for rule_name, rule_candidates in candidates.items():
        daily, trades, portfolio_summary = _simulate_portfolio(
            rule_candidates, market_dates, panel, brokerage, slippage)
        if not trades.empty:
            trades.insert(0, "strategy", rule_name)
            rejected = trades.loc[trades.accepted.eq(False), "listing_event_id"]
            for event_id in rejected:
                flag = f"{rule_name}_capacity_rejected"
                audit[str(event_id)][flag] = True
        if not daily.empty:
            daily.insert(1, "strategy", rule_name)
            daily_frames.append(daily)
        all_trade_rows.append(trades)
        completed = trades.loc[trades.net_return.notna()] if not trades.empty else trades
        portfolio_summary.update({
            "candidate_signals": len(rule_candidates),
            "accepted_completed_trades": int(len(completed)),
            "trade_net_mean": float(completed.net_return.mean()) if len(completed) else None,
            "trade_net_median": float(completed.net_return.median()) if len(completed) else None,
            "trade_net_win_rate": float(completed.net_return.gt(0).mean()) if len(completed) else None,
            "trade_net_p05": float(completed.net_return.quantile(.05)) if len(completed) else None,
            "trade_results_by_listing_year": (
                completed.assign(listing_year=pd.to_datetime(completed.listing_date).dt.year)
                .groupby("listing_year").net_return.agg(
                    trades="count", mean="mean", median="median", win_rate=lambda s: s.gt(0).mean())
                .reset_index().to_dict("records") if len(completed) else []
            ),
        })
        report["strategies"][rule_name] = portfolio_summary
    for row in candidates.get("close_above_ipo", []):
        audit[row["listing_event_id"]]["close_above_ipo_signal"] = True
    for row in candidates.get("positive_premium_momentum", []):
        audit[row["listing_event_id"]]["positive_premium_momentum_signal"] = True
    audit_frame = pd.DataFrame(audit.values())
    for flag in ("close_above_ipo_signal", "positive_premium_momentum_signal"):
        if flag not in audit_frame:
            audit_frame[flag] = False
        else:
            audit_frame[flag] = audit_frame[flag].map(
                lambda value: bool(value) if pd.notna(value) else False)
    report["opportunity_audit"] = audit_frame.groupby("status").size().to_dict()
    trades_out = pd.concat(all_trade_rows, ignore_index=True) if all_trade_rows else pd.DataFrame()
    daily_out = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    audit_out = pd.DataFrame(audit.values())
    return report, trades_out, daily_out, audit_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--ipos", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--brokerage-per-side", type=float, default=.0025)
    parser.add_argument("--slippage-per-side", type=float, default=.0025)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, parse_dates=["date", "listing_date"])
    ipos = pd.read_csv(args.ipos, dtype={"stock_code": str}, parse_dates=["listing_date"])
    report, trades, daily, audit = run_backtest(panel, ipos,
        brokerage=args.brokerage_per_side, slippage=args.slippage_per_side)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.output_dir / "trades.csv", index=False)
    daily.to_csv(args.output_dir / "daily_equity.csv", index=False, date_format="%Y-%m-%d")
    audit.to_csv(args.output_dir / "event_audit.csv", index=False, date_format="%Y-%m-%d")
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
