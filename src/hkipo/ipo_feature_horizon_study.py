"""Rolling cross-IPO DRL study with static IPO features and horizon sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN as SB3DQN

from .dqn_aftermarket import DQN, load_episodes, run_episode
from .finrl_dqn_baseline import IPOTradingEnv


def aggregate(frame: pd.DataFrame, out: Path, bootstrap: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    # Average model seeds within each IPO before resampling. B&H appears once.
    event = frame.groupby(["horizon", "test_year", "event_id", "policy", "fee_bps"],
                          as_index=False).agg(final_return=("final_return", "mean"),
                                              max_drawdown=("max_drawdown", "mean"),
                                              turnover=("turnover", "mean"))
    event.to_csv(out / "event_level_averaged.csv", index=False)
    summaries, pairs = [], []
    for (horizon, test_year, fee), g in event.groupby(["horizon", "test_year", "fee_bps"], sort=True):
        for policy, p in g.groupby("policy", sort=True):
            vals = p[["final_return", "max_drawdown", "turnover"]].to_numpy(float)
            ix = rng.integers(0, len(p), size=(bootstrap, len(p)))
            ret_b, dd_b = vals[ix, 0].mean(1), vals[ix, 1].mean(1)
            summaries.append({
                "horizon": int(horizon), "test_year": int(test_year), "events": len(p),
                "policy": policy, "fee_bps": float(fee), "mean_return": vals[:, 0].mean(),
                "median_return": np.median(vals[:, 0]), "win_rate": (vals[:, 0] > 0).mean(),
                "return_ci_low": np.quantile(ret_b, .025), "return_ci_high": np.quantile(ret_b, .975),
                "mean_max_drawdown": vals[:, 1].mean(),
                "drawdown_ci_low": np.quantile(dd_b, .025), "drawdown_ci_high": np.quantile(dd_b, .975),
                "mean_turnover": vals[:, 2].mean(),
            })
        wide_r = g.pivot(index="event_id", columns="policy", values="final_return")
        wide_d = g.pivot(index="event_id", columns="policy", values="max_drawdown")
        main = "custom_DQN_ipo_features"
        comparators = ["buy_and_hold", "FinRL_SB3_DQN_ipo_features"]
        if horizon == 30:
            comparators += ["custom_DQN_price_only", "FinRL_SB3_DQN_price_only"]
        for comparator in comparators:
            if not {main, comparator}.issubset(wide_r.columns):
                continue
            pair = wide_r[[main, comparator]].dropna()
            dd = wide_d.loc[pair.index, [main, comparator]]
            rd = (pair[main] - pair[comparator]).to_numpy()
            db = (dd[comparator] - dd[main]).to_numpy()  # positive favors IPO-feature DQN
            ix = rng.integers(0, len(pair), size=(bootstrap, len(pair)))
            rb, dbb = rd[ix].mean(1), db[ix].mean(1)
            pairs.append({
                "horizon": int(horizon), "test_year": int(test_year), "fee_bps": float(fee),
                "primary_policy": main, "comparator": comparator, "paired_events": len(pair),
                "return_difference": rd.mean(), "return_ci_low": np.quantile(rb, .025),
                "return_ci_high": np.quantile(rb, .975), "drawdown_improvement": db.mean(),
                "drawdown_ci_low": np.quantile(dbb, .025), "drawdown_ci_high": np.quantile(dbb, .975),
            })
        if horizon == 30 and {"FinRL_SB3_DQN_ipo_features", "FinRL_SB3_DQN_price_only"}.issubset(wide_r.columns):
            main, comparator = "FinRL_SB3_DQN_ipo_features", "FinRL_SB3_DQN_price_only"
            pair = wide_r[[main, comparator]].dropna()
            dd = wide_d.loc[pair.index, [main, comparator]]
            rd = (pair[main] - pair[comparator]).to_numpy()
            db = (dd[comparator] - dd[main]).to_numpy()
            ix = rng.integers(0, len(pair), size=(bootstrap, len(pair)))
            rb, dbb = rd[ix].mean(1), db[ix].mean(1)
            pairs.append({"horizon": int(horizon), "test_year": int(test_year), "fee_bps": float(fee),
                          "primary_policy": main, "comparator": comparator, "paired_events": len(pair),
                          "return_difference": rd.mean(), "return_ci_low": np.quantile(rb, .025),
                          "return_ci_high": np.quantile(rb, .975), "drawdown_improvement": db.mean(),
                          "drawdown_ci_low": np.quantile(dbb, .025), "drawdown_ci_high": np.quantile(dbb, .975)})

    # Pooled horizon estimates use a stratified bootstrap, retaining each
    # testing year's observed number of events and therefore its weight.
    for (horizon, fee), g in event.groupby(["horizon", "fee_bps"], sort=True):
        for policy, p in g.groupby("policy", sort=True):
            p = p.reset_index(drop=True)
            vals = p[["final_return", "max_drawdown", "turnover"]].to_numpy(float)
            chunks = []
            for year in sorted(p.test_year.unique()):
                positions = np.flatnonzero(p.test_year.to_numpy() == year)
                chunks.append(positions[rng.integers(0, len(positions), size=(bootstrap, len(positions)))])
            ix = np.concatenate(chunks, axis=1)
            rb, db = vals[ix, 0].mean(1), vals[ix, 1].mean(1)
            summaries.append({
                "horizon": int(horizon), "test_year": "pooled", "events": len(p),
                "policy": policy, "fee_bps": float(fee), "mean_return": vals[:, 0].mean(),
                "median_return": np.median(vals[:, 0]), "win_rate": (vals[:, 0] > 0).mean(),
                "return_ci_low": np.quantile(rb, .025), "return_ci_high": np.quantile(rb, .975),
                "mean_max_drawdown": vals[:, 1].mean(), "drawdown_ci_low": np.quantile(db, .025),
                "drawdown_ci_high": np.quantile(db, .975), "mean_turnover": vals[:, 2].mean(),
            })
        # Pairwise pooled comparisons, stratified by test year.
        main = "custom_DQN_ipo_features"
        comparators = ["buy_and_hold", "FinRL_SB3_DQN_ipo_features"]
        if horizon == 30:
            comparators += ["custom_DQN_price_only", "FinRL_SB3_DQN_price_only"]
        for comparator in comparators:
            wide_r = g.pivot(index=["test_year", "event_id"], columns="policy", values="final_return")
            wide_d = g.pivot(index=["test_year", "event_id"], columns="policy", values="max_drawdown")
            if not {main, comparator}.issubset(wide_r.columns):
                continue
            pair = wide_r[[main, comparator]].dropna()
            dd = wide_d.loc[pair.index, [main, comparator]]
            rd = (pair[main] - pair[comparator]).to_numpy()
            db = (dd[comparator] - dd[main]).to_numpy()
            chunks = []
            years = pair.index.get_level_values("test_year")
            for year in sorted(years.unique()):
                positions = np.flatnonzero(years.to_numpy() == year)
                chunks.append(positions[rng.integers(0, len(positions), size=(bootstrap, len(positions)))])
            ix = np.concatenate(chunks, axis=1)
            rb, dbb = rd[ix].mean(1), db[ix].mean(1)
            pairs.append({
                "horizon": int(horizon), "test_year": "pooled", "fee_bps": float(fee),
                "primary_policy": main, "comparator": comparator, "paired_events": len(pair),
                "return_difference": rd.mean(), "return_ci_low": np.quantile(rb, .025),
                "return_ci_high": np.quantile(rb, .975), "drawdown_improvement": db.mean(),
                "drawdown_ci_low": np.quantile(dbb, .025), "drawdown_ci_high": np.quantile(dbb, .975),
            })
        if horizon == 30 and {"FinRL_SB3_DQN_ipo_features", "FinRL_SB3_DQN_price_only"}.issubset(g.policy.unique()):
            main, comparator = "FinRL_SB3_DQN_ipo_features", "FinRL_SB3_DQN_price_only"
            wide_r = g.pivot(index=["test_year", "event_id"], columns="policy", values="final_return")
            wide_d = g.pivot(index=["test_year", "event_id"], columns="policy", values="max_drawdown")
            pair = wide_r[[main, comparator]].dropna()
            dd = wide_d.loc[pair.index, [main, comparator]]
            rd = (pair[main] - pair[comparator]).to_numpy()
            db = (dd[comparator] - dd[main]).to_numpy()
            chunks = []
            years = pair.index.get_level_values("test_year")
            for year in sorted(years.unique()):
                positions = np.flatnonzero(years.to_numpy() == year)
                chunks.append(positions[rng.integers(0, len(positions), size=(bootstrap, len(positions)))])
            ix = np.concatenate(chunks, axis=1)
            rb, dbb = rd[ix].mean(1), db[ix].mean(1)
            pairs.append({"horizon": int(horizon), "test_year": "pooled", "fee_bps": float(fee),
                          "primary_policy": main, "comparator": comparator, "paired_events": len(pair),
                          "return_difference": rd.mean(), "return_ci_low": np.quantile(rb, .025),
                          "return_ci_high": np.quantile(rb, .975), "drawdown_improvement": db.mean(),
                          "drawdown_ci_low": np.quantile(dbb, .025), "drawdown_ci_high": np.quantile(dbb, .975)})
    pd.DataFrame(summaries).to_csv(out / "summary_event_metrics_ci.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "paired_comparisons_ci.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/hkipo_event_panel_2021_2026_60sessions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hkipo_dqn_ipo_features_horizons"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[20, 30, 40, 60])
    parser.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    parser.add_argument("--train-start-year", type=int, default=2021,
                        help="First calendar year eligible for training; set to 2015 to use the expanded panel")
    parser.add_argument("--budget-reference-start-year", type=int,
                        help="Anchor each fold's total transitions to this training start year; use the same value for paired data-start comparisons")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fees-bps", type=float, nargs="+", default=[10, 25, 50])
    parser.add_argument("--train-fee-bps", type=float, default=25)
    parser.add_argument("--base-epochs", type=int, default=6,
                        help="30-session equivalent interaction budget per training IPO")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--skip-price-ablation", action="store_true",
                        help="Skip the additional price-only model fits at the 30-session horizon")
    args = parser.parse_args()
    all_rows, fold_meta = [], []
    for horizon in args.horizons:
        if horizon not in {20, 30, 40, 60}:
            raise ValueError("Use one of the predeclared horizons: 20, 30, 40, 60")
        for test_year in args.test_years:
            train = load_episodes(args.panel, set(range(args.train_start_year, test_year)), horizon=horizon)
            test = load_episodes(args.panel, {test_year}, horizon=horizon)
            if not train or not test:
                print(f"Skip empty fold: horizon={horizon}, test={test_year}")
                continue
            train_ids = {str(e.listing_event_id.iloc[0]) for e in train}
            test_ids = {str(e.listing_event_id.iloc[0]) for e in test}
            if train_ids & test_ids:
                raise AssertionError(f"IPO event leakage: horizon={horizon}, test={test_year}")
            # Optionally anchor compute to a reference training start, so a
            # larger dataset does not also receive a larger interaction budget.
            budget_train = train
            if args.budget_reference_start_year is not None:
                budget_train = load_episodes(
                    args.panel,
                    set(range(args.budget_reference_start_year, test_year)),
                    horizon=horizon,
                )
                if not budget_train:
                    raise ValueError(
                        f"No budget-reference training episodes for start={args.budget_reference_start_year}, "
                        f"test={test_year}, horizon={horizon}"
                    )
            steps = args.base_epochs * len(budget_train) * 29
            epsilon_reference_steps = sum(len(ep) - 1 for ep in budget_train)
            fold_meta.append({"horizon": horizon, "test_year": test_year,
                              "train_years": list(range(args.train_start_year, test_year)),
                              "train_events": len(train), "test_events": len(test),
                              "budget_reference_start_year": args.budget_reference_start_year,
                              "budget_reference_train_events": len(budget_train),
                              "exploration_reference_transitions": epsilon_reference_steps,
                              "transitions_per_seed": steps})
            print(f"horizon={horizon}, test={test_year}, train={len(train)}, test={len(test)}, steps={steps}", flush=True)

            for seed in args.seeds:
                # IPO-feature state is the primary treatment; price-only state
                # is included at 30 sessions to isolate the added IPO context.
                feature_modes = [("ipo", True)]
                if horizon == 30 and not args.skip_price_ablation:
                    feature_modes.append(("price", False))
                for _mode, include_ipo in feature_modes:
                    label = "ipo_features" if include_ipo else "price_only"
                    model = DQN(seed, args.train_fee_bps / 10000, hidden_dims=(64, 64),
                                include_ipo_features=include_ipo, target_update_interval=1000)
                    model.fit(train, total_transitions=steps,
                              exploration_reference_transitions=epsilon_reference_steps)
                    for fee in args.fees_bps:
                        for ep in test:
                            result = run_episode(ep, model.greedy, fee / 10000, include_ipo)
                            all_rows.append({"horizon": horizon, "test_year": test_year,
                                             "train_events": len(train), "seed": seed,
                                             "event_id": result.event_id, "policy": f"custom_DQN_{label}",
                                             "fee_bps": fee, "final_return": result.final_return,
                                             "max_drawdown": result.max_drawdown, "turnover": result.turnover})

                    env = IPOTradingEnv(train, args.train_fee_bps / 10000, seed,
                                        include_ipo_features=include_ipo)
                    sb3 = SB3DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=1_000_000,
                                 learning_starts=100, batch_size=32, tau=1.0, gamma=.99,
                                 train_freq=4, gradient_steps=1, target_update_interval=10000,
                                 exploration_fraction=.1, exploration_initial_eps=1.0,
                                 exploration_final_eps=.05, max_grad_norm=10,
                                 policy_kwargs={"net_arch": [64, 64]}, verbose=0,
                                 seed=seed, device="cpu")
                    sb3.learn(total_timesteps=steps, progress_bar=False)
                    env.close()

                    def policy(state, _i, _ep):
                        action, _ = sb3.predict(state, deterministic=True)
                        return int(action)

                    for fee in args.fees_bps:
                        for ep in test:
                            result = run_episode(ep, policy, fee / 10000, include_ipo)
                            all_rows.append({"horizon": horizon, "test_year": test_year,
                                             "train_events": len(train), "seed": seed,
                                             "event_id": result.event_id, "policy": f"FinRL_SB3_DQN_{label}",
                                             "fee_bps": fee, "final_return": result.final_return,
                                             "max_drawdown": result.max_drawdown, "turnover": result.turnover})

            for fee in args.fees_bps:
                for ep in test:
                    result = run_episode(ep, lambda _s, _i, _ep: 4, fee / 10000)
                    all_rows.append({"horizon": horizon, "test_year": test_year,
                                     "train_events": len(train), "seed": -1,
                                     "event_id": result.event_id, "policy": "buy_and_hold",
                                     "fee_bps": fee, "final_return": result.final_return,
                                     "max_drawdown": result.max_drawdown, "turnover": result.turnover})

    if not all_rows:
        raise RuntimeError("No complete evaluation events were found")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(out / "per_event_seed_results.csv", index=False)
    aggregate(raw, out, args.bootstrap, seed=20261029)
    panel_frame = pd.read_csv(args.panel, parse_dates=["date", "listing_date"])
    ohlc_bad = int((~panel_frame.ohlc_consistent.astype(bool)).sum()) if "ohlc_consistent" in panel_frame else None
    metadata = {
        "research_question": "Can DRL generalize across Hong Kong IPOs by combining post-listing market observations with IPO characteristics and dynamically controlling exposure?",
        "primary_horizon_sessions": 30, "sensitivity_horizons_sessions": [20, 40, 60],
        "historical_status": "Retrospective exploratory rolling-origin evaluation; historical data have been inspected previously",
        "panel": str(args.panel), "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "training_start_year": args.train_start_year,
        "data_source": "Yahoo Finance daily OHLCV joined to HKEX New Listing Reports; exploratory, not official market feed",
        "panel_coverage": {"rows": len(panel_frame), "events": int(panel_frame.listing_event_id.nunique()),
                           "date_start": str(panel_frame.date.min().date()),
                           "date_end": str(panel_frame.date.max().date()),
                           "ohlc_inconsistent_rows": ohlc_bad,
                           "ohlc_inconsistent_fraction": (ohlc_bad / len(panel_frame) if ohlc_bad is not None else None)},
        "available_static_ipo_features": {
            "ipo_price_log": "log1p(ipo_price) / log1p(1000), clipped to [0,1.1]",
            "sponsor_count_scaled": "slash-delimited sponsor count / 5, clipped to [0,1]",
            "prospectus_lead_days_scaled": "calendar days from prospectus to listing / 60, clipped to [0,1]",
            "excluded_for_point_in_time_uncertainty": ["funds_raised_hkd"],
            "unavailable_excluded": ["lot_size", "free_float_shares", "cornerstone_pct", "industry"]},
        "market_features": ["premium", "ret1", "ret5", "intraday_range", "volume_change", "rv5", "age", "position_weight"],
        "target_weights": [0, .25, .5, .75, 1], "execution": "observe close t, rebalance open t+1, mark close t+1",
        "fixed_interaction_budget": {"base_30_session_equivalent_epochs": args.base_epochs,
                                     "transitions_per_training_event": args.base_epochs * 29,
                                     "budget_reference_start_year": args.budget_reference_start_year,
                                     "per_seed_steps_by_fold": fold_meta},
        "training_fee_bps_per_side": args.train_fee_bps, "evaluation_fee_bps_per_side": args.fees_bps,
        "seeds": args.seeds, "folds": fold_meta,
        "custom_dqn": {"hidden_dims": [64, 64], "learning_rate": .0005, "gamma": .97,
                       "replay_capacity": 30000, "batch_size": 64, "update_every_transitions": 32,
                       "target_network_sync_every_transitions": 1000,
                       "exploration_schedule": "epsilon= max(0.05, 0.8*0.97^virtual_epoch), with virtual epochs anchored to the same reference-training-set transition count in paired data-window comparisons"},
        "finrl_baseline": {"implementation": "Stable-Baselines3 DQN in the matched IPO Gym environment",
                           "hidden_dims": [64, 64], "learning_rate": .0001,
                           "training_steps_matched_to_custom": True},
        "bootstrap": {"replicates": args.bootstrap, "unit": "IPO event after averaging training seeds",
                      "pooled_intervals": "stratified within test year"},
        "limitations": ["event bootstrap does not remove common market-regime dependence",
                        "IPO feature set is limited to attributes available in the present panel",
                        "fractional shares and proportional cost proxy; no auction fill, slippage, impact, or lot constraint",
                        "daily bars from Yahoo Finance are exploratory, not a licensed exchange feed"]}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out}; horizons={sorted(raw.horizon.unique())}; events={raw.event_id.nunique()}")


if __name__ == "__main__":
    main()
