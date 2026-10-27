# 2015 vs 2021 training-window comparison

## Research question

Does adding older Hong Kong IPO aftermarket episodes improve cross-IPO generalization when the test years, model setup, random seeds, transaction costs, and total training interactions are held constant?

This is a retrospective exploratory comparison. The 2023–2026 test years have already been inspected in earlier work, so these are not untouched confirmatory holdouts.

## Paired design

- Primary horizon: 30 listed sessions; test years: 2023, 2024, 2025, and 2026.
- Test set: the same 348 IPO events in both arms (68, 66, 110, and 104 events by year).
- Training starts: 2021 baseline versus 2015 expanded history.
- Per-fold training IPO counts (2021 start vs 2015 start): 163 vs 748 (2023 test), 231 vs 816 (2024), 297 vs 882 (2025), and 407 vs 992 (2026).
- The total transition budget is anchored to the 2021 arm: 28,362; 40,194; 51,678; and 70,818 transitions per seed by test year. Both arms use the same budget in each fold.
- Both use seeds 0, 1, and 2; 25 bps one-side training cost; 10, 25, and 50 bps one-side evaluation costs; custom DQN, FinRL-compatible Stable-Baselines3 DQN, and buy-and-hold. The 30-session IPO-feature and price-only ablations are also included.
- Custom DQN uses 64×64 hidden layers, learning rate 0.0005, gamma 0.97, replay capacity 30,000, batch size 64, updates every 32 transitions, and target sync every 1,000 transitions. FinRL DQN uses the matched Gym environment, 64×64 layers, learning rate 0.0001, and the same number of total environment steps.
- The custom DQN's original exploration decay was measured in passes through the training set, which would change merely because one arm has more IPOs. The expanded run therefore uses a virtual-epoch schedule anchored to the 2021 training set's transition count. For the 30-session baseline, this reproduces the original six-pass epsilon schedule exactly; SB3's exploration schedule is already defined by total training steps.

The larger training pool necessarily spreads the same total interactions over more IPO events. Thus this experiment tests the value of added event diversity under a fixed compute budget, not the effect of giving the expanded model more gradient updates.

## Results at 25 bps one-side evaluation cost

Returns and drawdowns below are event-level averages after averaging the three training seeds within each event. Differences are **2015 start minus 2021 start**, with 95% IPO-event bootstrap intervals stratified by test year. Return and drawdown values are percentage points (pp).

| Policy | Mean return, 2021 start | Mean return, 2015 start | Return change (95% CI) | Mean max drawdown, 2021 start | Mean max drawdown, 2015 start | Drawdown change (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom DQN, IPO features | 2.61% | 3.34% | +0.73 pp (−0.06, +1.53) | 13.90% | 13.99% | +0.09 pp (−0.28, +0.45) |
| FinRL DQN, IPO features | 3.53% | 3.20% | −0.34 pp (−1.45, +0.76) | 15.50% | 14.80% | −0.70 pp (−1.23, −0.19) |
| Buy and hold | 9.01% | 9.01% | 0.00 pp | 25.08% | 25.08% | 0.00 pp |

The custom DQN's return point estimate rose, but its interval includes zero and its average drawdown did not improve. FinRL DQN's return point estimate fell; its average drawdown improved by about 0.70 pp. Neither model produced a clear return uplift from the older training history. Both DQN return averages remained below buy-and-hold by about 5.7–5.8 pp at this cost setting, while showing lower average drawdowns.

Cost sensitivity did not change that reading. For the custom IPO-feature DQN, the pooled return change was +0.71 pp at 10 bps (95% CI −0.11 to +1.52), +0.73 pp at 25 bps (−0.06 to +1.53), and +0.77 pp at 50 bps (−0.01 to +1.55). Each interval includes zero. FinRL IPO-feature return changes were −0.35, −0.34, and −0.32 pp, respectively, also with intervals including zero. The FinRL drawdown reduction was directionally consistent across all three fee settings.

## Interpretation and limitations

The expanded dataset materially increases training IPO diversity, but this comparison does **not** show a robust improvement in out-of-sample returns. The custom DQN has a small positive return estimate with uncertainty spanning no change; the FinRL baseline shows no return benefit. The risk-return trade-off also differs by algorithm, so there is no common improvement across both agents.

The event bootstrap does not remove shared market-regime dependence across IPOs. 2023–2026 were previously inspected, Yahoo Finance daily bars are exploratory rather than an exchange-licensed feed, and 66 of the 651 added 2015–2020 IPO listings lack retrieved bars. These results should be treated as a useful retrospective check, not evidence that the added years improve future live performance.

## Artifacts and reproduction

- Baseline run: [`../reports/hkipo_dqn_ipo_features_horizons/`](../reports/hkipo_dqn_ipo_features_horizons/).
- 2015-start run: [`../reports/hkipo_dqn_train_start_comparison_2015/`](../reports/hkipo_dqn_train_start_comparison_2015/).
- Paired event deltas and bootstrap intervals: [`../reports/hkipo_train_history_comparison/`](../reports/hkipo_train_history_comparison/).
- Yearly training counts and transition budgets are recorded in each run's `metadata.json`.
- Rebuild the expanded run with `python -m hkipo.ipo_feature_horizon_study --panel data/processed/hkipo_event_panel_2015_2026_60sessions.csv --output-dir reports/hkipo_dqn_train_start_comparison_2015 --train-start-year 2015 --budget-reference-start-year 2021 --test-years 2023 2024 2025 2026 --horizons 30 --seeds 0 1 2 --fees-bps 10 25 50 --train-fee-bps 25 --base-epochs 6 --bootstrap 10000`.
- Rebuild the paired analysis with `python -m hkipo.compare_training_history --baseline reports/hkipo_dqn_ipo_features_horizons/per_event_seed_results.csv --expanded reports/hkipo_dqn_train_start_comparison_2015/per_event_seed_results.csv --output-dir reports/hkipo_train_history_comparison --horizon 30 --bootstrap 10000`.
