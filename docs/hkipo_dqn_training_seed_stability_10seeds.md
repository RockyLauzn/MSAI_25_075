# Ten-seed stability check: 2015 vs 2021 training start

> Scope note: the current manuscript excludes all 2015–2020 IPO data. This comparison is retained as an auxiliary historical experiment only; use the 2021-start ten-seed metrics, not the 2015-vs-2021 differences, in the paper's main analysis.

## Design

This follow-up expands the primary 30-session comparison from three to ten matched training seeds. It compares training starts in 2015 and 2021 on the same 2023–2026 event set, keeping the model parameters, training costs, evaluation costs, and per-fold transition budgets fixed. Newly run seeds are 3–9; the original matched seeds 0–2 are retained. The main policies are custom DQN with IPO features, FinRL/SB3 DQN with IPO features, and buy-and-hold. Price-only ablation remains a separate three-seed exploratory analysis.

There are 348 paired test IPOs (68/66/110/104 per year). At the 25 bps one-side evaluation cost, 2015-start versus 2021-start transition budgets are 28,362 / 40,194 / 51,678 / 70,818 per seed by test year. The training IPO counts are 748/816/882/992 versus 163/231/297/407. Exploration decay for custom DQN is anchored to the 2021 reference-set interaction count so the larger training pool does not change the exploration schedule merely by changing the number of data passes.

## Ten-seed result at 25 bps

The first interval below bootstraps held-out IPO events after averaging the ten model seeds within each event. The second interval bootstraps the ten **matched seed-level effects**, where each seed's performance difference is first averaged across all 348 test IPOs. All differences are 2015 start minus 2021 start and are expressed in percentage points (pp).

| Policy | Return change, event bootstrap (95% CI) | Return change, seed bootstrap (95% CI) | Seeds with positive return change | Drawdown change, event bootstrap (95% CI) | Drawdown change, seed bootstrap (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom DQN, IPO features | +0.20 pp (−0.55, +1.07) | +0.20 pp (−0.47, +0.81) | 6/10 | −0.63 pp (−0.92, −0.34) | −0.63 pp (−2.05, +0.71) |
| FinRL DQN, IPO features | −0.53 pp (−1.38, +0.44) | −0.53 pp (−1.55, +0.47) | 5/10 | +0.04 pp (−0.26, +0.35) | +0.04 pp (−1.37, +1.68) |

At 25 bps, ten-seed mean returns were 3.17% for the 2015-start custom DQN, 2.90% for the 2015-start FinRL DQN, and 9.01% for buy-and-hold. The corresponding 2021-start averages were 2.97%, 3.43%, and 9.01%. The DQN policies therefore remained below buy-and-hold returns, although their average drawdowns were lower.

At 10 and 50 bps, the return-change intervals also included zero for both DQN policies. The custom DQN's event-level drawdown interval favored 2015-start at all three fee levels, but its seed-level interval included zero at each fee. Seed-level effects were mixed: only six of ten custom-DQN seeds had a positive pooled return difference, while FinRL had five of ten.

## Conclusion

The expanded training history has not shown a reliable out-of-sample return improvement. The earlier three-seed custom-DQN increase was sensitive to the seed mix: with ten matched seeds the average return difference is small, its interval includes zero, and positive effects occur in only six seeds. Event-level bootstrap suggested lower custom-DQN drawdown after expansion, but seed-level uncertainty is wide and includes no change. FinRL showed no improvement in either return or drawdown.

This is still retrospective: test years 2023–2026 were previously inspected, event bootstrap does not remove shared market-regime dependence, and the added 2015–2020 price-history panel omits 66 IPOs with empty Yahoo Finance histories. The missing-history audit is in [`hkipo_market_data_missingness_audit_2015_2020.md`](hkipo_market_data_missingness_audit_2015_2020.md). A genuinely prospective confirmation should freeze the protocol and use future IPO events.

## Artifacts

- Ten-seed pooled and yearly event/seed intervals: [`../reports/hkipo_train_history_seed10/`](../reports/hkipo_train_history_seed10/).
- Paired event and seed-level differences: `paired_event_deltas.csv` and `paired_event_seed_deltas.csv` in that directory.
- Training runs for newly added seeds 3–9: [`../reports/hkipo_dqn_train_start_comparison_2021_seeds3_9/`](../reports/hkipo_dqn_train_start_comparison_2021_seeds3_9/) and [`../reports/hkipo_dqn_train_start_comparison_2015_seeds3_9/`](../reports/hkipo_dqn_train_start_comparison_2015_seeds3_9/).
- The first three-seed comparison remains in [`hkipo_train_start_history_comparison.md`](hkipo_train_start_history_comparison.md).
