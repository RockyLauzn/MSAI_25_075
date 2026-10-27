# Current manuscript scope: Hong Kong IPO aftermarket, 2021–2026

## Scope decision

The manuscript's primary data window is restricted to IPO listings from 2021 through 2026. The 2015–2020 IPO observations and the 2015-start training experiments are excluded from the paper's training, testing, headline results, and conclusions. Older-data analyses remain archived as separate exploration only.

## Clean primary empirical result

The primary comparison now uses the clean 30-session event panel and an expanding rolling-origin split: train on IPOs listed from 2021 up to (but excluding) the test year, then evaluate on 2023, 2024, 2025, and 2026. The test folds contain 53, 55, 97, and 94 IPOs, respectively (299 total). Each DQN was trained with ten seeds; fee scenarios are 10, 25, and 50 bps per side. At the prespecified 25 bps per-side cost:

| Policy | IPOs | Mean net return (95% event CI) | Mean max drawdown (95% event CI) |
| --- | ---: | ---: | ---: |
| Custom DQN with IPO features | 299 | 2.89% [0.55%, 5.40%] | 14.42% [13.59%, 15.27%] |
| FinRL/SB3 DQN with IPO features | 299 | 3.31% [0.69%, 6.19%] | 15.46% [14.56%, 16.37%] |
| Buy and hold | 299 | 10.39% [4.91%, 16.37%] | 24.30% [22.85%, 25.80%] |

Custom DQN's paired return difference versus buy-and-hold is **-7.49 percentage points** (95% CI [-11.35, -4.32]); its mean drawdown improves by **9.87 points** (95% CI [9.20, 10.58]). Versus FinRL/SB3 DQN, its return difference is **-0.42 points** (95% CI [-1.18, 0.28]), while drawdown improves by **1.04 points** (95% CI [0.76, 1.32]). Thus the evidence supports a lower-drawdown trade-off, not a return advantage. The DQN results also involve about 6.4–6.8 one-way turnover versus about 1.0 for buy-and-hold.

At 10/25/50 bps, custom DQN mean return is 3.89% / 2.89% / 1.27%; the corresponding 95% event CIs are [1.58%, 6.37%] / [0.55%, 5.40%] / [-1.09%, 3.71%]. Paired return gaps to buy-and-hold are negative at all three costs (-6.66, -7.49, and -8.84 points), with CIs excluding zero. All test years and several configurations have been inspected during prior exploration, so these are retrospective rolling-origin results, not untouched confirmation data.

The earlier 348-event main comparison and earlier 20/40/60-session estimates on the unfiltered panel are **superseded**. The additional horizons have now been separately cleaned and re-trained under predeclared horizon-specific rules. At 25 bps per-side costs:

| Horizon | Test IPOs | Custom DQN mean return | Custom DQN mean max drawdown | Buy-and-hold mean return | Buy-and-hold mean max drawdown | DQN minus B&H return (95% paired CI) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 318 | 1.46% | 11.01% | 5.38% | 21.16% | -3.92 pp [-6.39, -1.60] |
| 30 | 299 | 2.89% | 14.42% | 10.39% | 24.30% | -7.49 pp [-11.35, -4.32] |
| 40 | 278 | 5.59% | 18.16% | 14.03% | 26.86% | -8.45 pp [-12.01, -5.19] |
| 60 | 240 | 6.55% | 22.27% | 19.62% | 32.01% | -13.07 pp [-18.57, -8.23] |

At every horizon, custom DQN has lower paired return but lower drawdown than buy-and-hold. The 20/40/60 test sets are not identical, and the fixed 30-session-equivalent transition budget implies different episode passes/exploration paths at different horizons, so these horizon differences are descriptive. Full cost sensitivity, confidence intervals, per-year metrics, and reproducing artifacts are in [`hkipo_horizon_sensitivity_results_clean.md`](hkipo_horizon_sensitivity_results_clean.md).

## Data window and limitations

- The 2021–2026 HKEX listing master has 533 events; 520 have at least one Yahoo bar, 13 have none, and 9 further events have fewer than 30 bars at the cutoff. Of 316 OHLC-inconsistent rows in the 60-session raw panel, 111 occur in the primary first 30 sessions; 86 complete IPO episodes contain at least one such row and are excluded as whole events. The other 205 inconsistent rows occur after session 30 and are outside the primary horizon. No missing prices or OHLC values are imputed/repaired. The clean panel has 425 events / 12,750 rows; the four test folds use 299 of them. Rule, counts, and reproduction command: [`hkipo_data_quality_rule_2021_2026_30session.md`](hkipo_data_quality_rule_2021_2026_30session.md), [`../reports/hkipo_30session_quality_report_2021_2026.json`](../reports/hkipo_30session_quality_report_2021_2026.json), and [`../reports/hkipo_30session_quality_exclusions_2021_2026.csv`](../reports/hkipo_30session_quality_exclusions_2021_2026.csv).
- Yahoo Finance daily bars are exploratory rather than an exchange-licensed feed. Turnover is estimated in parts of the panel.
- The simulator uses fractional positions and proportional cost assumptions; it does not model Hong Kong board lots, auction fills, market impact, or event-specific slippage.
- A future IPO cohort after the model and evaluation protocol are frozen is needed for prospective confirmation.

## Paper framing

Use a question such as: **Can DQN learn a position-sizing policy that generalizes across Hong Kong IPO aftermarket episodes under a chronological rolling evaluation?** Present the result as an empirical evaluation of return-risk tradeoffs and data/validation constraints. Avoid a claim that DRL creates reliable excess returns.

Discuss Kim et al. (2026), DQN-P, as direct IPO-RL precedent. It is a same-day case study of four Korean IPO stocks and explicitly does not target generalization to the full Korean IPO market ([KCI record](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003394232)). The potential distinction here is chronological, expanding-window cross-IPO evaluation of Hong Kong IPO aftermarket decisions. Do not claim “first to apply DRL to IPOs.” A scoped “to our knowledge, first study of DRL for cross-IPO generalization in Hong Kong IPO aftermarket trading” is only defensible after a documented systematic literature search. See [`hkipo_literature_novelty_note.md`](hkipo_literature_novelty_note.md).

## Reproducibility artifacts

- Clean primary panel: [`../data/processed/hkipo_event_panel_2021_2026_30session_ohlc_clean.csv`](../data/processed/hkipo_event_panel_2021_2026_30session_ohlc_clean.csv).
- Horizon-specific sensitivity rules, quality panels, and outputs: [`hkipo_data_quality_rule_2021_2026_20_40_60session.md`](hkipo_data_quality_rule_2021_2026_20_40_60session.md), [`hkipo_horizon_sensitivity_results_clean.md`](hkipo_horizon_sensitivity_results_clean.md), and `../reports/hkipo_dqn_2021_2026_{20,40,60}session_clean_seed10/`.
- Clean primary run outputs: [`../reports/hkipo_dqn_2021_2026_30session_clean_seed10/summary_event_metrics_ci.csv`](../reports/hkipo_dqn_2021_2026_30session_clean_seed10/summary_event_metrics_ci.csv), [`../reports/hkipo_dqn_2021_2026_30session_clean_seed10/paired_comparisons_ci.csv`](../reports/hkipo_dqn_2021_2026_30session_clean_seed10/paired_comparisons_ci.csv), [`../reports/hkipo_dqn_2021_2026_30session_clean_seed10/per_event_seed_results.csv`](../reports/hkipo_dqn_2021_2026_30session_clean_seed10/per_event_seed_results.csv), and [`../reports/hkipo_dqn_2021_2026_30session_clean_seed10/metadata.json`](../reports/hkipo_dqn_2021_2026_30session_clean_seed10/metadata.json).
- Reproduce the clean primary run using the command recorded in [`hkipo_data_quality_rule_2021_2026_30session.md`](hkipo_data_quality_rule_2021_2026_30session.md), then:
  `python -m hkipo.ipo_feature_horizon_study --panel data/processed/hkipo_event_panel_2021_2026_30session_ohlc_clean.csv --output-dir reports/hkipo_dqn_2021_2026_30session_clean_seed10 --train-start-year 2021 --budget-reference-start-year 2021 --test-years 2023 2024 2025 2026 --horizons 30 --seeds 0 1 2 3 4 5 6 7 8 9 --fees-bps 10 25 50 --train-fee-bps 25 --base-epochs 6 --bootstrap 10000 --skip-price-ablation`.
- Earlier 2015–2020 expansion and 2015-start comparison are documented separately and are outside this manuscript scope.
