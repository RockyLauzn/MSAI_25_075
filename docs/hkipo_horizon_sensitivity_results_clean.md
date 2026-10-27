# Clean 20/40/60-session horizon sensitivity results

**Run date:** 2026-10-27. These are the horizon-specific reruns governed by [`hkipo_data_quality_rule_2021_2026_20_40_60session.md`](hkipo_data_quality_rule_2021_2026_20_40_60session.md). They supersede the earlier 20/40/60 results on an unfiltered panel. They use 2021–2026 only, expanding rolling tests on 2023–2026, ten training seeds, 10/25/50 bps per-side evaluation fees, and event-level bootstrap intervals. All intervals below are 95% bootstrap intervals after averaging model seeds within IPO; pooled intervals are stratified by test year.

## Horizon-specific coverage

| Horizon | Eligible episodes in 2021–2026 | Test IPOs | Test cohorts (2023 / 2024 / 2025 / 2026) | Excluded for no bars | Incomplete horizon | Events with an OHLC flag in horizon |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 20 sessions | 464 | 318 | 59 / 60 / 101 / 98 | 13 | 7 | 49 |
| 40 sessions | 388 | 278 | 47 / 50 / 92 / 89 | 13 | 15 | 117 |
| 60 sessions | 334 | 240 | 37 / 41 / 85 / 77 | 13 | 17 | 169 |

Each panel contains exactly `horizon × eligible IPOs` rows, every row passes the OHLC check, and each event is used as a complete episode. Horizon samples are nested complete-case cohorts, not identical cross-horizon samples. The three quality reports and event exclusion lists are saved in `reports/hkipo_{20,40,60}session_quality_report_2021_2026.json` and `reports/hkipo_{20,40,60}session_quality_exclusions_2021_2026.csv`.

## Pooled performance at 25 bps one-way cost

| Horizon | Policy | Events | Mean net return (95% CI) | Mean maximum drawdown (95% CI) | One-way turnover |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | Custom DQN + IPO features | 318 | 1.46% [-0.19%, 3.22%] | 11.01% [10.37%, 11.67%] | 4.30× |
|  | SB3 DQN + IPO features | 318 | 1.72% [-0.35%, 3.86%] | 14.22% [13.41%, 15.04%] | 4.68× |
|  | Buy-and-hold | 318 | 5.38% [1.56%, 9.45%] | 21.16% [19.80%, 22.50%] | 1.00× |
| 40 | Custom DQN + IPO features | 278 | 5.59% [2.49%, 8.95%] | 18.16% [17.16%, 19.19%] | 7.98× |
|  | SB3 DQN + IPO features | 278 | 4.30% [1.35%, 7.58%] | 18.10% [17.09%, 19.14%] | 9.28× |
|  | Buy-and-hold | 278 | 14.03% [7.88%, 20.85%] | 26.86% [25.33%, 28.46%] | 1.00× |
| 60 | Custom DQN + IPO features | 240 | 6.55% [2.27%, 11.32%] | 22.27% [21.05%, 23.50%] | 13.21× |
|  | SB3 DQN + IPO features | 240 | 4.75% [0.87%, 8.91%] | 22.03% [20.88%, 23.22%] | 13.06× |
|  | Buy-and-hold | 240 | 19.62% [10.81%, 29.67%] | 32.01% [30.21%, 33.89%] | 1.00× |

## Paired comparisons at 25 bps

Positive drawdown improvement means the custom DQN has a lower mean maximum drawdown than the comparator.

| Horizon | Comparator | DQN minus comparator return (95% CI) | Drawdown improvement (95% CI) |
| ---: | --- | ---: | ---: |
| 20 | Buy-and-hold | -3.92 pp [-6.39, -1.60] | 10.14 pp [9.39, 10.93] |
| 20 | SB3 DQN | -0.26 pp [-0.98, 0.46] | 3.21 pp [2.90, 3.53] |
| 40 | Buy-and-hold | -8.45 pp [-12.01, -5.19] | 8.71 pp [8.06, 9.36] |
| 40 | SB3 DQN | 1.29 pp [0.40, 2.21] | -0.06 pp [-0.34, 0.24] |
| 60 | Buy-and-hold | -13.07 pp [-18.57, -8.23] | 9.74 pp [8.99, 10.50] |
| 60 | SB3 DQN | 1.80 pp [0.67, 2.98] | -0.24 pp [-0.60, 0.12] |

The custom DQN earns less than buy-and-hold in every horizon. Its drawdown is materially lower at every horizon. Relative to the SB3 DQN, the custom policy has no clear return advantage at 20 sessions; it has higher returns at 40 and 60 sessions in this retrospective comparison, while drawdown differences at those longer horizons are not distinguishable from zero by the paired bootstrap intervals.

## Custom-DQN cost sensitivity

| Horizon | 10 bps mean return (95% CI) | 25 bps mean return (95% CI) | 50 bps mean return (95% CI) |
| ---: | ---: | ---: | ---: |
| 20 | 2.11% [0.48%, 3.84%] | 1.46% [-0.19%, 3.22%] | 0.38% [-1.24%, 2.13%] |
| 40 | 6.86% [3.67%, 10.19%] | 5.59% [2.49%, 8.95%] | 3.48% [0.38%, 6.65%] |
| 60 | 8.64% [4.30%, 13.37%] | 6.55% [2.27%, 11.32%] | 3.17% [-1.00%, 7.69%] |

At each horizon and all three cost assumptions, the custom DQN’s paired return difference versus buy-and-hold is negative, with its 95% interval below zero. Higher fees reduce DQN returns, especially for the longer-horizon, higher-turnover policies.

## Interpretation limits

Horizon comparisons are descriptive. The eligible event set changes with horizon, and models receive the same 30-session-equivalent transition budget per training IPO: `6 × number_of_training_IPOs × 29` steps. Since episode length is `h − 1` transitions, the number of full episode passes and the epsilon-decay trajectory vary by horizon. Thus these estimates do not isolate the causal effect of holding an otherwise identical policy for longer. All cohorts were inspected in prior exploration; the results are retrospective rather than untouched prospective confirmation.

## Reproducibility artifacts

- Clean panels: `data/processed/hkipo_event_panel_2021_2026_{20,40,60}session_ohlc_clean.csv`.
- Per-horizon quality coverage/exclusions: `reports/hkipo_{20,40,60}session_quality_report_2021_2026.json` and `reports/hkipo_{20,40,60}session_quality_exclusions_2021_2026.csv`.
- Per-horizon metrics: `reports/hkipo_dqn_2021_2026_{20,40,60}session_clean_seed10/summary_event_metrics_ci.csv`, `paired_comparisons_ci.csv`, `per_event_seed_results.csv`, and `metadata.json`.
- The main clean 30-session result is recorded in [`hkipo_paper_scope_2021_2026.md`](hkipo_paper_scope_2021_2026.md).

To reproduce one horizon after building its clean panel with `quality_gate_horizon_panel`, run the following, substituting `H` with `20`, `40`, or `60` in both paths and `--horizons`:

```powershell
python -m hkipo.ipo_feature_horizon_study `
  --panel data/processed/hkipo_event_panel_2021_2026_Hsession_ohlc_clean.csv `
  --output-dir reports/hkipo_dqn_2021_2026_Hsession_clean_seed10 `
  --train-start-year 2021 --budget-reference-start-year 2021 `
  --test-years 2023 2024 2025 2026 --horizons H `
  --seeds 0 1 2 3 4 5 6 7 8 9 --fees-bps 10 25 50 `
  --train-fee-bps 25 --base-epochs 6 --bootstrap 10000 --skip-price-ablation
```
