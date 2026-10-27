# HKEX IPO aftermarket panel expansion: 2015–2026

## What was added

HKEX New Listing Reports were imported for 2015–2020 and joined with the existing 2021–2026 panel. The expanded master list contains 1,184 IPO events. Daily bars are available for 1,105 events; 1,096 have at least 30 listed sessions and 1,087 have at least 60.

| Period | Master IPOs | Events with bars | Complete 60 sessions |
| --- | ---: | ---: | ---: |
| 2015–2020 added | 651 | 585 | 584 |
| 2021–2026 existing | 533 | 520 | 503 |
| 2015–2026 total | 1,184 | 1,105 | 1,087 |

Coverage by year, including unavailable histories and incomplete 2026 episodes, is recorded in [`../reports/hkipo_market_data_coverage_2015_2026.csv`](../reports/hkipo_market_data_coverage_2015_2026.csv). The combined panel currently runs from 2015-01-09 through 2026-10-26; 2026 is a partial calendar year.

## Files

- `data/processed/hkex_ipos_2015_2026.csv`: combined HKEX listing master.
- `data/processed/hkipo_daily_bars_2015_2026_60sessions.csv`: daily bars, up to 60 listed sessions per event.
- `data/processed/hkipo_event_panel_2015_2026_60sessions.csv`: feature panel for historical evaluation.
- `data/processed/hkipo_daily_bars_2015_2020_60sessions_failures.csv`: 66 older listing events for which the Yahoo Finance query returned empty history.

## Coverage and data-quality limits

In 2015–2020, 66 of 651 master-list events (10.1%) have no bars in this retrieval. Missingness is not random by assumption and could create survivorship or availability bias. The combined panel has 557 OHLC-inconsistent rows (0.85%); these are flagged in `ohlc_consistent` and should be inspected or excluded under a predeclared rule before a final run. Yahoo Finance daily bars remain exploratory data, not an official exchange feed. HKEX listing reports provide the IPO event universe; they do not validate the Yahoo price history.

The data extension increases the event pool but does not itself establish higher strategy returns. Previously inspected test years remain retrospective. For the next experiment, preserve the already declared test years and compare (a) the previous 2021 training start with (b) 2015 training start, using the same event sets, model configuration, fees, seeds, and training interaction budget. With the expanded training pool, total compute grows because the budget remains proportional to the number of training IPOs; report that interaction count explicitly.

## Reproduction

The study script retains 2021 as its default training start so the previous experiment remains reproducible. To use the expanded history while keeping the declared 2023–2026 test years:

```powershell
python -m hkipo.ipo_feature_horizon_study `
  --panel data/processed/hkipo_event_panel_2015_2026_60sessions.csv `
  --output-dir reports/hkipo_dqn_ipo_features_horizons_train2015 `
  --train-start-year 2015 `
  --test-years 2023 2024 2025 2026
```

This is a new retrospective exploratory comparison, not an independent prospective validation.
