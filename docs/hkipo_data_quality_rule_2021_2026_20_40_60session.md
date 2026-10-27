# 2021–2026 horizon-specific data-quality rules for 20/40/60 sessions

**Rule recorded before the horizon clean-panel builds and model reruns: 2026-10-27.** Eligibility is determined only by data coverage and the existing OHLC-consistency flag, never by strategy performance.

## Fixed rule

For each horizon \(h \in \{20,40,60\}\), independently:

1. Keep the complete 2021–2026 HKEX listing master as the coverage denominator.
2. Exclude and report an IPO with no retrieved daily bars; do not infer or impute its price path.
3. Require at least \(h\) consecutive post-listing sessions, with `listed_day` exactly 1 through \(h\). Exclude shorter histories without filling missing rows.
4. Use exactly the first \(h\) sessions. If any of those rows has `ohlc_consistent=false`, exclude the **whole IPO event** for that horizon. Do not alter OHLC values, drop isolated bad rows, or carry prices forward.
5. Create a separate clean panel and exclusion report for each horizon. Use the same horizon-specific set of IPOs for every policy, training fold, test fold, and cost scenario for that horizon. The eligible set may differ across horizons; comparisons across horizons are descriptive and must report their different sample sizes.
6. Keep the 2021 training start; test on 2023–2026 with expanding chronological training, seeds 0–9, 25 bps per-side training cost, 10/25/50 bps per-side evaluation costs, and the already used fixed interaction budget and model configurations. Do not use 2015–2020 data.

The primary 30-session rule remains separately recorded in [`hkipo_data_quality_rule_2021_2026_30session.md`](hkipo_data_quality_rule_2021_2026_30session.md). These rules extend the horizon-specific complete-case logic to the additional windows before seeing their rerun results.

## Reproduction

Run once for each horizon and corresponding file name:

```powershell
python -m hkipo.quality_gate_horizon_panel `
  --panel data/processed/hkipo_event_panel_2021_2026_60sessions.csv `
  --ipos data/processed/hkex_ipos_2021_2026.csv `
  --horizon 20 `
  --output data/processed/hkipo_event_panel_2021_2026_20session_ohlc_clean.csv `
  --excluded reports/hkipo_20session_quality_exclusions_2021_2026.csv `
  --quality-report reports/hkipo_20session_quality_report_2021_2026.json
```

Substitute `40` or `60` consistently in the horizon and file names for the other panels. Each generated JSON captures the raw OHLC anomaly count, its allocation inside/outside the chosen horizon, event exclusions, and year-level sample counts.
