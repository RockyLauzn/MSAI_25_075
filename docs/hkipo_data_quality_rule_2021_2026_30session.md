# 2021–2026 primary 30-session data-quality rule

**Rule recorded before the clean primary rerun: 2026-10-27.** It is based only on data availability and the OHLC consistency flag, not on strategy returns.

## Primary rule

1. The IPO universe is the 2021–2026 HKEX listing master. Keep all master events in the coverage denominator.
2. For events with no daily bars, do not impute or infer a price path; exclude them from return estimation and report them as unavailable.
3. For events with fewer than 30 post-listing daily rows, do not extend or fill the episode; exclude from the 30-session primary result.
4. For a complete episode, use only its first 30 listed sessions. If any row in those 30 has `ohlc_consistent=false` (high below open/close, or low above open/close), exclude the **entire IPO event** from the primary result. Do not repair OHLC values and do not carry forward prices.
5. Apply the same filtered event set to custom DQN, FinRL DQN, and every baseline, for both training and evaluation. Retain 2021 as the training start, use the existing 2023–2026 rolling test years, matched seeds, model settings, training budget, and fee protocol.

This event-level complete-case rule is deliberately conservative: an inconsistent OHLC bar could affect both the state (intraday range) and execution/marking prices. Excluding only the bad row would create a discontinuous 30-session path and alter the horizon.

## Initial coverage facts

The 2021–2026 master has 533 listings. The raw 60-session panel has 30,643 rows, of which 316 are OHLC-inconsistent across 169 events. Of these, 111 rows occur in sessions 1–30 and 205 occur in sessions 31–60. For the fixed 30-session primary analysis, 86 complete IPO episodes contain at least one inconsistent row in sessions 1–30; exclude those whole episodes. The 205 later-session anomalies are outside the primary horizon and are not used in the clean 30-session panel. The predeclared horizon-specific gates and clean reruns for 20/40/60 days are documented separately in [`hkipo_data_quality_rule_2021_2026_20_40_60session.md`](hkipo_data_quality_rule_2021_2026_20_40_60session.md). Thirteen IPOs have no bars, and nine additional 2026 events have fewer than 30 sessions at the panel cutoff. The resulting clean primary panel has 425 events (12,750 rows). Full-year counts and raw-versus-horizon anomaly counts are saved in the JSON report.

Rows with inconsistencies in sessions 31–60 do not enter the 30-session primary panel. The supplementary horizon-specific panels apply equivalent event-level complete-case rules at 20, 40, and 60 sessions.

## Reproduction

```powershell
python -m hkipo.quality_gate_30day_panel `
  --panel data/processed/hkipo_event_panel_2021_2026_60sessions.csv `
  --ipos data/processed/hkex_ipos_2021_2026.csv `
  --output data/processed/hkipo_event_panel_2021_2026_30session_ohlc_clean.csv `
  --excluded reports/hkipo_30session_quality_exclusions_2021_2026.csv `
  --quality-report reports/hkipo_30session_quality_report_2021_2026.json
```

The exclusion list and JSON report are part of the research record. The previous raw-panel runs remain exploratory and are not the clean primary result.
