# Missing-history audit for the 2015–2020 IPO expansion

The listing master contains 651 HKEX IPOs. The Yahoo Finance retrieval returned empty history for 66 (10.1%) and bars for 585. Missing-history rates by listing year range from 8.1% to 14.9%. All 66 retrieval failures are recorded as `empty history`; that error does not establish whether the security stopped trading, changed symbols, or was omitted by the provider.

The observed missing rates are not uniform by IPO price band: 6.6% for prices at or below HKD 1, 9.3% for HKD 1–2, 15.3% for HKD 2–5, 12.0% for HKD 5–10, and 9.5% for HKD 10–100. Median fundraising is HKD 66.94m among missing events and HKD 62.50m among events with bars; median IPO prices are HKD 2.74 and HKD 2.00, respectively. These crude descriptive comparisons do not rule out selection bias; current-status, symbol-change, industry, and issuer-survival effects have not been resolved.

The expanded-history experiment should therefore be interpreted as adding the **retrievable** 2015–2020 IPOs, not the complete historical IPO universe. The event-level coverage table is in [`../reports/hkipo_market_data_missingness_2015_2020.csv`](../reports/hkipo_market_data_missingness_2015_2020.csv). The raw failed-symbol list is [`../data/processed/hkipo_daily_bars_2015_2020_60sessions_failures.csv`](../data/processed/hkipo_daily_bars_2015_2020_60sessions_failures.csv).
