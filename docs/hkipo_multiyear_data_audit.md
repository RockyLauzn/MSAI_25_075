# 港股 IPO 多年份配发数据核验（2021–2025）

## 核验范围

核验对象是港交所主板 IPO 事件、官方配发结果公告、公告中的一手申购结果分组，以及 Yahoo Finance/yfinance 提供的上市首日 OHLC 价格代理。配发公告通过 HKEXnews 搜索发现；旧公告有 PDF，也有 HTML 多附件页面。发现程序现同时处理两类链接，并从 HTML 附件页优先提取 Cover/Summary PDF。

公告解析后逐事件核对公告摘要总人数/股数，并对一手档位各分组的申请人数、获配人数和分配比例进行重算。模拟器数据只保留摘要对账和一手结果档位比例都通过的事件；随后要求有首日开盘价且 IPO 主表发行价匹配。任何一项失败均不进入当前模拟器样本。

## 年度覆盖

| 年份 | IPO 事件 | 找到官方公告 | PDF/公告成功解析 | 摘要总数对账通过 | 一手档位比例对账通过 | 全部配发核验通过 | 有首日价格并进入模拟器 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 95 | 90 | 68 | 61 | 56 | 51 | 49 |
| 2022 | 75 | 73 | 42 | 40 | 38 | 37 | 37 |
| 2023 | 68 | 68 | 50 | 43 | 46 | 40 | 38 |
| 2024 | 67 | 67 | 61 | 56 | 49 | 46 | 42 |
| 2025 | 113 | 113 | 109 | 105 | 107 | 104 | 100 |
| **合计** | **418** | **411** | **330** | **305** | **296** | **278** | **266** |

“摘要对账”与“档位比例对账”是不同检查；某公告可能通过其中一项而未通过另一项。年度计数来自各年的官方公告 manifest、parse audit 和模拟器覆盖报告。411 份公告被找到，330 份完成解析，278 份满足两项配发核验，266 份另有可用首日价格且通过发行价匹配。

266 个模拟事件按年为 49、37、38、42、100。十个核验事件因缺上市日价格被排除，另有两个发行价与 IPO 主表不匹配。应以 [模拟器覆盖表](../reports/hkipo_simulator_data_2021_2025/coverage_by_year.csv) 和 [排除事件表](../reports/hkipo_simulator_data_2021_2025/excluded_verified_events.csv) 为逐事件明细。

## 数据修正与复现入口

- 旧年份公告链接包含 `.htm` 多附件页面；此前只识别 PDF 导致低覆盖。发现器和下载器已修正为提取 HTML 页面内的摘要 PDF。
- 解析器兼容旧版公告标题、旧式申请人数措辞和官方按档分组表格；保留严格对账状态，不以“解析成功”替代核验通过。
- 原始来源和解析输出：`data/processed/hkex_allotment_manifest_2021_2024.csv`、`data/processed/hkipo_allotments_2021_2024/parse_audit.csv`、`data/processed/hkipo_allotments_2025/`。
- 可重建模拟器输入：`python -m hkipo.prepare_subscription_sim_data --root . --output-dir reports/hkipo_simulator_data_2021_2025 --fallback-lock-days 5`。

## 进入策略学习前仍缺的核验

本轮完成了配发统计、最高申购价条款及部分申购时间表核验，但仍不等于历史申购资金时间线、经纪费率和可成交价格已完整验证。266 个事件均有官方最高申购价条款；96 个事件有精确的官方申购截止日期，另有 1 个时间表只提取到月日、年份推断后仍按情景日期处理，170 个事件仍按配发结果日减 5 个日历日生成决策日。首日价格来自 Yahoo/yfinance，开盘价只是代理。严格模拟器使用最高申购价与估算费用计算申请现金，但真实退款时间和经纪渠道成本仍未完整还原。

因此，当前数据集足以支持透明标注假设的确定性环境原型和敏感性探索，不足以产生可靠的可实现收益结论。若要做严谨 DRL 样本外比较，必须先补齐可用的申购前点时特征、历史资金需求与冻结时间，或将缺项明确限制为独立情景，并冻结 2025 测试集后再选模。

另对 SenseTime（00020）做了发行版本核对：12 月 7 日初版时间表的 12 月 10 日截止日属于后来暂停并退款的旧发行。港交所 12 月 13 日公告说明发行延期且申请款全额退款；最终重启发行以 12 月 20 日补充招股书为准，申请截止为 12 月 23 日。模拟器使用最终时间表；证据和旧/新版本对应关系见 [港交所退款公告](https://www1.hkexnews.hk/listedco/listconews/sehk/2021/1213/2021121300251.pdf)、[补充招股书](https://www1.hkexnews.hk/listedco/listconews/sehk/2021/1220/2021122000084.htm) 及 [`timetable_manual_overrides.csv`](../reports/hkipo_timetable_manual_overrides_2021_2025.csv)。

## 点时特征候选集

已新增 `python -m hkipo.build_point_in_time_features`，把 prospectus date、首位 sponsor 及先前已上市 IPO 的历史配发净收益整理成候选决策特征。历史 sponsor 统计仅纳入 `listing_date < decision_date` 的事件，因此不会用当前 IPO 的配发或上市结果。更新后的输出为 [`candidate_features.csv`](../reports/hkipo_point_in_time_features_2021_2025_timetable_augmented/candidate_features.csv)，未来标签单独保存在同目录的 `candidate_labels_separate.csv`。

266 个事件中 211 个有至少一笔可用的历史首位 sponsor 记录；158 个有三笔以上。2025 年测试候选事件中，92/100 有 sponsor 历史，85/100 有三笔以上。现已为所有 266 个事件核实精确招股截止日期；sponsor 名称变体尚未完全归一。特征候选集没有加入当前 IPO 最终发行价、配发结果或上市价格。申购截止日的来源与补查证据见 [`hkipo_application_close_dates_audit.md`](hkipo_application_close_dates_audit.md)。

这使“历史 sponsor 信息能否成为可用状态特征”可以进入下一步评估，但候选状态仍未纳入账户现金、冻结申购款及以最高申购价计算的资金需求。完整特征覆盖见 [`feature_coverage.json`](../reports/hkipo_point_in_time_features_2021_2025_timetable_augmented/feature_coverage.json)。

随后补查并逐步核验申请时最高发售价：目前 266 个模拟事件均已从官方申请表、配发公告或 HKEX PDF 正文恢复，144 个条款由官方 PDF 正文直接核验。逐年覆盖、来源优先级和资金模拟边界见 [`hkipo_application_cash_terms_audit.md`](hkipo_application_cash_terms_audit.md)。
