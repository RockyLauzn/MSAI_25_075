# 港股 IPO 资金模拟器：2021–2025 探索结果

## 环境范围

`src/hkipo/subscription_simulator.py` 实现离散事件现金账户，可重放“跳过”或“现金申购一手”。它按申请、配发/退款和上市日卖出处理账户现金、冻结资金、期望获配股数、费用及净值。配发用官方一手档位的结果组做确定性期望，不抽取单个账户的真实随机中签路径。

结果只用于核对环境账务流程和暴露假设敏感性。当前不是可实现策略回测：严格现金条款情景按已核验最高申购价估算申请现金；266 个申购截止日现均有官方日期，但不同渠道费用与实际扣款/退款到账时间仍不完整；开盘价格来自 Yahoo/yfinance，未建模滑点、盘口、停牌或成交队列；期望获配股数可以是小数。

## 基准回放

| 情景 | 初始现金 | 资金占用/费用假设 | 申购数 | 因现金不足跳过 | 期末净值 | 模拟净收益 | 最大回撤 |
|---|---:|---|---:|---:|---:|---:|---:|
| 全部可负担即申购 | HK$100,000 | 1.0 倍最高申购价款 | 266 | 0 | HK$182,219 | +HK$82,219（+82.2%） | -22.4% |
| 全部可负担即申购 | HK$20,000 | 1.0 倍最高申购价款 | 250 | 16 | HK$117,348 | +HK$97,348（+486.7%） | -29.4% |
| 全部可负担即申购 | HK$100,000 | 每次另加 HK$50 固定申请费 | 266 | 0 | HK$168,919 | +HK$68,919（+68.9%） | -24.8% |
| 全部可负担即申购 | HK$100,000 | 冻结资金年化机会成本 5% | 266 | 0 | HK$181,190 | +HK$81,190（+81.2%） | -22.7% |
| 全部可负担即申购 | HK$100,000 | 1.2 倍最高申购价款 | 266 | 0 | HK$182,219 | +HK$82,219（+82.2%） | -22.4% |
| 全部跳过 | HK$100,000 | — | 0 | 0 | HK$100,000 | HK$0 | 0% |

20,000 港元情景出现比 100,000 港元更高的收益比例与绝对收益，原因是有限现金改变了事件选择与再投资路径；这不是小资金具有优势的证据。16 次被跳过的事件改变了组合路径，最终结果对申购时间、资金占用假设及少数事件高度敏感。1.2 倍情景在 100,000 港元下没有改变可申购事件集合。

## 严格现金条款情景

当前 266/266 个 IPO 均有官方最高申购价条款，按最高申购价 × 最低申购股数及估算 IPO 费用计算申请资金。HK$100,000 初始现金下，266 次均申购，期末净值为 HK$182,219，净收益 HK$82,219（+82.2%），最大回撤 -22.4%；预计 IPO 费用 HK$4,923，卖出费用 HK$1,251，无条款缺失或现金不足跳过。申购截止日现已从 HKEX 官方公告或招股书时间表核实 266/266 个，年份推断与情景补值均为 0。逐年覆盖及补查来源见 [`hkipo_application_close_dates_audit.md`](hkipo_application_close_dates_audit.md)。复现命令：

```powershell
python -m hkipo.subscription_simulator --events reports/hkipo_simulator_data_2021_2025/verified_events_timetable_augmented.csv --output-dir reports/hkipo_simulator_2021_2025_verified_terms_timetable_final --cash-terms data/processed/hkex_ipo_cash_terms_2021_2025.csv --require-verified-cash-terms --initial-cash 100000
```

基准情景的年度事件净损益汇总：

| 上市年 | 事件数 | 期望净损益合计 | 单事件中位数 |
|---|---:|---:|---:|
| 2021 | 49 | -HK$12,847 | -HK$16.83 |
| 2022 | 37 | +HK$36,033 | -HK$19.03 |
| 2023 | 38 | +HK$54,620 | -HK$41.67 |
| 2024 | 42 | -HK$2,749 | -HK$0.91 |
| 2025 | 100 | +HK$7,162 | +HK$2.16 |

基准回放中，`02147`（2023）和 `06610`（2022）分别贡献约 HK$56,328 和 HK$41,875 的期望净损益，两者合计超过全期净收益。排除这两只后，全期结果转负。大额亏损也集中在少数 IPO。正收益因此不能解释为普遍、稳定的单只 IPO 优势。

## 可复现输出

- 输入事件与结果分组：[`verified_events.csv`](../reports/hkipo_simulator_data_2021_2025/verified_events.csv)、[`one_lot_outcomes.csv`](../reports/hkipo_simulator_data_2021_2025/one_lot_outcomes.csv)
- 申购时点候选特征（结果标签独立存放）：[`candidate_features.csv`](../reports/hkipo_point_in_time_features_2021_2025_timetable_final/candidate_features.csv)、[`candidate_labels_separate.csv`](../reports/hkipo_point_in_time_features_2021_2025_timetable_final/candidate_labels_separate.csv)
- 覆盖与排除：[`coverage_by_year.csv`](../reports/hkipo_simulator_data_2021_2025/coverage_by_year.csv)、[`excluded_verified_events.csv`](../reports/hkipo_simulator_data_2021_2025/excluded_verified_events.csv)
- 严格最高价/官方截止日重放：`reports/hkipo_simulator_2021_2025_verified_terms_timetable_final/`；对应资金、费用和机会成本敏感性输出位于同名前缀的 `_cash20k`、`_fee50`、`_opp5`、`_lock120`、`_5pct`、`_skip` 目录。

命令行可运行 `python -m hkipo.subscription_simulator --output-dir reports/hkipo_simulator_2021_2025_always --policy always --initial-cash 100000`。敏感性示例可调整 `--cash-lock-multiplier`、`--fixed-application-fee`、`--fallback-lock-days`、`--annual-opportunity-rate`。

## 当前结论与 DRL 闸门

模拟器已具备最小资金重放能力，最高申购价条款覆盖为 266/266，其中 144 条由官方 PDF 正文核验。申购截止日期现为 266/266 官方核验，年份推断和情景补值均为 0；逐年来源见 [`hkipo_cash_timeline_coverage_2021_2025.csv`](../reports/hkipo_cash_timeline_coverage_2021_2025.csv)，补查依据见 [`hkipo_application_close_dates_audit.md`](hkipo_application_close_dates_audit.md)。申购款真实扣留/退款到账时点和价格执行仍部分依赖代理；配发期望值路径也不等于一个真实账户。申购时点候选集现包含 prospectus date、首位 sponsor 和决策日前已上市 IPO 的 sponsor 历史结果；266 个事件中 211 个具备历史 sponsor 记录。它仍不是完整状态：Sponsor 名称别名未完全统一，也缺行业、发行价格区间等逐字段发布时间记录。因此当前点时环境还不足以支持公平、可信的 DRL 样本外比较。

在完成日期核验和模拟器对账后，已运行小样本 DQN 探索性比较；其 2025 留出表现对随机种子敏感，验证集选出的种子仅申购一只，结果未优于全部申购或简单历史保荐人规则。该试跑不是可靠泛化结论。下一阶段仍需补足特征可得时间、券商收费/资金释放依据，并在多个滚动测试年份重复比较。完整结果见 [`hkipo_dqn_pilot_results_2021_2025.md`](hkipo_dqn_pilot_results_2021_2025.md)。
