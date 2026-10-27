# Deep Reinforcement Learning for Quantitative Trading

## 当前论文范围：2021–2026 港股 IPO 上市后 DQN

研究问题：DQN 能否结合上市后行情与 IPO 特征，在港股 IPO 上市后前 20–60 个交易日动态管理仓位，并泛化到后续 IPO？主期限为 30 日，20/40/60 日为期限敏感性分析。收盘 t 观察、次日开盘调仓；行情基于 Yahoo Finance 日线，仅作历史探索，不是可执行交易策略。论文主样本限于 2021–2026，不使用 2015–2020 IPO 数据。2021–2026 30 日主分析先按预先记录的 OHLC 完整事件规则清洗；规则、覆盖与排除清单见 [`docs/hkipo_data_quality_rule_2021_2026_30session.md`](docs/hkipo_data_quality_rule_2021_2026_30session.md)。主范围与清洗后结果见 [`docs/hkipo_paper_scope_2021_2026.md`](docs/hkipo_paper_scope_2021_2026.md)。

清洗后 30 日主结果（逐年滚动、10 种子、事件级 bootstrap）见 [`docs/hkipo_paper_scope_2021_2026.md`](docs/hkipo_paper_scope_2021_2026.md) 与 [`reports/hkipo_dqn_2021_2026_30session_clean_seed10/`](reports/hkipo_dqn_2021_2026_30session_clean_seed10/)。20/40/60 日独立清洗、重训后的成本敏感性与逐年结果见 [`docs/hkipo_horizon_sensitivity_results_clean.md`](docs/hkipo_horizon_sensitivity_results_clean.md) 及各 horizon 的报告目录。较早的未清洗面板结果仍标作探索性，见 [`docs/hkipo_dqn_ipo_features_horizons_results.md`](docs/hkipo_dqn_ipo_features_horizons_results.md)。此前仅行情、30 日 DQN 结果见 [`docs/hkipo_dqn_aftermarket_pilot.md`](docs/hkipo_dqn_aftermarket_pilot.md)。

20/40/60 日各自的质量规则和清洁面板重跑见 [`docs/hkipo_data_quality_rule_2021_2026_20_40_60session.md`](docs/hkipo_data_quality_rule_2021_2026_20_40_60session.md)。

相关工作与创新性措辞（含 2026 年韩国 IPO DQN-P 论文）记录于 [`docs/hkipo_literature_novelty_note.md`](docs/hkipo_literature_novelty_note.md)。

英文论文初稿（30 日为主分析，含质量闸门后的 20/40/60 日敏感性；2015–2020 数据排除）见 [`docs/manuscript_hkipo_drl_cross_ipo_draft.md`](docs/manuscript_hkipo_drl_cross_ipo_draft.md)。

2015–2020 行情扩展及 2015 起训对照保留为独立历史探索，不属于当前论文样本或结论。相关记录仍可见 [`docs/hkipo_train_start_history_comparison.md`](docs/hkipo_train_start_history_comparison.md)。

扩充样本的 66 个无行情 IPO 缺失率及其与上市年份、发行价档的分布见 [`docs/hkipo_market_data_missingness_audit_2015_2020.md`](docs/hkipo_market_data_missingness_audit_2015_2020.md)。

此前 2015 vs 2021 训练起点的十种子对照见 [`docs/hkipo_dqn_training_seed_stability_10seeds.md`](docs/hkipo_dqn_training_seed_stability_10seeds.md)，仅作为独立历史探索；论文主结果只采用 2021 起训样本。

IPO 配对 bootstrap 与 10/25/50 bps 单边成本敏感性结果见 [`docs/hkipo_dqn_aftermarket_robustness.md`](docs/hkipo_dqn_aftermarket_robustness.md)，复算入口为 `python -m hkipo.analyze_aftermarket_robustness`。

FinRL 兼容的 Stable-Baselines3 标准 DQN 基线、参数、留出比较和配对 bootstrap 见 [`docs/hkipo_finrl_dqn_baseline.md`](docs/hkipo_finrl_dqn_baseline.md)。安装可选依赖：`python -m pip install -e ".[finrl-baseline]"`；训练入口：`python -m hkipo.finrl_dqn_baseline`。

此前的 30 日、仅行情特征滚动结果见 [`docs/hkipo_dqn_rolling_aftermarket_results.md`](docs/hkipo_dqn_rolling_aftermarket_results.md)。该旧配置训练仅 6 轮而未同步 target network，已被当前 IPO 特征与 20–60 日研究取代。

当前研究的核心约束是 IPO 级训练样本规模与策略容量；分层嵌套训练样本、跨年份留出和网络规模学习曲线见 [`docs/hkipo_dqn_sample_complexity.md`](docs/hkipo_dqn_sample_complexity.md)，实验入口为 `python -m hkipo.dqn_sample_complexity`。

首轮样本规模 × 网络复杂度跨 IPO 泛化结果见 [`docs/hkipo_dqn_sample_complexity_results.md`](docs/hkipo_dqn_sample_complexity_results.md)，IPO 事件配对 bootstrap 的复算入口为 `python -m hkipo.analyze_sample_complexity`。

等总交互步数的补充曲线见 [`docs/hkipo_dqn_fixed_budget_results.md`](docs/hkipo_dqn_fixed_budget_results.md)，用于分离新增 IPO 多样性与训练更新次数的影响。

2024 验证候选筛选与 13 个种子的稳定性检查见 [`docs/hkipo_dqn_candidate_stability_results.md`](docs/hkipo_dqn_candidate_stability_results.md)。2025/2026 已用于早期探索，仅作为回顾性稳健性检查；未来新上市 IPO 才是独立确认集。

此前 30 日、仅行情状态的前瞻信号器和模型冻结协议保留为旧版本记录；它不包含本研究新增的 IPO 特征和 20/40/60 日期限。

旧版 30 日研究信号流程：`python -m hkipo.prospective_signals --panel <更新后的事件面板.csv> --as-of YYYY-MM-DD`；满 30 个交易日后评分：`python -m hkipo.score_prospective_events --panel <含完整后续行情的事件面板.csv>`。首次试运行冻结日之后的 IPO 事件数为 0，未生成信号。

## 港股 IPO 打新历史研究分支

本课题的核心是用深度强化学习学习量化交易策略；港股 IPO 打新是当前研究环境。智能体按真实时间顺序面对 IPO 申购机会，在现金、已冻结申购资金和风险预算约束下，决定跳过或申购、申购档位及是否融资，并以扣除申购、交易和资金成本后的组合财富变化获得奖励。单只 IPO 的净收益估算用于构建和核验环境奖励，不是课题终点。

状态只使用申购决策时已经公开的信息及账户资金状态；配发结果、首日价格和实际费用在模拟中作为后续转移与奖励。监督学习和固定规则仅作为 DRL 的基准比较。当前先验证历史配发、价格与成本数据，再构建离线事件驱动环境并比较 DRL 策略。研究阶段不接券商、不自动下单，也不预设策略可以盈利。此前的次新股二级市场和 SPY 期权实验保留为历史研究。

以下研究路线和协议记录了此前的打新现金分配方向，现作为可复用的数据与模拟器工作保留。

## 研究路线

1. 已完成 2021–2025 官方配发公告、一手结果档位及申购截止日期的多年份核验；继续核实历史保证金和渠道费用。
2. 已实现最小事件驱动现金模拟器，重放跳过/一手申购，记录资金冻结、配发、退款、上市卖出和现金变化。
3. 补齐带可得时间的申购前状态数据，固定无前视数据切分和可比基准，再进入 DRL。
4. 已完成一个小样本 DQN 探索性试跑，并增加三个年份滚动留出；验证集择优与固定训练终点的五种子多数投票均未显示稳定的 DRL 增益。可信资金释放/渠道费用和更完整的点时状态仍待补齐。

## 第一版范围

- 决策对象：每个 IPO 申购机会的申请/手数/融资；目标是组合层面净财富增长。
- 股票池：HKEX Main Board 新上市事件；应筛除不适用的证券类型并审计重复上市事件。
- 特征截止：申购截止前。最终定价、公开超购和中签率等事后信息不得进入申购前模型。
- 验证：按上市日期滚动切分，整个 IPO 事件只能落在一个数据分区。

香港 IPO 稳定价格操作不保证股价有底；是否有绿鞋及实际操作应从招股书和公告取得。首日行情回报不能代表申购收益：模型还需要实际配发数据和申购成本。

## 数据与代码

- 数据字段及时点契约：[`docs/hkipo_data_contract.md`](docs/hkipo_data_contract.md)
- 行情源局限与可选验证：[`docs/hkipo_reliable_market_data.md`](docs/hkipo_reliable_market_data.md)
- 研究设计和验证计划：[`docs/hkipo_research_plan.md`](docs/hkipo_research_plan.md)
- DRL 环境、状态、动作、奖励和验证设计：[`docs/hkipo_drl_framework.md`](docs/hkipo_drl_framework.md)
- DQN 探索性训练与 2025 留出比较：[`docs/hkipo_dqn_pilot_results_2021_2025.md`](docs/hkipo_dqn_pilot_results_2021_2025.md)
- DQN 多年份滚动留出与种子稳定性：[`docs/hkipo_dqn_rolling_evaluation.md`](docs/hkipo_dqn_rolling_evaluation.md)
- 完整 DRL 研究协议、数据闸门、样本外设计与结论边界：[`docs/hkipo_drl_research_protocol.md`](docs/hkipo_drl_research_protocol.md)
- 2021–2025 多年份官方配发核验和逐年覆盖：[`docs/hkipo_multiyear_data_audit.md`](docs/hkipo_multiyear_data_audit.md)
- 最小现金模拟器的探索结果、敏感性和局限：[`docs/hkipo_simulator_results_2021_2025.md`](docs/hkipo_simulator_results_2021_2025.md)
- 申购最高发售价条款的来源核验与逐年覆盖：[`docs/hkipo_application_cash_terms_audit.md`](docs/hkipo_application_cash_terms_audit.md)
- 申购最高发售价官方 PDF 正文逐条审计（144 个匹配）：[`reports/hkipo_max_price_pdf_source_audit_2021_2025.json`](reports/hkipo_max_price_pdf_source_audit_2021_2025.json)
- 5 个申请表/配发公告抽取差异的人工复核记录：[`reports/hkipo_cash_term_source_conflict_review_2021_2025.csv`](reports/hkipo_cash_term_source_conflict_review_2021_2025.csv)
- 申购截止日期逐年来源覆盖：[`reports/hkipo_cash_timeline_coverage_2021_2025.csv`](reports/hkipo_cash_timeline_coverage_2021_2025.csv)；招股书时间表原文提取：[`reports/hkipo_prospectus_timetable_dates_2021_2025.json`](reports/hkipo_prospectus_timetable_dates_2021_2025.json)
- 266 个事件的申购截止日核验方法与补查事件：[`docs/hkipo_application_close_dates_audit.md`](docs/hkipo_application_close_dates_audit.md)
- 招股时间表提取器与日期合并器：[`src/hkipo/discover_hkex_ipo_terms.py`](src/hkipo/discover_hkex_ipo_terms.py)、[`src/hkipo/apply_prospectus_timetable_dates.py`](src/hkipo/apply_prospectus_timetable_dates.py)
- SenseTime 旧发行退款与最终补充时间表的人工核验：[`reports/hkipo_timetable_manual_overrides_2021_2025.csv`](reports/hkipo_timetable_manual_overrides_2021_2025.csv)
- 2021–2025 HKEX 索引补充搜索结果及来源证据：[`reports/hkipo_web_search_cash_terms_2021_2025.json`](reports/hkipo_web_search_cash_terms_2021_2025.json)
- 官方 PDF 原文最高发售价逐项核验结果：[`reports/hkipo_indexed_cash_term_pdf_verification_2021_2025.json`](reports/hkipo_indexed_cash_term_pdf_verification_2021_2025.json)
- 合并替代检索后的完整 PDF 核验结果：[`reports/hkipo_indexed_cash_term_pdf_verification_2021_2025_combined.json`](reports/hkipo_indexed_cash_term_pdf_verification_2021_2025_combined.json)
- 替代检索尝试及未命中记录：[`reports/hkipo_web_search_cash_terms_alternates_all.json`](reports/hkipo_web_search_cash_terms_alternates_all.json)
- 首轮数据覆盖与基线结果：[`docs/hkipo_initial_results.md`](docs/hkipo_initial_results.md)
- 港股 IPO 打新研究计划：[`docs/hkipo_subscription_research.md`](docs/hkipo_subscription_research.md)
- 港股 IPO 打新首轮结果：[`docs/hkipo_subscription_initial_results.md`](docs/hkipo_subscription_initial_results.md)
- 港股 IPO 配发数据录入模板：[`data/templates/hkipo_allotment_tiers_template.csv`](data/templates/hkipo_allotment_tiers_template.csv) 与 [`data/templates/hkipo_allotment_events_template.csv`](data/templates/hkipo_allotment_events_template.csv)
- 配发结果组模板和解析样例：[`data/templates/hkipo_allotment_outcomes_template.csv`](data/templates/hkipo_allotment_outcomes_template.csv)、[`docs/hkipo_allotment_sample_02685.md`](docs/hkipo_allotment_sample_02685.md)
- 2025 HKEX 官方配发公告发现清单及解析覆盖限制：[`docs/hkipo_allotment_2025_parser_status.md`](docs/hkipo_allotment_2025_parser_status.md)
- 2025 港股 IPO 一手申购净收益初估与费用敏感性：[`docs/hkipo_subscription_net_estimate_2025.md`](docs/hkipo_subscription_net_estimate_2025.md)
- 旧 SPY 期权结果：[`docs/research_results.md`](docs/research_results.md)
- 新模块入口：`src/hkipo/`
- DQN 原型实现：[`src/hkipo/drl_cash_allocator.py`](src/hkipo/drl_cash_allocator.py)；安装可选依赖：`python -m pip install -e ".[drl]"`

## 当前探索流程（Yahoo/yfinance）

```powershell
python -m pip install -e ".[excel,marketdata]"
python -m hkipo.import_hkex_reports --start-year 2021 --end-year 2026 --output data/processed/hkex_ipos_2021_2026.csv
python -m hkipo.download_yfinance_bars --ipos data/processed/hkex_ipos_2021_2026.csv --output data/processed/hkipo_daily_bars_2021_2026.csv
python -m hkipo.build_panel --ipos data/processed/hkex_ipos_2021_2026.csv --bars data/processed/hkipo_daily_bars_2021_2026.csv --output data/processed/hkipo_event_panel_2021_2026.csv
python -m hkipo.baselines --panel data/processed/hkipo_event_panel_2021_2026.csv --output-dir reports/hkipo_baselines_2021_2026 --brokerage-per-side 0.0025
python -m hkipo.forecast --panel data/processed/hkipo_event_panel_2021_2026.csv --output-dir reports/hkipo_forecast_2021_2026
python -m hkipo.rule_backtest --panel data/processed/hkipo_event_panel_2021_2026.csv --ipos data/processed/hkex_ipos_2021_2026.csv --output-dir reports/hkipo_rule_backtest_2021_2026
python -m hkipo.subscription_study --panel data/processed/hkipo_event_panel_2021_2026.csv --output-dir reports/hkipo_subscription_2021_2026
```

上述 IPO 打新输出目前只研究首日行情结果，不代表获配后的可实现收益。次新股回测流程保留作对照；数据截至 2026-10-25，2026 是年内不完整样本。

## 可选：导入其他来源 CSV

```powershell
python -m hkipo.import_bars_csv --input data/raw/market_bars.csv --output data/processed/hkipo_daily_bars_external.csv --source-name "provider/product name" --column-map data/raw/column_map.json
```

这个入口可用于授权文件或其他供应商的 CSV；当前探索不要求购买行情。结果见 `reports/hkipo_rule_backtest_2021_2026/`。

IPO 基础记录来自 [HKEX New Listing Reports](https://www2.hkexnews.hk/New-Listings/New-Listing-Information/Main-Board?sc_lang=en)。当前行情来自 Yahoo Finance/yfinance，来源质量、缺失和 OHLC 异常均应结合输出报告判断。若以后需要评估更严谨的回测，再考虑 HKEX 授权数据或明确说明授权来源的供应商；行情源说明见 [HKEX Data Marketplace](https://data.hkex.com.hk/)。

原始和处理后的市场数据不纳入 Git。探索结果应保留数据源、覆盖范围与质量限制说明。
