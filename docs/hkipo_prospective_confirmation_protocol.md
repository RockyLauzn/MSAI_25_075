# 港股 IPO 上市后交易：前瞻确认协议

## 冻结时间与模型

- 冻结日：2026-10-27；现有面板行情截止 2026-10-23，已进入全部历史比较的 IPO 不再计入前瞻确认。
- 仅评估上市日期不早于 2026-10-27、且不属于历史研究数据的 IPO。
- 冻结候选：Small 32、Medium 64×64；基线：FinRL 兼容 SB3 标准 DQN。各自参数、成员模型 SHA-256 和投票规则见 [`frozen_policy_manifest.json`](../reports/hkipo_dqn_candidate_seed_stability/frozen_policy_manifest.json)。
- 集成方式：每个 ensemble 成员独立输出离散动作，按多数票选目标持仓；平票选较低持仓。冻结后不重新训练、不改变状态、动作、成本或集成规则。

## 时序与每只 IPO 的观测

1. 只在第 t 个交易日收盘后生成信号；状态使用截至该收盘可见的 8 个变量（发行价溢价、1/5 日收益、日内高低范围、成交量变化、近 5 日波动率、上市日序号、当前仓位）。
2. 动作是目标持仓比例 `[0, 25, 50, 75, 100]%`。信号拟于下一交易日开盘执行；记录开盘参考价，但不把它当成保证成交的成交价。
3. 采用 25 bps 单边成本、整只 IPO 独立初始资金的探索性模拟；不接券商、不下单。与买入持有的比较统一从智能体首次可决策后的下一开盘起算。
4. 保存原始数据来源、获取时间、决策日、观测截止时间、模型版本与数据快照哈希。信号一经记录不回写；后续开盘与收盘数据只补入结果列。
5. 每只 IPO 达到完整 30 个交易日后，计算净收益、最大回撤、盈利标记、单边换手；未满 30 日只报告未成熟，不提前当作完整事件。

## 比较和停止规则

- 主对照：买入持有；算法对照：冻结的 FinRL-SB3 标准 DQN；DQN 候选之间比较风险/复杂度权衡。
- 主要统计单位是一只新 IPO，不是每日记录或 ensemble 成员。至少等 30 个新的、完整 IPO 事件，或连续观察 2 年后，才做首次正式总评；若达到 2 年但事件不足 30，只报告低样本描述统计，不宣称泛化。
- 报告所有冻结候选，不按前瞻结果删掉表现差的模型。若任何代码或成本参数改变，则提高新版本号并重新开始一条前瞻序列。
- bootstrap 以 IPO 事件为单位；另报告按上市月份/市场 regime 分组的敏感性。新事件数量少时明确标注区间宽和检验力不足。

## 文件流程

- 冻结模型及哈希：[`reports/hkipo_dqn_candidate_seed_stability/frozen_policy_manifest.json`](../reports/hkipo_dqn_candidate_seed_stability/frozen_policy_manifest.json)
- 决策日志空模板：[`data/templates/hkipo_prospective_signal_log_template.csv`](../data/templates/hkipo_prospective_signal_log_template.csv)
- 逐日生成/追加冻结策略研究信号（仅候选决策，无下单）：`python -m hkipo.prospective_signals --panel <更新后的事件面板.csv> --as-of YYYY-MM-DD`
- 事件满 30 个交易日后计算净值结果：`python -m hkipo.score_prospective_events --panel <含完整后续行情的事件面板.csv>`；结果写入 `reports/hkipo_dqn_candidate_seed_stability/prospective_outcomes.csv`，信号文件保持只追加。
- 更新未来 IPO 的 HKEX 主表和日线 CSV 后，沿用现有 `import_hkex_reports`、`download_yfinance_bars`、`build_panel` 流程构建数据；核对事件上市日不早于冻结日，再按以上时序记录结果。

首次试运行使用当前历史面板（行情截至 2026-10-23，冻结日期从 2026-10-27 起），符合条件的新 IPO 为 0，因而没有生成信号或事件结果。这是预期的保护行为，不会把旧 IPO 当作前瞻样本。

## 边界

这是一项前瞻研究记录协议，不是实盘策略。当前仍使用 Yahoo Finance 探索性日线，开盘参考价、成交量和比例成本无法复原真实整手成交、滑点或市场冲击。若未来上市事件无法及时获取点时数据，应保留缺失记录，不得用后来的信息回填决策时状态。
