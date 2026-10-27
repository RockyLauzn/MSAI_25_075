# 港股 IPO：FinRL 兼容标准 DQN 基线

## 基线定义

FinRL 是 DRL 框架，不对应唯一交易策略。此处明确将基线实现为 FinRL 文档所支持的 **Stable-Baselines3 标准 DQN**，在我们自己的港股 IPO 环境训练；并非未修改的 FinRL 通用股票环境，也不是韩国论文确切 FinRL 参数的复现。FinRL 教程将 DQN 列为可用 DRL 算法，并基于 Gym 风格环境与 SB3/Stable Baselines 组织训练。[FinRL 官方教程](https://github.com/AI4Finance-Foundation/FinRL/blob/master/docs/source/tutorial/Introduction/SingleStockTrading.rst)

为了进行同口径比较，FinRL-SB3 DQN 和自定义百分比动作 DQN 共用 8 个状态特征、5 个目标持仓动作、单 IPO 30 日 episode、收盘后观察/次日开盘调仓/次日收盘计价、对数净值奖励、训练/验证/测试年份切分和成本情景。SB3 算法与 DQN 网络配置遵循其公开的 DQN 接口默认值，显式固定 MLP 为 `[64, 64]`。[SB3 DQN 参数文档](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)

| 参数 | 设置 |
|---|---:|
| Policy | MlpPolicy |
| MLP | 64, 64 |
| Learning rate | 0.0001 |
| Replay buffer | 1,000,000 |
| Learning starts | 100（SB3 2.9.2 默认值） |
| Batch size | 32 |
| Gamma | 0.99 |
| Train frequency / gradient steps | 4 / 1 |
| Target update interval | 10,000 steps |
| Exploration | 1.0 → 0.05，前 10% 训练步数线性衰减 |
| 每种子环境步 | 80,388（与自定义模型同交互预算） |
| 随机种子 | 3 |

自定义 DQN 与 SB3 标准 DQN 的 replay 更新频率并不相同：前者每 32 步更新一次、batch 64；SB3 按标准配置每 4 步更新一次、batch 32。故这是**同市场交互预算下的标准配置对照**，不是严格控制梯度更新次数的网络结构消融，不能把胜负归因于某一个组件。

## 25 bps 留出集比较

均值按 IPO 事件计算，3 个随机种子先在每只 IPO 内取平均。DQN-P 这里指本项目的自定义百分比动作 DQN 原型，不代表完全复现韩国论文。

| 留出期 | 策略 | 平均净收益 | 中位净收益 | 盈利事件占比 | 平均最大回撤 | 单边换手 |
|---|---|---:|---:|---:|---:|---:|
| 2025（108 IPO） | 自定义百分比 DQN | 3.68% | 0.05% | 50.9% | 10.29% | 6.19× |
|  | FinRL-SB3 标准 DQN | 2.82% | -0.21% | 47.5% | 11.28% | 7.57× |
| 2026 年内（104 IPO） | 自定义百分比 DQN | 1.61% | -4.10% | 41.0% | 14.92% | 6.42× |
|  | FinRL-SB3 标准 DQN | 1.71% | -4.65% | 36.9% | 16.88% | 7.74× |

### IPO 配对 Bootstrap

10,000 次 IPO 事件配对 bootstrap，正的回撤差表示自定义 DQN 回撤较低。

| 留出期 | 自定义 DQN - FinRL DQN 平均收益差 | 95% CI | FinRL DQN - 自定义 DQN 平均最大回撤 | 95% CI |
|---|---:|---:|---:|---:|
| 2025 | +0.86 个百分点 | [-0.59, +2.28] | +0.99 个百分点 | [+0.34, +1.66] |
| 2026 年内 | -0.09 个百分点 | [-2.45, +2.10] | +1.97 个百分点 | [+0.88, +3.01] |

两个留出年份的平均收益差置信区间都跨 0，没有证据表明自定义 DQN 的收益高于标准 SB3 DQN。它的平均最大回撤在两个年份均较低，事件配对区间未跨 0；相对标准 DQN 的回撤改善幅度较小（约 1–2 个百分点），弱于此前与买入持有比较所得的回撤差。

## 成本敏感性

DQN 只在 25 bps 单边成本下训练，再固定模型评估不同成本。下表展示留出集净收益：

| 留出期 | 策略 | 10 bps | 25 bps | 50 bps |
|---|---|---:|---:|---:|
| 2025 | 自定义百分比 DQN | 4.65% | 3.68% | 2.11% |
|  | FinRL-SB3 标准 DQN | 4.02% | 2.82% | 0.92% |
| 2026 年内 | 自定义百分比 DQN | 2.57% | 1.61% | -0.00% |
|  | FinRL-SB3 标准 DQN | 2.89% | 1.71% | -0.23% |

成本升高时两种 DQN 的收益都下降。自定义 DQN 在 2025 的点估计较高、2026 与基线接近；这些配对收益差的 95% CI 都跨 0。回撤差的置信区间在两个测试年份和三档成本下都偏向自定义 DQN，但需要留意样本年份与成本模型限制。

## 结论与后续

现有结果显示，自定义 DQN 原型没有显著优于 FinRL-SB3 标准 DQN 的平均收益；较低回撤的迹象在事件层面存在，但差距较小。下一步如要归因于百分比动作或 DQN-P 特有改动，应固定相同梯度更新预算，分别做标准 DQN、Double DQN、自定义归一化/初始化和 dropout 的消融。要复现韩国论文基线，还需取得论文全文并核实其 FinRL 算法版本与参数。

## 文件

- 实现：[`src/hkipo/finrl_dqn_baseline.py`](../src/hkipo/finrl_dqn_baseline.py)
- 完整参数：[`finrl_dqn_config.json`](../reports/hkipo_dqn_aftermarket/finrl_dqn_config.json)
- 汇总：[`finrl_dqn_summary.csv`](../reports/hkipo_dqn_aftermarket/finrl_dqn_summary.csv)
- 逐 IPO 结果：[`finrl_dqn_per_event.csv`](../reports/hkipo_dqn_aftermarket/finrl_dqn_per_event.csv)
- 配对 Bootstrap：[`paired_bootstrap_custom_dqn_vs_finrl_dqn.csv`](../reports/hkipo_dqn_aftermarket/paired_bootstrap_custom_dqn_vs_finrl_dqn.csv)
- 模型：`reports/hkipo_dqn_aftermarket/finrl_dqn_models/`
