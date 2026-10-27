# Deep Reinforcement Learning for Quantitative Trading

当前研究模块：SPY 期权波动率预测与策略回测。后续可在此基础上扩展到强化学习交易决策。

一个从美股期权研究开始的小型项目。第一阶段不自动交易，也不直接让模型预测买卖方向：先用 SPY 日线预测未来约一个月（20 个交易日）的实现波动率，再用历史期权链检验这个信号是否有可交易价值。20 日目标更适合和约一个月期权的隐含波动率做后续比较。

## 研究步骤

1. 建立干净的标的日线数据和期权链数据。
2. 用过去收益率构造波动率、动量等特征；目标只使用预测日之后的收益。
3. 用按时间顺序划分的训练、验证和测试区间，与简单历史波动率基准比较。
4. 若波动率预测有稳定的样本外增益，再用当时可见的期权报价构造有限风险策略回测。
5. 回测计入买卖价差、手续费、滑点和资金占用；之后再考虑纸面交易。

## 数据格式

首个数据文件为 `data/raw/spy_daily.csv`，可手工提供，也可用 yfinance 下载 Yahoo Finance 的 SPY 日线调整收盘数据：

| 列 | 含义 |
| --- | --- |
| `date` | 交易日 |
| `adj_close` | 调整收盘价，正数 |

历史期权报价使用规范化 CSV，字段见 `docs/data_contract.md`，空白模板在 `data/templates/spy_option_quotes_template.csv`。必须保留每个历史时点实际可见的合约和报价；不能用今天的期权链代替历史链。

## 下载和构造数据集

```powershell
python -m pip install -e ".[yahoo]"
python -m optionlab.download_spy --start 2010-01-01 --output data/raw/spy_daily.csv
python -m optionlab.dataset --input data/raw/spy_daily.csv --output data/processed/spy_volatility.csv --horizon 20
python -m optionlab.forecast --input data/processed/spy_volatility.csv --prices data/raw/spy_daily.csv --output-dir reports
python -m optionlab.straddle_backtest --predictions reports/volatility_predictions.csv --quotes data/raw/spy_option_quotes.csv
```

本项目也接入了一个公开的 SPY 日终期权链（2010–2025）。安装 Parquet 依赖并下载年度文件后，可用以下命令构造多年度前推预测和 EOD 回测：

```powershell
python -m pip install -e ".[parquet]"
python -m optionlab.download_public_eod --start-year 2010 --end-year 2025
python -m optionlab.dataset --input data/raw/spy_daily.csv --output data/processed/spy_volatility_eod.csv --horizon 20 --target-offset 2
$optionFiles = 2010..2025 | ForEach-Object { "data/raw/spy_options_$_.parquet" }
python -m optionlab.import_public_eod --options $optionFiles[0] --additional-options $optionFiles[1..15] --underlying data/raw/spy_underlying_prices.parquet --output data/processed/spy_options_eod_2010_2025.parquet
python -m optionlab.forecast --input data/processed/spy_volatility_eod.csv --output-dir reports/walkforward --horizon 20 --target-offset 2 --walk-forward-start-year 2015
python -m optionlab.build_option_features --quotes data/processed/spy_options_eod_2010_2025.parquet --output data/processed/spy_atm_iv_30.csv
python -m optionlab.dataset --input data/raw/spy_daily.csv --output data/processed/spy_volatility_with_iv.csv --horizon 20 --target-offset 2 --option-features data/processed/spy_atm_iv_30.csv
python -m optionlab.forecast --input data/processed/spy_volatility_with_iv.csv --output-dir reports/iv_model --horizon 20 --target-offset 2 --walk-forward-start-year 2015
python -m optionlab.straddle_backtest --predictions reports/iv_model/volatility_predictions.csv --quotes data/processed/spy_options_eod_2010_2025.parquet --output-dir reports/iv_model/straddle --quote-mode eod
python -m optionlab.forecast --input data/processed/spy_volatility_with_iv.csv --output-dir reports/variance_premium --horizon 20 --target-offset 2 --walk-forward-start-year 2015 --predict-variance-premium
python -m optionlab.straddle_backtest --predictions reports/variance_premium/volatility_predictions.csv --quotes data/processed/spy_options_eod_2010_2025.parquet --output-dir reports/variance_premium/straddle --quote-mode eod
python -m optionlab.build_oi_features --quotes data/processed/spy_options_eod_2010_2025.parquet --output data/processed/spy_oi_features.csv
python -c "import pandas as pd; iv=pd.read_csv('data/processed/spy_atm_iv_30.csv',parse_dates=['date'])[['date','atm_iv_30']]; oi=pd.read_csv('data/processed/spy_oi_features.csv',parse_dates=['date']); iv.merge(oi,on='date').to_csv('data/processed/spy_option_iv_oi_features.csv',index=False)"
python -m optionlab.dataset --input data/raw/spy_daily.csv --output data/processed/spy_volatility_with_iv_oi.csv --horizon 20 --target-offset 2 --option-features data/processed/spy_option_iv_oi_features.csv
python -m optionlab.forecast --input data/processed/spy_volatility_with_iv_oi.csv --output-dir reports/oi_model --horizon 20 --target-offset 2 --walk-forward-start-year 2015
python -m optionlab.analyze_friday_oi --quotes data/processed/spy_options_eod_2010_2025.parquet --prices data/raw/spy_daily.csv --output-dir reports/friday_oi
```

Yahoo Finance 的下载由社区维护的 [yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) 工具完成；它不是 Yahoo 官方 SDK，项目说明数据用于研究/教育并要求遵守 Yahoo 的数据条款。下载脚本显式关闭 OHLC 自动调整，并只输出 `Adj Close`。该序列用于波动率研究；期权真实成交模拟仍需未调整的标的价格与历史期权报价。

数据集输出包含特征 `rv_5`, `rv_20`, `rv_60`, `rv_120`, `mom_5`, `mom_20`, `mom_60` 和目标 `target_rv_20`。波动率使用过去/未来收益平方均值的平方根计算并按 252 个交易日年化。评估程序以时间顺序切分训练、验证和测试段，在边界清除 20 个交易日以避免目标窗口跨段；比较 Ridge 回归和 `rv_20` 持续性基准，输出测试集预测、MAE、RMSE、方差 QLIKE 和相对基准 OOS R²。逐日 20 日目标窗口彼此重叠，因此报告也单列每隔 20 个交易日抽样的指标，减少重叠窗口对样本量观感的影响。由于目标需要未来 20 个交易日，带标签数据会比最新行情晚约 20 个交易日；传入 `--prices` 后脚本另用所有已知标签重训并生成一条最新未标注预测。报告分别记录测试段评估和最新预测日期，不能混为一谈。

策略原型只测试**最大亏损限定在权利金的买入平值跨式**：当模型预测波动率高于配对平值 call/put 平均 IV 至少 10% 时入场，选择约 30 天到期合约，持有 20 个交易日。信号在收盘后形成，入场使用下一交易日 09:30–09:45 ET 的报价，平仓使用第 20 个交易日 15:45–16:00 ET 的报价；按卖价买入、买价卖出，默认每张每次成交费用 0.65 美元，最多持有一组。它要求历史报价文件在上述时段有可用的 bid/ask、报价大小、标的价、IV 及合约交割资料；原型只接受乘数 100、交割 100 股且无现金交割的标准 SPY 合约。默认费用和 10% 门槛只是可调研究参数，需按实际券商费用与验证区间确定；缺少对应时段报价的交易会跳过。该原型尚未模拟逐笔成交、部分成交、资金复利或公司行动，回测结果不能代表可实现收益。

公开样本是 4:00 PM ET 快照，不含盘中报价。专用导入器按日期补入 SPY 未调整收盘价，并将 `contract_id` 映射为合约标识；默认保留到期日距快照 0–40 天、行权价在标的价 ±50% 内的标准合约，以保留 20 日持仓退出报价。`--quote-mode eod` 假设当日收盘报价可以按 ask/bid 成交；真实成交不保证。为避免信号之后的未来收益泄漏，EOD 模式将信号 t 的目标从 t+2 开始计 20 个日收益区间，次日收盘买入，之后第 20 个完整区间收盘卖出。`--walk-forward-start-year` 会逐年预测，每年仅使用预测年之前已完整实现的标签；当前多年度样本从 2015 年开始预测。

## 历史期权数据源

数据源先保持可替换。当前可考察 Massive（原 Polygon）：其官方页面列出免费计划及付费计划；日聚合、逐笔成交、历史报价的可用历史长度和权限各不相同。页面当前显示 Options Starter 为 29 美元/月、Developer 为 79 美元/月、Advanced 为 199 美元/月；其中历史逐笔报价（quotes）标示为 Advanced 权限。价格、权限和数据授权可能变更，开始付费前应在供应商账户中确认。请见 [Options 产品与计划](https://massive.com/options)、[历史报价文件说明](https://massive.com/docs/flat-files/options/quotes) 和 [日聚合说明](https://massive.com/docs/flat-files/options/day-aggregates)。

因此先用公开/自有 CSV 和 Parquet 完成流程，不把付费行情当作已接入。免费或日线聚合可用于探索，不能替代可按历史时点还原的盘中报价；收盘快照回测只作为粗略筛选。

当前免费样本来自 [options-dataset-hist](https://github.com/anahatsingh-ui/options-dataset-hist)，这是 Philipp D. Dubach 原数据集的保存镜像。镜像 README 声明数据集采用 MIT 许可、市场数据仅供研究和教育用途；请在再分发或商业使用前自行核对该项目的许可和数据权利。全市场文件合计约 1.6 GB；脚本默认只下载 SPY 指定年份，2010–2025 年 SPY 文件合计约 570 MB。

## 重要边界

预测实现波动率并不等于策略盈利。期权隐含波动率、偏斜、期限结构、成交质量、交易成本和仓位风险都需要单独建模。此仓库是研究原型，不连接券商，也不构成投资建议。

多年度前推预测和 EOD 跨式策略的当前结果与限制见 [初步研究结果](docs/research_results.md)。
