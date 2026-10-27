# 配发公告样本：量化派（02685）

来源：HKEX 官方《Announcement of Final Offer Price and Allotment Results》，公告时间 2025-11-26 22:22 HKT，上市日 2025-11-27。PDF：[官方公告](https://www1.hkexnews.hk/listedco/listconews/sehk/2025/1126/2025112601522.pdf)。

## 解析与核对

- 最终发行价：HK$9.80；每手：500 股。
- 香港公开发售有效申请：159,327；成功申请：2,670；认购倍数：9,366.28 倍。
- 解析到的各档结果组人数总和为 159,327；获配非零股数的申请人数合计为 2,670，与公告汇总相符。
- 500 股申购档位拆为 52,201 份获配 0 股、314 份获配 500 股。合计有效申请 52,515。
- 因此一手档位获配概率为 `314 / 52,515 = 0.5979%`，平均获配约 `2.9896` 股；公告显示的获配比例为 0.60%，与计算值四舍五入一致。

上述均值是对该申购档位所有申请的期望值，不表示个人可能实际获配零碎股份。实际结果是没有获配，或获配 500 股。

## 与首日回报结合的示例

当前 Yahoo/yfinance 面板记录首日收盘 HK$18.50，发行价至收盘毛回报约 +88.8%。若每位中签者获配 500 股，单个成功申请的首日收盘毛盈利是 HK$4,350；乘以 0.5979% 获配概率，按一手申请计算的事后期望毛盈利约 HK$26.01。

这只是一个事后核对示例，尚未扣申购、交易及融资成本，也未处理资金占用；行情为非交易所核验数据。不能据单只 IPO 推断有可重复的申购优势。

## 输出文件

- [`../data/processed/hkipo_allotment_seed/hkipo_allotment_events.csv`](../data/processed/hkipo_allotment_seed/hkipo_allotment_events.csv)
- [`../data/processed/hkipo_allotment_seed/hkipo_allotment_tiers.csv`](../data/processed/hkipo_allotment_seed/hkipo_allotment_tiers.csv)
- [`../data/processed/hkipo_allotment_seed/hkipo_allotment_outcomes.csv`](../data/processed/hkipo_allotment_seed/hkipo_allotment_outcomes.csv)

原始 PDF 保存在 `data/raw/hkex_allotment_results/02685_20251126.pdf`（本地原始数据，不纳入版本控制）。解析方式：

```powershell
python -m pip install -e ".[pdf]"
python -m hkipo.parse_allotment_pdf --pdf data/raw/hkex_allotment_results/02685_20251126.pdf --event-id 02685_20251127 --stock-code 02685 --listing-date 2025-11-27 --release-time-hkt "2025-11-26 22:22" --source-url https://www1.hkexnews.hk/listedco/listconews/sehk/2025/1126/2025112601522.pdf --out-dir data/processed/hkipo_allotment_seed
```
