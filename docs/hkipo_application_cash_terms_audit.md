# 港股 IPO 申购最高发售价条款核验

## 核验方式

申请时所需现金通常按招股书披露的最高发售价及相关费用收取，最终定价较低时再退款。港交所的上市申请人指引示例明确列出最高发售价在申购时缴付、定价较低时退款的规则，见 [HKEX 申请人指引示例](https://www.hkex.com.hk/-/media/HKEX-Market/Listing/Rules-and-Guidance/Interpretation-and-Guidance/Guide-for-New-Listing-Applicants/chapter_4.pdf)。

先从港交所同日发布的白色申请表提取最高发售价和申购费用率；再从已核验的配发公告 PDF 文本补充。针对缺项，使用精确公司名检索港交所 HKEXnews 索引，提取索引摘要中的最高发售价并保留对应 PDF 链接及摘要证据。申请表优先于公告文本，公告文本优先于索引摘要；索引结果必须链接至 HKEX 官方公告，且最高价不得低于最终发行价。

除申请表与配发公告外，还从官方 HKEX PDF 正文恢复缺项：包括直接招股章程、招股书摘要 PDF，以及索引候选。合并审计中 144 个官方 PDF 的正文抽取均匹配最高发售价，来源按事件去重；原件位于 `data/raw/hkex_ipo_indexed_cash_term_pdfs_2021_2025/`、`data/raw/hkex_ipo_missing_cash_term_prospectuses_2021_2025/` 和 `data/raw/hkex_ipo_html_prospectus_sections_2021_2025/`。9639 榮利營造的招股章程正文核实最高价 HK$0.73。费用率、现金扣留的准确时点和历史券商渠道仍须另行核验。完整逐条审计见 [`hkipo_max_price_pdf_source_audit_2021_2025.json`](../reports/hkipo_max_price_pdf_source_audit_2021_2025.json)。

## 覆盖

| 上市年 | 模拟器事件 | 找到可用最高申购价 | 申请表来源 | 配发公告文字来源 | HKEX PDF 正文来源 |
|---|---:|---:|---:|---:|---:|
| 2021 | 49 | 49 | 43 | 5 | 1 |
| 2022 | 37 | 37 | 32 | 4 | 1 |
| 2023 | 38 | 38 | 24 | 4 | 10 |
| 2024 | 42 | 42 | 0 | 0 | 42 |
| 2025 | 100 | 100 | 0 | 10 | 90 |
| **合计** | **266** | **266** | **99** | **23** | **144** |

申请表和配发公告都能提取的 106 个事件中，101 个价格一致，5 个抽取值有差异；这 5 项已逐份对照申请表和配发公告正文完成复核，均采用申请时有效的申请表最高价。4 项公告解析抓到了最终定价或认购金额阈值，另 1 项是发行价下调后的最终价。逐项说明见 [`hkipo_cash_term_source_conflict_review_2021_2025.csv`](../reports/hkipo_cash_term_source_conflict_review_2021_2025.csv)。已恢复样本的最高申购价/最终发行价比率中位数为 1.071、95 分位为 1.341，最大值为 2.125；分布按年份差异明显，不能把中位数机械当成所有事件的保证金倍率。

逐事件数据在 [`hkex_ipo_cash_terms_2021_2025.csv`](../data/processed/hkex_ipo_cash_terms_2021_2025.csv)，覆盖在 [`hkipo_cash_terms_coverage_2021_2025.csv`](../reports/hkipo_cash_terms_coverage_2021_2025.csv)。144 条主搜索结果和 36 条替代搜索结果（均包括未命中项）分别在 [`hkipo_web_search_cash_terms_2021_2025.json`](../reports/hkipo_web_search_cash_terms_2021_2025.json) 和 [`hkipo_web_search_cash_terms_alternates_round2_all.json`](../reports/hkipo_web_search_cash_terms_alternates_round2_all.json)；PDF 提取核验脚本为 [`verify_indexed_cash_term_pdfs.py`](../src/hkipo/verify_indexed_cash_term_pdfs.py)。申请表原件与条款审计表分别在 `data/processed/hkex_ipo_application_forms_2021_2025.csv` 和 `data/raw/hkex_ipo_application_forms_2021_2025/`。

## 对模拟器和 DRL 的影响

最高申购价条款现已覆盖 266/266，2021–2025 每年均全覆盖；其中 144 个价格由官方 PDF 正文抽取并逐事件核对。严格现金条款情景现可在全体 266 个事件上执行：HK$100,000 初始现金，期末净值 HK$182,219，净收益 HK$82,219（+82.2%），最大回撤 -22.4%，无因资金不足或缺少条款而跳过的事件。此结果仍是配发档位期望值路径和价格代理，不是单一投资者可实现的历史资金回测或策略证据。

最高价条款覆盖闸门已通过；申购截止日期现已 266/266 官方核验，但费用率和资金冻结/退款准确时点仍未按经纪渠道、年份完整审计；5 个原始抽取差异已完成逐项人工复核，明细保留在差异审计表。已用现有假设运行 DQN 探索性训练与 2025 留出比较，结果不稳定且未显示稳健增益；在取得可信点时特征、完整账户状态并冻结统一成本与成交假设前，不把它当成正式 OOS 结论。
