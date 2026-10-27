# 港股 IPO 申购截止日期核验（2021–2025）

截至本次核验，进入现金模拟器的 266 个事件全部有精确到年月日的官方申购截止日期：266/266 来自 HKEX 发行人公告或招股书时间表，年份推断日期为 0，情景补值为 0。按来源拆分为公告 87 条、HTML 招股书时间表 9 条、PDF 招股书时间表 170 条。逐年覆盖见 [`../reports/hkipo_cash_timeline_coverage_2021_2025.csv`](../reports/hkipo_cash_timeline_coverage_2021_2025.csv)。

## 补查的 9 个事件

下表记录先前缺失或年份只靠推断的事件。PDF 均从 HKEX 官方域名下载，原文哈希保存在 [`../reports/hkipo_prospectus_timetable_pdf_extraction_2021_2025.json`](../reports/hkipo_prospectus_timetable_pdf_extraction_2021_2025.json) 及专项复核文件 [`../reports/hkipo_unresolved_close_date_pdf_review_2021_2025.json`](../reports/hkipo_unresolved_close_date_pdf_review_2021_2025.json)、[`../reports/hkipo_unresolved_close_date_confirmations_2021_2025.json`](../reports/hkipo_unresolved_close_date_confirmations_2021_2025.json)。

| 事件 | 申购截止日 | HKEX 原始 PDF | 本地 SHA-256 |
|---|---|---|---|
| Vanov (02260) | 2022-01-03 | [时间表 PDF](https://www1.hkexnews.hk/listedco/listconews/sehk/2021/1229/10065877/2021122900081.pdf) | `3d6605f5233d4d4158c58622a45f8117b83c60197b2c5bac8491fa4d7fadea6b` |
| Maiyue (02501) | 2023-10-04 | [时间表 PDF](https://www1.hkexnews.hk/listedco/listconews/sehk/2023/0928/10913343/2023092800103.pdf) | `1d4830e1505452d2424b6a31b6fbe17df9b6120050b8674234827221ca86dcd4` |
| FAR International (02516) | 2023-12-19 | [时间表 PDF](https://www1.hkexnews.hk/listedco/listconews/sehk/2023/1212/11003145/2023121200095.pdf) | `8e5a6ebab9aa6392166b73a0f0f2d5e531b064d04dd66c70d5a8f71941ae4ce8` |
| Persistence Resources (02489) | 2023-12-19 | [时间表 PDF](https://www1.hkexnews.hk/listedco/listconews/sehk/2023/1214/11005541/2023121400091.pdf) | `e10e5d708ea32e9cff0c9365d0d9cf204f201667abaf492ba5dee356dcd7642c` |
| WellCell (02477) | 2024-01-09 | [招股书 PDF](https://www1.hkexnews.hk/listedco/listconews/sehk/2023/1228/2023122800033.pdf) | `e6104144719ab3e20a8bf00dc2f6324f7b4175a4e50016d363141348c20e0b81` |
| WK Group (02535) | 2024-03-05 | [时间表 PDF](https://www1.hkexnews.hk/listedco/listconews/sehk/2024/0229/11078996/2024022900021.pdf) | `a75b6a54eb61722cf5e9e56b3850b7faa0b4d94f994f124d76061252245039ff` |
| Wing Lee Development (09639) | 2024-10-04 | [招股书 PDF](https://www1.hkexnews.hk/listedco/listconews/sehk/2024/0927/2024092700039.pdf) | `0d05e112892ebd8816a33ad2187b03f0ceed557bcffdb662c816b416b3f9c8f6` |
| APT Electronics (02551) | 2024-11-05 | [时间表 PDF](https://www1.hkexnews.hk/listedco/listconews/sehk/2024/1031/11420785/2024103100081.pdf) | `375b5a7d43009bdffa657b17ce02e3b1df3ef042733670b80cdbda60efd76080` |
| Jiangsu Guofu (02582) | 2024-11-12 | [招股书 PDF](https://www1.hkexnews.hk/listedco/listconews/sehk/2024/1107/2024110700009.pdf) | `b2521d2260513569e82ce8e78cf28bc86681d1a02f3c260c3e01387453692ed5` |

日期通过招股书/配发结果/上市日期先后关系检查，且截止日在招股书日期之后、配发结果日当天或之前、上市日前 45 天内。PDF 时间表有些把年份放在表头或脚注，没有紧贴日期重复，因此提取器会在同一文件的时间表正文中寻找相同月日的显式年份；通过该核验后才标记为精确来源。日期表示预期申购截止日；如有后续改期公告，应以后续正式公告为准。申购截止日期本身不等于券商接受委托的最后时刻。

## 对模拟器的意义

重建后，现金模拟器不再使用“配发结果日减 5 天”的申购日情景。严格现金条款模拟仍只使用一手申购、官方最高申购价计算申请现金，并以首日 Yahoo/yfinance 开盘价作为卖出价格代理；配发结果为期望值分布。因此日期覆盖完整提升了资金占用的时间定位，但没有把它变成真实逐账户成交回测，也没有补齐券商实际扣款、退款到账及成交队列记录。
