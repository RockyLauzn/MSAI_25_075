# 行情源限制与可选验证

## 当前阶段

项目当前只做策略探索，不要求购买行情。Yahoo Finance/yfinance 可以用于尝试事件特征、规则和组合构造；输出结果需要标为探索性，不应当成真实可执行收益。现有报告会保留覆盖范围、OHLC 冲突和未取到行情的数量。

如果某条策略在探索中呈现稳定现象，之后再考虑用 HKEX 授权历史数据或能说明授权来源的供应商数据复核。HKEX 官方历史数据产品需订阅/购买，产品目录和字段样例见 [HKEX Data Marketplace](https://data.hkex.com.hk/)；交付方式说明见 [HKEX 历史数据 FAQ](https://www.hkex.com.hk/Global/Exchange/FAQ/Market-Data?sc_lang=en)。不同产品简介列出的字段可能不覆盖完整 OHLCV，届时要先确认样例和许可范围。

## 可选 CSV 导入

如以后拿到授权文件或供应商 CSV，可把文件放进 `data/raw/`，创建字段映射 JSON。左侧为项目标准字段，右侧为源文件实际表头：

```json
{
  "stock_code": "Stock Code",
  "date": "Trade Date",
  "open": "Open",
  "high": "High",
  "low": "Low",
  "close": "Close",
  "volume": "Volume",
  "turnover": "Turnover"
}
```

```powershell
python -m hkipo.import_bars_csv `
  --input data/raw/market_bars.csv `
  --output data/processed/hkipo_daily_bars_external.csv `
  --source-name "provider/product name" `
  --column-map data/raw/column_map.json
```

导入器会检查列映射、缺失值、重复证券/日期及价格、成交量边界，输出文件哈希、来源和 OHLC 质量摘要；不会自动改写异常行。这个入口只是以后更换或对照数据源时可用，当前研究继续用现有探索行情即可。
