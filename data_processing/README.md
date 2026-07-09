# data_processing

`data_processing` 是一个可独立复用的 A 股数据处理模块，主要负责：

- 通过 AkShare 获取或补充标准化市场数据
- 维护轻量 CSV 数据仓库和版本清单
- 执行原始数据质量检查

当前仓库包含以下核心文件：

- `providers.py`：数据源接口、SyntheticProvider、AkShareProvider、`build_provider`
- `quality.py`：字段完整性、主键、价格逻辑和覆盖度检查
- `storage.py`：CSV 持久化、版本清单和外侧日期增量补拉判断
- `exceptions.py`：模块内自包含异常定义

## 环境安装

推荐方式一：

```bash
pip install -r requirements.txt
```

推荐方式二：

```bash
conda env create -f environment.yml
conda activate data-processing
```

## 快速使用

```python
from providers import build_provider
from quality import quality_report

config = {
    "provider": "akshare",
    "start_date": "2020-01-02",
    "end_date": "2025-12-31",
    "symbols": ["600519.SH", "000858.SZ"],
    "adjust": "qfq",
    "cache_dir": "data/akshare_cache",
    "storage_dir": "data",
}

provider = build_provider(config, random_seed=42)
panel = provider.fetch("2020-01-02", "2025-12-31")
report = quality_report(panel)
print(panel.head())
print(report)
```

## 输出字段说明

标准化面板通常包含：

- 交易字段：`date, symbol, open, high, low, close, volume, amount`
- 市场状态：`is_suspended, is_st, limit_up, limit_down`
- 基本面字段：`revenue, net_profit, assets, liabilities` 等
- 另类数据：`northbound_flow, macro_pmi`

## 说明

- 当前默认真实数据源为 AkShare。
- 价格口径默认建议使用前复权 `qfq`，以避免技术因子受到分红送股跳空影响。



