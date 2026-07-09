# factor

`factor` 是一个可独立复用的多因子计算模块，负责：

- 从标准化 A 股日频面板中计算原始因子
- 对因子执行缺失值处理、缩尾、中性化和标准化
- 生成因子版本清单与血缘哈希

当前仓库包含以下核心文件：

- `library.py`：41 个因子的计算逻辑
- `preprocess.py`：截面因子治理流程
- `manifest.py`：因子版本、数据哈希和预处理口径清单
- `exceptions.py`：模块内自包含异常定义

## 环境安装

```bash
pip install -r requirements.txt
```

或：

```bash
conda env create -f environment.yml
conda activate factor-module
```

## 快速使用

```python
import pandas as pd

from library import FACTOR_NAMES, compute_factors
from preprocess import preprocess_factors
from manifest import build_factor_manifest, factor_manifest_frame

panel = pd.read_csv("standardized_panel.csv", parse_dates=["date"])
raw = compute_factors(panel)
processed = preprocess_factors(
    raw,
    FACTOR_NAMES,
    lower_quantile=0.01,
    upper_quantile=0.99,
    neutralize=True,
    min_cross_section=15,
)

settings = {
    "root_dir": ".",
    "project": {"name": "A股多因子研究"},
    "data": {
        "provider": "akshare",
        "start_date": "2020-01-02",
        "end_date": "2025-12-31",
    },
    "factor": {
        "winsor_lower": 0.01,
        "winsor_upper": 0.99,
        "neutralize": True,
        "min_cross_section": 15,
    },
}

manifest = build_factor_manifest(
    raw_factors=raw,
    processed_factors=processed,
    factor_names=FACTOR_NAMES,
    settings=settings,
)

factor_manifest_frame(manifest).to_csv("factor_manifest.csv", index=False)
processed.to_csv("factors.csv", index=False)
```

## 因子覆盖

当前共 41 个因子，覆盖以下维度：

- 技术面：动量、反转、波动率、均线偏离、RSI、价格位置
- 量价流动性：量价背离、量价相关、换手率、非流动性
- 基本面：估值、盈利质量、增长、杠杆、现金流
- 风格/另类/宏观：规模、情绪、北向资金、PMI

## 清单文件说明

`manifest.py` 会输出：

- 因子分类与版本号
- 原始因子和处理后因子的缺失率
- 预处理参数
- 代码文件哈希
- 数据内容哈希

