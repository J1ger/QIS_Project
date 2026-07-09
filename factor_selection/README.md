# factor_selection

`factor_selection` 当前是一个“因子评价模块”，还不是完整的“因子筛选模块”。

目前它已经能做：

- 远期收益标签构造
- Rank IC 序列计算
- IC/IR/显著性统计
- 分位数收益检验
- 按行业或市场环境做分组因子评价
- 市场环境划分

目前它还不能做：

- 因子相关性约束筛选
- 共线性处理
- 岭回归/联合重要性排序
- 同逻辑因子家族上限控制
- 最终 `selected_factors.json` 式的选择结果输出

## 当前包含文件

- `evaluation.py`：IC/IR、分组表现、分位收益与市场环境划分

## 环境安装

```bash
pip install -r requirements.txt
```

或：

```bash
conda env create -f environment.yml
conda activate factor-selection
```

## 快速使用

```python
import pandas as pd

from evaluation import (
    add_forward_returns,
    evaluate_factors,
    evaluate_factor_by_group,
    add_market_regimes,
)

data = pd.read_csv("factors.csv", parse_dates=["date"])
data = add_forward_returns(data, period=1)
data = add_market_regimes(data)

factor_names = ["momentum_20", "book_to_price", "roe"]
summary, ic_series, quantile_returns = evaluate_factors(
    data,
    factor_names,
    quantiles=5,
    min_observations=30,
)

group_eval = evaluate_factor_by_group(
    data,
    factor_names,
    group_column="market_regime",
)
```

