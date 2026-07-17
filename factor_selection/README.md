# A-share Factor Evaluation and Selection

一个可独立运行的 A 股因子有效性检验与因子筛选模块。它接收已经完成因子计算和预处理的日频面板数据，不包含数据下载、因子计算或策略回测功能。

## 功能

- 按股票计算下一期收益标签，避免同日使用未来收益；
- 计算日度 Rank IC、IC 均值、IC_IR、IC 为正比例、t 值和五分位收益；
- 按行业或市场环境检查因子稳健性；
- 计算逐日截面秩相关矩阵；
- 使用 Ridge 回归估计联合特征重要性；
- 逐日 Gram-Schmidt 正交化，降低入选因子的信息重叠；
- 按 `0.6 × |IC| + 0.4 × Ridge importance` 排序，控制高相关因子和同逻辑家族上限。

默认同逻辑家族上限为 2，包含动量、波动率、趋势结构、量价流动性、估值、质量、成长和另类/宏观等家族。

## 输入要求

输入为 CSV 或 DataFrame，至少包含：

```text
date, symbol, close, market_return, <factor columns>
```

`date` 为交易日；`symbol` 为证券代码；每一个候选因子必须是一列数值。因子评价前会新增 `forward_return`。若做行业分组，还需要 `industry` 列。

本仓库不上传原始行情、因子结果或回测输出；请使用上游数据处理与因子计算项目生成输入文件。

## 安装

```powershell
python -m pip install -e ".[dev]"
```

或使用 Conda：

```powershell
conda env create -f environment.yml
conda activate factor-selection
```

## 使用示例

```python
import pandas as pd

from evaluation import add_forward_returns, add_market_regimes, evaluate_factors
from selection import (
    factor_correlation,
    orthogonalize_factors,
    ridge_feature_importance,
    select_factors,
)

data = pd.read_csv("factors.csv", parse_dates=["date"])
factor_names = ["momentum_20", "book_to_price", "roe", "illiquidity_20"]

# 1. 只按股票向前构造一期收益标签。
research = add_forward_returns(data, period=1)
research = add_market_regimes(research)

# 2. 单因子有效性。
summary, ic_series, quantile_returns = evaluate_factors(
    research, factor_names, quantiles=5, min_observations=30
)

# 3. 仅使用评价合格的因子进行联合筛选。
eligible = summary["factor"].tolist()
correlation = factor_correlation(research, eligible)
ridge = ridge_feature_importance(research, eligible, regularization=1.0)
selected, weights = select_factors(
    summary,
    correlation,
    ridge,
    max_correlation=0.75,
    max_factors=12,
    max_factors_per_family=2,
)

# 4. 对已入选因子正交化，供下游打分或回测使用。
orthogonalized = orthogonalize_factors(research, selected)

pd.DataFrame({"factor": selected, "weight": [weights[x] for x in selected]}).to_csv(
    "selected_factors.csv", index=False, encoding="utf-8-sig"
)
```

## 检验

```powershell
python -m pytest
```

测试覆盖前瞻收益标签、IC/分层收益输出、相关性矩阵、Ridge 重要性、同族上限和高相关约束。

## 方法边界

- IC、分层收益和 Ridge 重要性均为研究评价指标，不构成投资建议；
- 正交化顺序会影响残差结果，应在研究记录中固定入选排序；
- 最终因子及权重应当仅在训练期确定，再在验证期与测试期进行样本外评估；
- 本模块不替代停牌、涨跌停、交易成本和成交容量等策略可交易性约束。

## 许可证

本项目使用 [MIT License](LICENSE)。
