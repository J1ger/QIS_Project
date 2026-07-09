"""因子缺失处理、缩尾、中性化和标准化。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _neutralize_cross_section(
    group: pd.DataFrame,
    factor: str,
    neutralize: bool,
    min_cross_section: int,
) -> pd.Series:
    """对单日截面执行行业与市值中性化，再进行 Z-score。"""

    values = group[factor].replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median())
    if values.notna().sum() < min_cross_section:
        return pd.Series(np.nan, index=group.index)

    if neutralize:
        industry = pd.get_dummies(group["industry"], drop_first=True, dtype=float)
        size = np.log(group["market_cap"].clip(lower=1)).rename("log_market_cap")
        design = pd.concat(
            [pd.Series(1.0, index=group.index, name="intercept"), size, industry],
            axis=1,
        ).astype(float)
        try:
            coefficients = np.linalg.lstsq(design.to_numpy(), values.to_numpy(), rcond=None)[0]
            values = values - design.to_numpy() @ coefficients
        except np.linalg.LinAlgError:
            values = values - values.mean()

    standard_deviation = values.std(ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=group.index)
    return (values - values.mean()) / standard_deviation


def preprocess_factors(
    data: pd.DataFrame,
    factor_names: list[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    neutralize: bool = True,
    min_cross_section: int = 15,
) -> pd.DataFrame:
    """按交易日治理因子并保留原始字段。"""

    frame = data.copy()
    for factor in factor_names:
        lower = frame.groupby("date")[factor].transform(
            lambda series: series.quantile(lower_quantile)
        )
        upper = frame.groupby("date")[factor].transform(
            lambda series: series.quantile(upper_quantile)
        )
        frame[factor] = frame[factor].clip(lower=lower, upper=upper)
        frame[factor] = frame.groupby("date", group_keys=False).apply(
            _neutralize_cross_section,
            factor=factor,
            neutralize=neutralize,
            min_cross_section=min_cross_section,
            include_groups=False,
        ).reset_index(level=0, drop=True).reindex(frame.index)
    return frame

