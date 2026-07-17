"""IC/IR、统计显著性和分层收益检验。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


_SUMMARY_COLUMNS = [
    "factor",
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_positive_ratio",
    "t_stat",
    "observations",
    "top_bottom_spread",
]


def _require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when the input panel is missing required columns."""

    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def add_forward_returns(data: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """添加不含未来信息的远期收益标签。"""

    if period < 1:
        raise ValueError("period must be at least 1 trading day")
    _require_columns(data, ["date", "symbol", "close"])
    frame = data.sort_values(["symbol", "date"]).copy()
    frame["forward_return"] = (
        frame.groupby("symbol")["close"].shift(-period) / frame["close"] - 1
    )
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    """使用秩变换计算 Spearman 相关系数。"""

    valid = left.notna() & right.notna()
    if valid.sum() < 5:
        return float("nan")
    left_rank = left[valid].rank()
    right_rank = right[valid].rank()
    if left_rank.std(ddof=0) <= 1e-12 or right_rank.std(ddof=0) <= 1e-12:
        return float("nan")
    return float(left_rank.corr(right_rank))


def calculate_ic_series(data: pd.DataFrame, factor: str) -> pd.Series:
    """计算因子日度 Rank IC 序列。"""

    _require_columns(data, ["date", factor, "forward_return"])
    return data.groupby("date").apply(
        lambda group: _spearman_correlation(group[factor], group["forward_return"]),
        include_groups=False,
    ).rename(factor)


def calculate_quantile_returns(
    data: pd.DataFrame, factor: str, quantiles: int = 5
) -> pd.DataFrame:
    """计算因子分位数组合的日均远期收益。"""

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    _require_columns(data, ["date", factor, "forward_return"])
    subset = data[["date", factor, "forward_return"]].dropna().copy()
    if subset.empty:
        return pd.DataFrame(columns=["date", "quantile", "mean_return"])

    def assign_quantile(group: pd.DataFrame) -> pd.Series:
        if len(group) < quantiles:
            return pd.Series(np.nan, index=group.index, dtype=float)
        ranks = group[factor].rank(method="first")
        try:
            return pd.qcut(ranks, quantiles, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series(np.nan, index=group.index)

    subset["quantile"] = subset.groupby("date", group_keys=False).apply(
        assign_quantile, include_groups=False
    ).reset_index(level=0, drop=True)
    return (
        subset.groupby(["date", "quantile"], as_index=False)["forward_return"]
        .mean()
        .rename(columns={"forward_return": "mean_return"})
    )


def evaluate_factors(
    data: pd.DataFrame,
    factors: list[str],
    quantiles: int = 5,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """批量输出因子摘要、IC 时间序列和分层收益。"""

    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    _require_columns(data, ["date", "forward_return", *factors])
    summaries: list[dict[str, float | str | int]] = []
    ic_frames: list[pd.Series] = []
    quantile_frames: list[pd.DataFrame] = []
    for factor in factors:
        ic = calculate_ic_series(data, factor).dropna()
        if len(ic) < min_observations:
            continue
        ic_mean = float(ic.mean())
        ic_std = float(ic.std(ddof=1))
        ic_ir = ic_mean / ic_std if ic_std > 0 else float("nan")
        t_stat = ic_mean / (ic_std / math.sqrt(len(ic))) if ic_std > 0 else float("nan")
        quantile_result = calculate_quantile_returns(data, factor, quantiles)
        top = quantile_result[quantile_result["quantile"] == quantiles]["mean_return"].mean()
        bottom = quantile_result[quantile_result["quantile"] == 1]["mean_return"].mean()
        summaries.append(
            {
                "factor": factor,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "ic_positive_ratio": float((ic > 0).mean()),
                "t_stat": t_stat,
                "observations": int(len(ic)),
                "top_bottom_spread": float(top - bottom),
            }
        )
        ic_frames.append(ic)
        quantile_result["factor"] = factor
        quantile_frames.append(quantile_result)

    summary = pd.DataFrame(summaries, columns=_SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values("ic_mean", key=lambda s: s.abs(), ascending=False)
    ic_data = pd.concat(ic_frames, axis=1, sort=False) if ic_frames else pd.DataFrame()
    quantile_data = (
        pd.concat(quantile_frames, ignore_index=True) if quantile_frames else pd.DataFrame()
    )
    return summary.reset_index(drop=True), ic_data, quantile_data


def evaluate_factor_by_group(
    data: pd.DataFrame,
    factors: list[str],
    group_column: str,
    minimum_rows: int = 20,
) -> pd.DataFrame:
    """按行业或市场环境计算分组 Rank IC。"""

    if minimum_rows < 1:
        raise ValueError("minimum_rows must be positive")
    _require_columns(data, ["date", "forward_return", group_column, *factors])
    records: list[dict[str, float | str | int]] = []
    for group_name, group_data in data.groupby(group_column):
        for factor in factors:
            values = group_data.groupby("date").apply(
                lambda cross_section: _spearman_correlation(
                    cross_section[factor], cross_section["forward_return"]
                ),
                include_groups=False,
            ).dropna()
            if len(values) < minimum_rows:
                continue
            records.append(
                {
                    "group_type": group_column,
                    "group": str(group_name),
                    "factor": factor,
                    "ic_mean": float(values.mean()),
                    "ic_ir": float(values.mean() / values.std(ddof=1))
                    if values.std(ddof=1) > 0
                    else float("nan"),
                    "observations": int(len(values)),
                }
            )
    return pd.DataFrame(records)


def add_market_regimes(data: pd.DataFrame) -> pd.DataFrame:
    """根据市场趋势与波动把交易日划分为四类环境。"""

    _require_columns(data, ["date", "market_return"])
    frame = data.copy()
    market = (
        frame[["date", "market_return"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")["market_return"]
    )
    trend = market.rolling(60, min_periods=30).mean()
    volatility = market.rolling(60, min_periods=30).std()
    volatility_threshold = volatility.expanding(min_periods=30).median()
    regime = pd.Series(index=market.index, dtype="object")
    regime[(trend >= 0) & (volatility <= volatility_threshold)] = "上涨-低波"
    regime[(trend >= 0) & (volatility > volatility_threshold)] = "上涨-高波"
    regime[(trend < 0) & (volatility <= volatility_threshold)] = "下跌-低波"
    regime[(trend < 0) & (volatility > volatility_threshold)] = "下跌-高波"
    frame["market_regime"] = frame["date"].map(regime).fillna("预热期")
    return frame
