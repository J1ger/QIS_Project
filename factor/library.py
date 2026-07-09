"""基本面、技术面和另类因子库。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exceptions import DataValidationError

FACTOR_NAMES = [
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "reversal_5",
    "volatility_20",
    "volatility_60",
    "ma_gap_5",
    "ma_gap_20",
    "ma_gap_60",
    "rsi_14",
    "price_position_20",
    "volume_ratio_20",
    "price_volume_divergence_20",
    "volume_price_corr_20",
    "turnover_mean_20",
    "turnover_stability_20",
    "illiquidity_20",
    "max_return_20",
    "min_return_20",
    "market_beta_60",
    "earnings_yield",
    "book_to_price",
    "sales_to_price",
    "cashflow_yield",
    "accrual_ratio",
    "cash_conversion_ratio",
    "roe",
    "roa",
    "gross_margin",
    "asset_turnover",
    "current_ratio",
    "leverage",
    "dividend_yield",
    "profit_growth_60",
    "revenue_growth_60",
    "log_size",
    "sentiment_5",
    "sentiment_20",
    "northbound_5",
    "northbound_20",
    "pmi_change_20",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """执行除法并将无穷值替换为缺失值。"""

    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def compute_factors(data: pd.DataFrame) -> pd.DataFrame:
    """计算 41 个标准因子。

    Args:
        data: 按 date/symbol 排列的标准日频面板。

    Returns:
        原始字段与因子列组成的面板。
    """

    required = {
        "date",
        "symbol",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "turnover_rate",
        "market_cap",
        "net_profit",
        "book_value",
        "revenue",
        "operating_cashflow",
        "assets",
        "liabilities",
        "gross_profit",
        "current_assets",
        "current_liabilities",
        "dividend",
        "sentiment",
        "northbound_flow",
        "macro_pmi",
    }
    missing = required.difference(data.columns)
    if missing:
        raise DataValidationError(f"因子计算缺少字段: {sorted(missing)}")

    frame = data.sort_values(["symbol", "date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    frame["return_1d"] = grouped["close"].pct_change()
    market_series = frame.groupby("date")["return_1d"].mean().sort_index()
    market_variance = market_series.rolling(60, min_periods=30).var()
    frame["market_return"] = frame["date"].map(market_series)
    frame["market_variance_60"] = frame["date"].map(market_variance)

    for window in (5, 20, 60):
        frame[f"momentum_{window}"] = grouped["close"].pct_change(window)
        moving_average = grouped["close"].transform(
            lambda series, size=window: series.rolling(size, min_periods=size // 2).mean()
        )
        frame[f"ma_gap_{window}"] = _safe_divide(frame["close"], moving_average) - 1

    frame["reversal_5"] = -frame["momentum_5"]
    for window in (20, 60):
        frame[f"volatility_{window}"] = grouped["return_1d"].transform(
            lambda series, size=window: series.rolling(size, min_periods=size // 2).std()
        )

    delta = grouped["close"].diff()
    gain = delta.clip(lower=0).groupby(frame["symbol"]).transform(
        lambda series: series.rolling(14, min_periods=7).mean()
    )
    loss = (-delta.clip(upper=0)).groupby(frame["symbol"]).transform(
        lambda series: series.rolling(14, min_periods=7).mean()
    )
    relative_strength = _safe_divide(gain, loss)
    frame["rsi_14"] = 100 - 100 / (1 + relative_strength)

    rolling_high = grouped["high"].transform(
        lambda series: series.rolling(20, min_periods=10).max()
    )
    rolling_low = grouped["low"].transform(
        lambda series: series.rolling(20, min_periods=10).min()
    )
    frame["price_position_20"] = _safe_divide(
        frame["close"] - rolling_low, rolling_high - rolling_low
    )
    volume_mean = grouped["volume"].transform(
        lambda series: series.rolling(20, min_periods=10).mean()
    )
    frame["volume_ratio_20"] = _safe_divide(frame["volume"], volume_mean)
    log_volume = np.log(frame["volume"].clip(lower=1))
    volume_change = log_volume.groupby(frame["symbol"]).diff()
    frame["price_volume_divergence_20"] = (
        grouped["close"].pct_change(20)
        - log_volume.groupby(frame["symbol"]).diff(20)
    )
    frame["volume_price_corr_20"] = frame.groupby("symbol", group_keys=False).apply(
        lambda group: group["return_1d"].rolling(20, min_periods=10).corr(
            volume_change.loc[group.index]
        ),
        include_groups=False,
    ).reset_index(level=0, drop=True).reindex(frame.index)
    frame["turnover_mean_20"] = grouped["turnover_rate"].transform(
        lambda series: series.rolling(20, min_periods=10).mean()
    )
    frame["turnover_stability_20"] = grouped["turnover_rate"].transform(
        lambda series: series.rolling(20, min_periods=10).std()
    )
    frame["illiquidity_20"] = (
        _safe_divide(frame["return_1d"].abs(), frame["amount"])
        .groupby(frame["symbol"])
        .transform(lambda series: series.rolling(20, min_periods=10).mean())
    )
    frame["max_return_20"] = grouped["return_1d"].transform(
        lambda series: series.rolling(20, min_periods=10).max()
    )
    frame["min_return_20"] = grouped["return_1d"].transform(
        lambda series: series.rolling(20, min_periods=10).min()
    )

    covariance = grouped.apply(
        lambda group: group["return_1d"].rolling(60, min_periods=30).cov(group["market_return"]),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    frame["market_beta_60"] = _safe_divide(
        covariance.reindex(frame.index), frame["market_variance_60"]
    )

    frame["earnings_yield"] = _safe_divide(frame["net_profit"], frame["market_cap"])
    frame["book_to_price"] = _safe_divide(frame["book_value"], frame["market_cap"])
    frame["sales_to_price"] = _safe_divide(frame["revenue"], frame["market_cap"])
    frame["cashflow_yield"] = _safe_divide(
        frame["operating_cashflow"], frame["market_cap"]
    )
    frame["accrual_ratio"] = _safe_divide(
        frame["net_profit"] - frame["operating_cashflow"], frame["assets"]
    )
    frame["cash_conversion_ratio"] = _safe_divide(
        frame["operating_cashflow"], frame["net_profit"].abs()
    )
    frame["roe"] = _safe_divide(frame["net_profit"], frame["book_value"])
    frame["roa"] = _safe_divide(frame["net_profit"], frame["assets"])
    frame["gross_margin"] = _safe_divide(frame["gross_profit"], frame["revenue"])
    frame["asset_turnover"] = _safe_divide(frame["revenue"], frame["assets"])
    frame["current_ratio"] = _safe_divide(
        frame["current_assets"], frame["current_liabilities"]
    )
    frame["leverage"] = _safe_divide(frame["liabilities"], frame["assets"])
    frame["dividend_yield"] = _safe_divide(frame["dividend"], frame["market_cap"])
    frame["profit_growth_60"] = grouped["net_profit"].pct_change(252)
    frame["revenue_growth_60"] = grouped["revenue"].pct_change(252)
    frame["log_size"] = np.log(frame["market_cap"].clip(lower=1))

    for window in (5, 20):
        frame[f"sentiment_{window}"] = grouped["sentiment"].transform(
            lambda series, size=window: series.rolling(size, min_periods=max(2, size // 2)).mean()
        )
        scaled_flow = _safe_divide(frame["northbound_flow"], frame["market_cap"])
        frame[f"northbound_{window}"] = scaled_flow.groupby(frame["symbol"]).transform(
            lambda series, size=window: series.rolling(size, min_periods=max(2, size // 2)).mean()
        )
    pmi_change = (
        frame[["date", "macro_pmi"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")["macro_pmi"]
        .diff(20)
    )
    frame["pmi_change_market"] = frame["date"].map(pmi_change)
    frame["pmi_change_20"] = frame.groupby("symbol", group_keys=False).apply(
        lambda group: group["return_1d"]
        .rolling(120, min_periods=60)
        .cov(group["pmi_change_market"]),
        include_groups=False,
    ).reset_index(level=0, drop=True).reindex(frame.index)

    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)
