"""金融面板数据质量检查。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from exceptions import DataValidationError

REQUIRED_MARKET_COLUMNS = {
    "date",
    "symbol",
    "industry",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "market_cap",
}


def validate_market_data(data: pd.DataFrame) -> None:
    """检查必要字段、主键和价格逻辑。"""

    missing = REQUIRED_MARKET_COLUMNS.difference(data.columns)
    if missing:
        raise DataValidationError(f"市场数据缺少字段: {sorted(missing)}")
    if data.empty:
        raise DataValidationError("市场数据为空")
    if data.duplicated(["date", "symbol"]).any():
        raise DataValidationError("存在重复的 date/symbol 主键")
    invalid_price = (
        (data["low"] > data["high"])
        | (data["open"] <= 0)
        | (data["close"] <= 0)
        | (data["volume"] < 0)
    )
    if invalid_price.any():
        raise DataValidationError(f"发现 {int(invalid_price.sum())} 行非法行情数据")


def quality_report(data: pd.DataFrame) -> dict[str, Any]:
    """生成覆盖度、缺失率、重复率和异常值摘要。"""

    validate_market_data(data)
    numeric = data.select_dtypes(include=[np.number])
    missing_ratio = data.isna().mean().sort_values(ascending=False)
    core_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover_rate",
            "market_cap",
            "industry",
            "net_profit",
            "revenue",
            "book_value",
        )
        if column in data.columns
    ]
    first_observation = data.groupby("symbol")["date"].min()
    pre_listing_rows = 0
    if "listing_date" in data.columns:
        listing_date = pd.to_datetime(data["listing_date"], errors="coerce")
        pre_listing_rows = int(
            (pd.to_datetime(data["date"]) < listing_date).fillna(False).sum()
        )
    return {
        "start_date": str(pd.to_datetime(data["date"]).min().date()),
        "end_date": str(pd.to_datetime(data["date"]).max().date()),
        "row_count": int(len(data)),
        "symbol_count": int(data["symbol"].nunique()),
        "trading_day_count": int(data["date"].nunique()),
        "duplicate_key_count": int(data.duplicated(["date", "symbol"]).sum()),
        "top_missing_ratios": {
            key: float(value) for key, value in missing_ratio.head(15).items()
        },
        "core_missing_ratios": {
            column: float(data[column].isna().mean()) for column in core_columns
        },
        "earliest_observation_by_symbol": {
            str(symbol): str(pd.Timestamp(date).date())
            for symbol, date in first_observation.items()
        },
        "pre_listing_row_count": pre_listing_rows,
        "suspended_row_count": int(
            data.get("is_suspended", pd.Series(False, index=data.index))
            .fillna(False)
            .sum()
        ),
        "suspended_symbol_count": int(
            data.loc[
                data.get("is_suspended", pd.Series(False, index=data.index)).fillna(False),
                "symbol",
            ].nunique()
        ),
        "suspended_ratio": float(
            data.get("is_suspended", pd.Series(False, index=data.index))
            .fillna(False)
            .mean()
        ),
        "zero_volume_row_count": int(
            data.get("is_zero_volume", pd.Series(False, index=data.index))
            .fillna(False)
            .sum()
        ),
        "official_suspension_row_count": int(
            data.get("official_suspension_flag", pd.Series(False, index=data.index))
            .fillna(False)
            .sum()
        ),
        "inferred_missing_suspension_row_count": int(
            data.get(
                "is_suspended_inferred_missing", pd.Series(False, index=data.index)
            )
            .fillna(False)
            .sum()
        ),
        "price_adjustment_modes": {
            str(key): int(value)
            for key, value in data.get(
                "price_adjustment", pd.Series("unknown", index=data.index)
            )
            .fillna("unknown")
            .value_counts()
            .items()
        },
        "adjusted_price_row_count": int(
            data.get("adjustment_applied", pd.Series(False, index=data.index))
            .fillna(False)
            .sum()
        ),
        "sentiment_non_null_ratio": float(
            1
            - data.get("sentiment", pd.Series(np.nan, index=data.index))
            .isna()
            .mean()
        ),
        "northbound_non_null_ratio": float(
            1
            - data.get("northbound_flow", pd.Series(np.nan, index=data.index))
            .isna()
            .mean()
        ),
        "sentiment_source_modes": {
            str(key): int(value)
            for key, value in data.get(
                "sentiment_source", pd.Series("unknown", index=data.index)
            )
            .fillna("unknown")
            .value_counts()
            .items()
        },
        "infinite_numeric_count": int(np.isinf(numeric.to_numpy()).sum()),
        "non_positive_close_count": int((data["close"] <= 0).sum()),
    }
