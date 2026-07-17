from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import add_forward_returns, calculate_quantile_returns, evaluate_factors


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=40)
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for symbol_index in range(8):
            rows.append(
                {
                    "date": date,
                    "symbol": f"{symbol_index:06d}.SZ",
                    "close": 10 + symbol_index + 0.05 * date_index,
                    "factor_a": symbol_index + 0.1 * np.sin(date_index),
                    "factor_b": 7 - symbol_index + 0.1 * np.cos(date_index),
                }
            )
    return pd.DataFrame(rows)


def test_forward_returns_are_computed_within_each_symbol() -> None:
    result = add_forward_returns(_panel(), period=1)
    symbol = result.loc[result["symbol"] == "000000.SZ"].sort_values("date")
    expected = symbol["close"].iloc[1] / symbol["close"].iloc[0] - 1
    assert symbol["forward_return"].iloc[0] == expected
    assert pd.isna(symbol["forward_return"].iloc[-1])


def test_evaluation_returns_expected_outputs() -> None:
    research = add_forward_returns(_panel())
    summary, ic_series, quantile_returns = evaluate_factors(
        research, ["factor_a", "factor_b"], quantiles=4, min_observations=20
    )
    assert set(summary["factor"]) == {"factor_a", "factor_b"}
    assert set(ic_series.columns) == {"factor_a", "factor_b"}
    assert {"date", "quantile", "mean_return", "factor"}.issubset(quantile_returns.columns)


def test_quantile_returns_rejects_invalid_quantile_count() -> None:
    research = add_forward_returns(_panel())
    try:
        calculate_quantile_returns(research, "factor_a", quantiles=1)
    except ValueError as error:
        assert "at least 2" in str(error)
    else:
        raise AssertionError("Expected a validation error")
