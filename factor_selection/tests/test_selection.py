from __future__ import annotations

import numpy as np
import pandas as pd

from selection import (
    factor_correlation,
    orthogonalize_factors,
    ridge_feature_importance,
    select_factors,
)


def _research_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=40)
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for symbol_index in range(10):
            signal = symbol_index + rng.normal(scale=0.1)
            rows.append(
                {
                    "date": date,
                    "momentum_5": signal,
                    "momentum_20": signal * 0.98 + rng.normal(scale=0.03),
                    "book_to_price": -signal + rng.normal(scale=0.3),
                    "forward_return": 0.001 * signal + rng.normal(scale=0.01),
                }
            )
    return pd.DataFrame(rows)


def test_correlation_ridge_and_orthogonalization_run() -> None:
    data = _research_panel().sample(frac=1.0, random_state=42)
    factors = ["momentum_5", "momentum_20", "book_to_price"]
    correlation = factor_correlation(data, factors)
    importance = ridge_feature_importance(data, factors)
    orthogonalized = orthogonalize_factors(data, factors)

    assert correlation.shape == (3, 3)
    assert np.allclose(np.diag(correlation), 1.0, equal_nan=False)
    assert set(importance["factor"]) == set(factors)
    assert set(factors).issubset(orthogonalized.columns)


def test_selection_respects_high_correlation_and_family_limit() -> None:
    summary = pd.DataFrame(
        {
            "factor": ["momentum_5", "momentum_20", "book_to_price"],
            "ic_mean": [0.05, 0.04, 0.03],
        }
    )
    correlation = pd.DataFrame(
        [[1.0, 0.95, 0.10], [0.95, 1.0, 0.12], [0.10, 0.12, 1.0]],
        index=summary["factor"],
        columns=summary["factor"],
    )
    ridge = pd.DataFrame(
        {"factor": summary["factor"], "importance": [0.5, 0.3, 0.2]}
    )
    selected, weights = select_factors(
        summary,
        correlation,
        ridge,
        max_correlation=0.75,
        max_factors=3,
        max_factors_per_family=1,
    )

    assert selected == ["momentum_5", "book_to_price"]
    assert set(weights) == set(selected)
