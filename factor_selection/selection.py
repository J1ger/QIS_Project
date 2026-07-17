"""因子正交化、相关性诊断和稳定筛选。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when the input panel is missing required columns."""

    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def _factor_family(factor: str) -> str:
    """Group related horizons into the same family to reduce duplicate logic."""

    if factor.startswith("momentum_") or factor == "reversal_5":
        return "momentum"
    if factor.startswith("volatility_"):
        return "volatility"
    if factor.startswith("ma_gap_") or factor in {"rsi_14", "price_position_20"}:
        return "trend_structure"
    if factor in {
        "volume_ratio_20",
        "price_volume_divergence_20",
        "volume_price_corr_20",
        "turnover_mean_20",
        "turnover_stability_20",
        "illiquidity_20",
    }:
        return "volume_liquidity"
    if factor in {
        "earnings_yield",
        "book_to_price",
        "sales_to_price",
        "cashflow_yield",
        "dividend_yield",
    }:
        return "valuation"
    if factor in {
        "roe",
        "roa",
        "gross_margin",
        "asset_turnover",
        "current_ratio",
        "leverage",
        "accrual_ratio",
        "cash_conversion_ratio",
    }:
        return "quality"
    if factor in {"profit_growth_60", "revenue_growth_60"}:
        return "growth"
    if factor.startswith("sentiment_") or factor.startswith("northbound_") or factor == "pmi_change_20":
        return "alternative_macro"
    return factor


def factor_correlation(data: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """计算各日期截面相关矩阵的时间均值。"""

    if not factors:
        return pd.DataFrame(dtype=float)
    _require_columns(data, ["date", *factors])
    matrices = []
    for _, group in data.groupby("date"):
        if len(group) < 5:
            continue
        ranked = group[factors].rank()
        matrix = pd.DataFrame(np.nan, index=factors, columns=factors)
        valid_factors = [
            factor
            for factor in factors
            if ranked[factor].count() >= 2
            and float(ranked[factor].std(ddof=0)) > 1e-12
        ]
        if valid_factors:
            matrix.loc[valid_factors, valid_factors] = ranked[valid_factors].corr()
        matrices.append(matrix)
    if not matrices:
        return pd.DataFrame(index=factors, columns=factors, dtype=float)
    return (
        pd.concat(matrices)
        .groupby(level=0, sort=False)
        .mean()
        .reindex(index=factors, columns=factors)
    )


def ridge_feature_importance(
    data: pd.DataFrame,
    factors: list[str],
    regularization: float = 1.0,
    max_rows: int = 50_000,
) -> pd.DataFrame:
    """使用岭回归估计因子的联合收益解释能力。

    该实现仅依赖 NumPy。输入因子已完成截面标准化，系数绝对值可作为同量纲的
    特征重要性参考。为控制内存，按固定间隔抽取最多 `max_rows` 行。
    """

    if regularization <= 0:
        raise ValueError("regularization must be positive for ridge regression")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if not factors:
        return pd.DataFrame(columns=["factor", "ridge_coefficient", "importance"])
    _require_columns(data, ["forward_return", *factors])
    columns = factors + ["forward_return"]
    sample = data[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) > max_rows:
        step = max(1, len(sample) // max_rows)
        sample = sample.iloc[::step].head(max_rows)
    if sample.empty or not factors:
        return pd.DataFrame(columns=["factor", "ridge_coefficient", "importance"])

    features = sample[factors].to_numpy(dtype=float)
    target = sample["forward_return"].to_numpy(dtype=float)
    target = target - target.mean()
    identity = np.eye(features.shape[1])
    try:
        coefficients = np.linalg.solve(
            features.T @ features + regularization * identity,
            features.T @ target,
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(
            features.T @ features + regularization * identity
        ) @ features.T @ target
    importance = np.abs(coefficients)
    total = float(importance.sum())
    normalized = importance / total if total > 0 else importance
    return pd.DataFrame(
        {
            "factor": factors,
            "ridge_coefficient": coefficients,
            "importance": normalized,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def orthogonalize_factors(data: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """使用逐日 Gram-Schmidt 残差法降低共线性。"""

    if not factors:
        return data.copy()
    _require_columns(data, ["date", *factors])
    frame = data.copy()

    def transform(group: pd.DataFrame) -> pd.DataFrame:
        result = group[factors].copy()
        basis: list[np.ndarray] = []
        for factor in factors:
            values = result[factor].fillna(0).to_numpy(dtype=float)
            residual = values.copy()
            for vector in basis:
                denominator = float(vector @ vector)
                if denominator > 1e-12:
                    residual -= vector * float(residual @ vector) / denominator
            deviation = residual.std()
            if deviation > 0:
                residual = (residual - residual.mean()) / deviation
            result[factor] = residual
            basis.append(residual)
        return result

    transformed = frame.groupby("date", group_keys=False).apply(
        transform, include_groups=False
    )
    # ``group_keys=False`` preserves the source row index. Reindex directly so
    # standalone callers with non-consecutive indexes do not get values shifted.
    frame[factors] = transformed.reindex(frame.index)
    return frame


def select_factors(
    summary: pd.DataFrame,
    correlation: pd.DataFrame,
    ridge_importance: pd.DataFrame | None = None,
    max_correlation: float = 0.75,
    max_factors: int = 12,
    max_factors_per_family: int = 2,
) -> tuple[list[str], dict[str, float]]:
    """结合 IC 与岭回归重要性排序，并剔除高度相关因子。"""

    required_summary = {"factor", "ic_mean"}
    if not required_summary.issubset(summary.columns):
        raise ValueError("summary must contain factor and ic_mean columns")
    if not 0 <= max_correlation <= 1:
        raise ValueError("max_correlation must be between 0 and 1")
    if max_factors < 1 or max_factors_per_family < 1:
        raise ValueError("max_factors and max_factors_per_family must be positive")
    if summary.empty:
        return [], {}
    ranking = summary[["factor", "ic_mean"]].copy()
    candidate_factors = set(ranking["factor"])
    if not candidate_factors.issubset(correlation.index) or not candidate_factors.issubset(
        correlation.columns
    ):
        raise ValueError("correlation must contain every factor in summary on both axes")
    ranking["ic_score"] = ranking["ic_mean"].abs()
    if ridge_importance is not None and not ridge_importance.empty:
        ranking = ranking.merge(
            ridge_importance[["factor", "importance"]], on="factor", how="left"
        )
        ranking["importance"] = ranking["importance"].fillna(0)
    else:
        ranking["importance"] = 0.0
    ic_total = ranking["ic_score"].sum()
    if ic_total > 0:
        ranking["ic_score"] /= ic_total
    importance_total = ranking["importance"].sum()
    if importance_total > 0:
        ranking["importance"] /= importance_total
    ranking["combined_score"] = 0.6 * ranking["ic_score"] + 0.4 * ranking["importance"]
    ranking = ranking.sort_values("combined_score", ascending=False)
    selected: list[str] = []
    family_counts: dict[str, int] = {}
    for factor in ranking["factor"]:
        family = _factor_family(factor)
        if family_counts.get(family, 0) >= max_factors_per_family:
            continue
        if all(
            abs(float(correlation.loc[factor, existing])) < max_correlation
            for existing in selected
        ):
            selected.append(factor)
            family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= max_factors:
            break
    if not selected:
        return [], {}

    selected_ranking = ranking.set_index("factor").reindex(selected)
    directions = np.sign(
        summary.set_index("factor")["ic_mean"].reindex(selected).fillna(0)
    )
    score_total = float(selected_ranking["combined_score"].sum())
    if score_total == 0:
        weights = {factor: 1 / len(selected) for factor in selected}
    else:
        weights = {
            factor: float(
                directions.loc[factor]
                * selected_ranking.loc[factor, "combined_score"]
                / score_total
            )
            for factor in selected
        }
    return selected, weights
