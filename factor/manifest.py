"""因子版本清单与可追溯信息。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


FACTOR_METADATA: dict[str, dict[str, str]] = {
    "momentum_5": {"category": "technical", "version": "1.0.0", "description": "5 日动量收益"},
    "momentum_20": {"category": "technical", "version": "1.0.0", "description": "20 日动量收益"},
    "momentum_60": {"category": "technical", "version": "1.0.0", "description": "60 日动量收益"},
    "reversal_5": {"category": "technical", "version": "1.0.0", "description": "5 日短期反转"},
    "volatility_20": {"category": "technical", "version": "1.0.0", "description": "20 日收益波动率"},
    "volatility_60": {"category": "technical", "version": "1.0.0", "description": "60 日收益波动率"},
    "ma_gap_5": {"category": "technical", "version": "1.0.0", "description": "收盘价相对 5 日均线偏离"},
    "ma_gap_20": {"category": "technical", "version": "1.0.0", "description": "收盘价相对 20 日均线偏离"},
    "ma_gap_60": {"category": "technical", "version": "1.0.0", "description": "收盘价相对 60 日均线偏离"},
    "rsi_14": {"category": "technical", "version": "1.0.0", "description": "14 日 RSI 强弱指标"},
    "price_position_20": {"category": "technical", "version": "1.0.0", "description": "20 日价格区间位置"},
    "volume_ratio_20": {"category": "technical", "version": "1.0.0", "description": "成交量相对 20 日均量"},
    "price_volume_divergence_20": {"category": "technical", "version": "1.0.0", "description": "20 日量价背离因子"},
    "volume_price_corr_20": {"category": "technical", "version": "1.0.0", "description": "20 日收益-量变相关性"},
    "turnover_mean_20": {"category": "technical", "version": "1.0.0", "description": "20 日平均换手率"},
    "turnover_stability_20": {"category": "technical", "version": "1.0.0", "description": "20 日换手波动率"},
    "illiquidity_20": {"category": "technical", "version": "1.0.0", "description": "20 日非流动性指标"},
    "max_return_20": {"category": "technical", "version": "1.0.0", "description": "20 日最大单日收益"},
    "min_return_20": {"category": "technical", "version": "1.0.0", "description": "20 日最小单日收益"},
    "market_beta_60": {"category": "technical", "version": "1.0.0", "description": "60 日市场 Beta"},
    "earnings_yield": {"category": "fundamental", "version": "1.0.0", "description": "盈利收益率"},
    "book_to_price": {"category": "fundamental", "version": "1.0.0", "description": "账面市值比"},
    "sales_to_price": {"category": "fundamental", "version": "1.0.0", "description": "销售市值比"},
    "cashflow_yield": {"category": "fundamental", "version": "1.0.0", "description": "经营现金流收益率"},
    "accrual_ratio": {"category": "fundamental", "version": "1.0.0", "description": "应计利润占总资产比"},
    "cash_conversion_ratio": {"category": "fundamental", "version": "1.0.0", "description": "经营现金流/净利润"},
    "roe": {"category": "fundamental", "version": "1.0.0", "description": "净资产收益率"},
    "roa": {"category": "fundamental", "version": "1.0.0", "description": "总资产收益率"},
    "gross_margin": {"category": "fundamental", "version": "1.0.0", "description": "毛利率"},
    "asset_turnover": {"category": "fundamental", "version": "1.0.0", "description": "营业收入/总资产"},
    "current_ratio": {"category": "fundamental", "version": "1.0.0", "description": "流动比率"},
    "leverage": {"category": "fundamental", "version": "1.0.0", "description": "资产负债率"},
    "dividend_yield": {"category": "fundamental", "version": "1.0.0", "description": "股息收益率"},
    "profit_growth_60": {"category": "fundamental", "version": "1.0.0", "description": "利润同比增长近似指标"},
    "revenue_growth_60": {"category": "fundamental", "version": "1.0.0", "description": "收入同比增长近似指标"},
    "log_size": {"category": "style", "version": "1.0.0", "description": "对数市值规模"},
    "sentiment_5": {"category": "alternative", "version": "1.0.0", "description": "5 日量价情绪"},
    "sentiment_20": {"category": "alternative", "version": "1.0.0", "description": "20 日量价情绪"},
    "northbound_5": {"category": "alternative", "version": "1.0.0", "description": "5 日北向资金强度"},
    "northbound_20": {"category": "alternative", "version": "1.0.0", "description": "20 日北向资金强度"},
    "pmi_change_20": {"category": "macro", "version": "1.0.0", "description": "PMI 变化敏感度"},
}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return _sha256_bytes(path.read_bytes())


def _frame_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return ""
    normalized = frame[available].copy()
    for column in normalized.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        normalized[column] = normalized[column].dt.strftime("%Y-%m-%d")
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _sha256_bytes(payload)


def _settings_section(settings: Any, key: str) -> Any:
    """Support both object-style Settings and plain dict configs."""

    if isinstance(settings, dict):
        return settings[key]
    return getattr(settings, key)


def build_factor_manifest(
    *,
    raw_factors: pd.DataFrame,
    processed_factors: pd.DataFrame,
    factor_names: list[str],
    settings: Any,
) -> dict[str, Any]:
    """构建因子版本、代码与数据血缘清单。"""

    root_dir = settings["root_dir"] if isinstance(settings, dict) else settings.root_dir
    root = Path(root_dir)
    source_files = {
        "factor_library": root / "library.py",
        "factor_preprocess": root / "preprocess.py",
        "factor_manifest": root / "manifest.py",
    }
    project = _settings_section(settings, "project")
    data = _settings_section(settings, "data")
    factor = _settings_section(settings, "factor")
    identity_columns = ["date", "symbol"]
    factor_columns = identity_columns + factor_names
    factors: list[dict[str, Any]] = []

    for factor in factor_names:
        metadata = FACTOR_METADATA.get(
            factor,
            {"category": "unknown", "version": "1.0.0", "description": ""},
        )
        raw_missing = (
            float(raw_factors[factor].isna().mean()) if factor in raw_factors else 1.0
        )
        processed_missing = (
            float(processed_factors[factor].isna().mean())
            if factor in processed_factors
            else 1.0
        )
        factors.append(
            {
                "factor": factor,
                "category": metadata["category"],
                "version": metadata["version"],
                "description": metadata["description"],
                "raw_missing_ratio": raw_missing,
                "processed_missing_ratio": processed_missing,
                "preprocess": {
                    "winsor_lower": factor["winsor_lower"],
                    "winsor_upper": factor["winsor_upper"],
                    "neutralize": factor["neutralize"],
                    "min_cross_section": factor["min_cross_section"],
                },
            }
        )

    manifest = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": project["name"],
        "data_provider": data["provider"],
        "sample_range": {
            "start_date": data["start_date"],
            "end_date": data["end_date"],
        },
        "factor_count": len(factor_names),
        "factor_library_version_hash": _frame_digest(
            pd.DataFrame(factors), ["factor", "category", "version", "description"]
        ),
        "raw_factor_data_hash": _frame_digest(raw_factors, factor_columns),
        "processed_factor_data_hash": _frame_digest(processed_factors, factor_columns),
        "source_code_hashes": {
            name: _sha256_file(path) for name, path in source_files.items()
        },
        "preprocess_config": {
            "winsor_lower": factor["winsor_lower"],
            "winsor_upper": factor["winsor_upper"],
            "neutralize": factor["neutralize"],
            "min_cross_section": factor["min_cross_section"],
        },
        "factors": factors,
    }
    return json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=True))


def factor_manifest_frame(manifest: dict[str, Any]) -> pd.DataFrame:
    """将 JSON 清单展开为便于人工检查的表格。"""

    rows = []
    for item in manifest["factors"]:
        preprocess = item.get("preprocess", {})
        rows.append(
            {
                "factor": item["factor"],
                "category": item["category"],
                "version": item["version"],
                "description": item["description"],
                "raw_missing_ratio": item["raw_missing_ratio"],
                "processed_missing_ratio": item["processed_missing_ratio"],
                "winsor_lower": preprocess.get("winsor_lower"),
                "winsor_upper": preprocess.get("winsor_upper"),
                "neutralize": preprocess.get("neutralize"),
                "min_cross_section": preprocess.get("min_cross_section"),
            }
        )
    return pd.DataFrame(rows)
