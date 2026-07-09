"""CSV 数据仓库与版本清单。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class CsvDataStore:
    """使用 CSV 和 JSON 清单实现轻量、可追溯的数据存储。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, data: pd.DataFrame) -> Path:
        """原子式写入数据集并更新版本清单。"""

        path = self.root / f"{name}.csv"
        temp_path = self.root / f".{name}.tmp.csv"
        data.to_csv(temp_path, index=False, encoding="utf-8-sig")
        temp_path.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "dataset": name,
            "path": str(path),
            "rows": int(len(data)),
            "columns": list(data.columns),
            "sha256": digest,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if "date" in data.columns and not data.empty:
            dates = pd.to_datetime(data["date"], errors="coerce").dropna()
            if not dates.empty:
                manifest["date_min"] = dates.min().strftime("%Y-%m-%d")
                manifest["date_max"] = dates.max().strftime("%Y-%m-%d")
        if "symbol" in data.columns:
            manifest["symbol_count"] = int(data["symbol"].nunique())
        (self.root / f"{name}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def exists(self, name: str) -> bool:
        """判断数据集是否已经存在。"""

        return (self.root / f"{name}.csv").exists()

    def load(self, name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
        """读取已落地的数据集。"""

        return pd.read_csv(
            self.root / f"{name}.csv",
            parse_dates=parse_dates,
        )

    def missing_outer_date_ranges(
        self, name: str, start_date: str, end_date: str
    ) -> list[tuple[str, str]]:
        """返回需要向两端补拉的日期区间。

        这里刻意只检查已有数据的外侧边界，避免把中国交易日历中的节假日
        误判为“中间缺口”并触发大量无效请求。
        """

        path = self.root / f"{name}.csv"
        if not path.exists():
            return [(start_date, end_date)]

        existing = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
        if existing.empty:
            return [(start_date, end_date)]

        requested_start = pd.Timestamp(start_date)
        requested_end = pd.Timestamp(end_date)
        current_start = pd.Timestamp(existing["date"].min())
        current_end = pd.Timestamp(existing["date"].max())
        ranges: list[tuple[str, str]] = []

        if requested_start < current_start:
            end = current_start - pd.Timedelta(days=1)
            ranges.append((requested_start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        if requested_end > current_end:
            start = current_end + pd.Timedelta(days=1)
            ranges.append((start.strftime("%Y-%m-%d"), requested_end.strftime("%Y-%m-%d")))
        return ranges

    def update(self, name: str, new_data: pd.DataFrame) -> Path:
        """按 date/symbol 去重后增量更新数据。"""

        path = self.root / f"{name}.csv"
        if path.exists():
            old_data = pd.read_csv(path, parse_dates=["date"])
            new_data = pd.concat([old_data, new_data], ignore_index=True)
        keys = [column for column in ("date", "symbol") if column in new_data.columns]
        if keys:
            new_data = new_data.drop_duplicates(keys, keep="last").sort_values(keys)
        return self.save(name, new_data)
