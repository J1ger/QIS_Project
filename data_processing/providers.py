"""标准市场数据接口及 AkShare 数据适配器。"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from exceptions import ProviderError


class MarketDataProvider(ABC):
    """市场数据源统一接口。"""

    @abstractmethod
    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取标准化证券日频面板数据。"""


@dataclass
class SyntheticProvider(MarketDataProvider):
    """生成具有行业、风格和市场结构的可复现实验数据。"""

    symbol_count: int = 80
    random_seed: int = 42

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """生成完整的 OHLCV、财务与另类字段。"""

        rng = np.random.default_rng(self.random_seed)
        dates = pd.bdate_range(start_date, end_date)
        if len(dates) < 120:
            raise ProviderError("合成数据至少需要 120 个交易日")

        symbols = [f"{600000 + i:06d}.SH" for i in range(self.symbol_count)]
        industries = np.array(["金融", "消费", "医药", "科技", "工业", "材料"])
        market_return = rng.normal(0.00025, 0.012, len(dates))
        pmi = 50 + np.cumsum(rng.normal(0, 0.08, len(dates)))
        risk_free = np.clip(0.02 + np.cumsum(rng.normal(0, 0.00003, len(dates))), 0.01, 0.04)
        records: list[pd.DataFrame] = []

        for index, symbol in enumerate(symbols):
            industry = industries[index % len(industries)]
            beta = rng.uniform(0.7, 1.3)
            quality = rng.normal()
            value = rng.normal()
            idiosyncratic = rng.normal(0, 0.018, len(dates))
            returns = 0.00008 * quality + beta * market_return + idiosyncratic
            close = rng.uniform(8, 35) * np.exp(np.cumsum(returns))
            open_price = close * (1 + rng.normal(0, 0.003, len(dates)))
            high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.018, len(dates)))
            low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.018, len(dates)))
            shares = rng.uniform(2e8, 2e9)
            volume = rng.lognormal(15.2, 0.55, len(dates))
            amount = volume * close
            market_cap = close * shares
            revenue = market_cap * rng.uniform(0.25, 1.0) * (1 + 0.00015 * np.arange(len(dates)))
            net_margin = np.clip(0.08 + 0.025 * quality + rng.normal(0, 0.01), 0.01, 0.25)
            net_profit = revenue * net_margin
            book_value = market_cap / np.clip(1.8 + 0.45 * value, 0.6, 5)
            assets = book_value * rng.uniform(1.6, 4.5)
            liabilities = assets - book_value
            operating_cashflow = net_profit * rng.uniform(0.75, 1.25)
            gross_profit = revenue * np.clip(0.25 + 0.05 * quality, 0.1, 0.6)
            current_assets = assets * rng.uniform(0.25, 0.55)
            current_liabilities = liabilities * rng.uniform(0.3, 0.7)
            dividend = np.maximum(net_profit * rng.uniform(0.05, 0.35), 0)
            sentiment = (
                pd.Series(returns).rolling(5, min_periods=1).mean().to_numpy() * 20
                + rng.normal(0, 0.4, len(dates))
            )
            northbound = market_cap * (
                0.0002 * np.sign(pd.Series(returns).rolling(10, min_periods=1).mean())
                + rng.normal(0, 0.0004, len(dates))
            )
            suspended = rng.random(len(dates)) < 0.002
            observed_return = pd.Series(close).pct_change().fillna(0).to_numpy()

            records.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "symbol": symbol,
                        "industry": industry,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "amount": amount,
                        "turnover_rate": volume / shares,
                        "market_cap": market_cap,
                        "revenue": revenue,
                        "net_profit": net_profit,
                        "book_value": book_value,
                        "assets": assets,
                        "liabilities": liabilities,
                        "operating_cashflow": operating_cashflow,
                        "gross_profit": gross_profit,
                        "current_assets": current_assets,
                        "current_liabilities": current_liabilities,
                        "dividend": dividend,
                        "sentiment": sentiment,
                        "northbound_flow": northbound,
                        "macro_pmi": pmi,
                        "risk_free_rate": risk_free,
                        "is_suspended": suspended,
                        "is_st": rng.random() < 0.04,
                        "limit_up": observed_return >= 0.095,
                        "limit_down": observed_return <= -0.095,
                        "data_source": "synthetic",
                    }
                )
            )

        return pd.concat(records, ignore_index=True).sort_values(
            ["date", "symbol"]
        ).reset_index(drop=True)


@dataclass
class AkShareProvider(MarketDataProvider):
    """从 AkShare 获取行情、财务、行业、北向资金和 PMI 数据。

    财务摘要接口仅给出报告期，未提供统一公告日期。本适配器使用可配置的保守
    滞后天数作为可得日期，避免直接在报告期末使用财务数据。
    """

    symbols: list[str] = field(default_factory=list)
    universe_index: str = "000300"
    max_symbols: int = 300
    adjust: str = "qfq"
    cache_dir: str = "data/akshare_cache"
    refresh_cache: bool = False
    request_interval: float = 0.25
    retries: int = 3
    financial_lag_days: int = 120
    risk_free_rate: float = 0.02
    require_membership_before_start: bool = True
    use_actual_notice_date: bool = True
    membership_file: str = ""
    _statement_cache: dict[tuple[str, str], pd.DataFrame] = field(
        init=False, repr=False, default_factory=dict
    )
    _northbound_cache: dict[tuple[str, ...], pd.DataFrame] = field(
        init=False, repr=False, default_factory=dict
    )
    _suspension_cache: dict[tuple[str, ...], pd.DataFrame] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.adjust = str(self.adjust).lower()
        if self.adjust not in {"qfq", "hfq"}:
            raise ProviderError(
                "AkShareProvider requires qfq or hfq adjusted prices for reliable technical factors."
            )

    @staticmethod
    def _raw_code(symbol: str) -> str:
        return str(symbol).split(".")[0].replace("sh", "").replace("sz", "").replace("bj", "")

    @staticmethod
    def _suffix(code: str) -> str:
        if code.startswith(("4", "8", "92")):
            return "BJ"
        if code.startswith(("5", "6", "9")):
            return "SH"
        return "SZ"

    @classmethod
    def _standard_symbol(cls, symbol: str) -> str:
        code = cls._raw_code(symbol).zfill(6)
        return f"{code}.{cls._suffix(code)}"

    @classmethod
    def _vendor_symbol(cls, symbol: str) -> str:
        code = cls._raw_code(symbol).zfill(6)
        return f"{cls._suffix(code).lower()}{code}"

    def _retry(self, label: str, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                result = loader()
                time.sleep(self.request_interval)
                return result
            except Exception as exc:  # AkShare 下游站点异常类型不统一
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.request_interval * (2**attempt))
        raise ProviderError(f"AkShare 接口 {label} 连续失败: {last_error}") from last_error

    def _cache_file(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        safe_name = "".join(char if char.isalnum() else "_" for char in key)[:45]
        return self.cache_path / f"{safe_name}_{digest}.csv"

    def _cached(self, key: str, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        path = self._cache_file(key)
        if path.exists() and not self.refresh_cache:
            return pd.read_csv(path)
        frame = self._retry(key, loader)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return frame

    def _load_membership_file(
        self, membership_path: Path
    ) -> tuple[
        list[str],
        dict[str, str],
        dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]],
    ]:
        """读取历史成分股区间文件。"""

        membership = pd.read_csv(membership_path)
        required = {"symbol", "effective_from"}
        missing = required.difference(membership.columns)
        if missing:
            raise ProviderError(f"历史成员文件缺少字段: {sorted(missing)}")

        membership["symbol"] = membership["symbol"].map(self._standard_symbol)
        membership["effective_from"] = pd.to_datetime(
            membership["effective_from"], errors="raise"
        )
        if "effective_to" not in membership:
            membership["effective_to"] = pd.NaT
        else:
            membership["effective_to"] = pd.to_datetime(
                membership["effective_to"], errors="coerce"
            )

        membership = membership.sort_values(["effective_from", "symbol"])
        membership_periods: dict[
            str, list[tuple[pd.Timestamp, pd.Timestamp | None]]
        ] = {}
        for symbol, group in membership.groupby("symbol", sort=False):
            membership_periods[symbol] = [
                (
                    pd.Timestamp(row.effective_from),
                    pd.Timestamp(row.effective_to) if pd.notna(row.effective_to) else None,
                )
                for row in group.itertuples()
            ]

        names: dict[str, str] = {}
        if "name" in membership:
            names = (
                membership.dropna(subset=["name"])
                .drop_duplicates("symbol", keep="last")
                .set_index("symbol")["name"]
                .astype(str)
                .to_dict()
            )

        symbols = list(dict.fromkeys(membership["symbol"]))[: self.max_symbols]
        return symbols, names, {symbol: membership_periods[symbol] for symbol in symbols}

    def _membership_file_needs_refresh(self, membership: pd.DataFrame) -> bool:
        """Refresh stale auto-generated membership files before loading them."""

        symbol_count = (
            int(membership["symbol"].nunique())
            if "symbol" in membership.columns and not membership.empty
            else 0
        )
        if symbol_count >= self.max_symbols:
            return False
        if "source" not in membership.columns:
            return False
        sources = membership["source"].dropna().astype(str)
        if sources.empty:
            return False
        return bool(
            sources.str.startswith("akshare.").all()
            or (sources == "akshare_cached_market_daily").all()
        )

    def _index_membership_frame(self, ak: Any) -> tuple[pd.DataFrame, str, str | None, str | None]:
        """Fetch the latest full constituent list and identify key columns."""

        full = self._cached(
            f"index_stock_cons_csindex_{self.universe_index}",
            lambda: ak.index_stock_cons_csindex(symbol=self.universe_index),
        )
        code_column = next(
            (
                column
                for column in ("成分券代码", "品种代码", "代码")
                if column in full.columns
            ),
            None,
        )
        if code_column is None:
            raise ProviderError(f"指数成分接口缺少代码列: {full.columns.tolist()}")
        name_column = next(
            (
                column
                for column in ("成分券名称", "品种名称", "名称")
                if column in full.columns
            ),
            None,
        )

        history = self._cached(
            f"index_stock_cons_{self.universe_index}",
            lambda: ak.index_stock_cons(symbol=self.universe_index),
        )
        history_code_column = next(
            (
                column
                for column in ("品种代码", "成分券代码", "代码")
                if column in history.columns
            ),
            None,
        )
        history_membership_column = next(
            (
                column
                for column in ("纳入日期", "日期", "生效日期")
                if column in history.columns
            ),
            None,
        )
        if history_code_column and history_membership_column:
            dates = history[[history_code_column, history_membership_column]].copy()
            dates[history_code_column] = (
                dates[history_code_column].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
            )
            dates[history_membership_column] = pd.to_datetime(
                dates[history_membership_column], errors="coerce"
            )
            dates = (
                dates.dropna(subset=[history_code_column])
                .sort_values(history_membership_column)
                .drop_duplicates(history_code_column, keep="last")
                .rename(columns={history_code_column: code_column})
            )
            full = full.merge(dates, on=code_column, how="left")

        code_column = next(
            (
                column
                for column in ("成分券代码", "品种代码", "代码")
                if column in full.columns
            ),
            None,
        )
        membership_column = next(
            (
                column
                for column in ("纳入日期", "日期", "生效日期")
                if column in full.columns
            ),
            None,
        )
        return full, code_column, name_column, membership_column

    def _write_membership_file(
        self,
        membership_path: Path,
        frame: pd.DataFrame,
        code_column: str,
        name_column: str | None,
        membership_column: str | None,
        start_date: str,
    ) -> None:
        """从 AkShare 指数当前成分接口生成可维护的历史成分文件。"""

        membership_path.parent.mkdir(parents=True, exist_ok=True)
        generated = pd.DataFrame()
        generated["symbol"] = frame[code_column].astype(str).map(self._standard_symbol)
        generated["name"] = (
            frame[name_column].astype(str).to_numpy() if name_column else ""
        )
        if membership_column:
            generated["effective_from"] = pd.to_datetime(
                frame[membership_column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        else:
            generated["effective_from"] = start_date
        generated["effective_from"] = generated["effective_from"].fillna(start_date)
        generated["effective_to"] = ""
        generated["source"] = (
            f"akshare.index_stock_cons_csindex+index_stock_cons:{self.universe_index}"
        )
        generated["history_scope"] = "current_constituents_with_entry_dates_only"
        generated["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        generated = generated.drop_duplicates(
            ["symbol", "effective_from"], keep="last"
        ).sort_values(["effective_from", "symbol"])
        generated.to_csv(membership_path, index=False, encoding="utf-8-sig")

    def _universe(
        self, start_date: str
    ) -> tuple[
        list[str],
        dict[str, str],
        dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]],
    ]:
        import akshare as ak

        names: dict[str, str] = {}
        membership_periods: dict[
            str, list[tuple[pd.Timestamp, pd.Timestamp | None]]
        ] = {}
        membership_path = Path(self.membership_file) if self.membership_file else None
        if membership_path and membership_path.exists():
            membership = pd.read_csv(membership_path)
            if self._membership_file_needs_refresh(membership):
                frame, code_column, name_column, membership_column = (
                    self._index_membership_frame(ak)
                )
                self._write_membership_file(
                    membership_path,
                    frame,
                    code_column,
                    name_column,
                    membership_column,
                    start_date,
                )
            return self._load_membership_file(membership_path)
        if self.symbols:
            symbols = [self._standard_symbol(symbol) for symbol in self.symbols]
            membership_periods = {
                symbol: [(pd.Timestamp(start_date), None)] for symbol in symbols
            }
            return symbols[: self.max_symbols], names, membership_periods

        frame, code_column, name_column, membership_column = self._index_membership_frame(
            ak
        )
        if membership_path and not membership_path.exists():
            self._write_membership_file(
                membership_path,
                frame,
                code_column,
                name_column,
                membership_column,
                start_date,
            )
            return self._load_membership_file(membership_path)
        if membership_column:
            frame[membership_column] = pd.to_datetime(
                frame[membership_column], errors="coerce"
            )
            if self.require_membership_before_start:
                eligible = frame[membership_column].isna() | (
                    frame[membership_column] <= pd.Timestamp(start_date)
                )
                frame = frame.loc[eligible].copy()
        symbols = [self._standard_symbol(value) for value in frame[code_column].astype(str)]
        if name_column:
            names = {
                self._standard_symbol(code): str(name)
                for code, name in zip(frame[code_column], frame[name_column])
            }
        if membership_column:
            membership_periods = {
                self._standard_symbol(code): [(pd.Timestamp(date), None)]
                for code, date in zip(frame[code_column], frame[membership_column])
                if pd.notna(date)
            }
        if len(symbols) < self.max_symbols:
            raise ProviderError(
                f"满足历史成员条件的证券仅 {len(symbols)} 只，少于 max_symbols="
                f"{self.max_symbols}；请改用历史成分文件或显式 symbols"
            )
        return symbols[: self.max_symbols], names, membership_periods

    def _fetch_price(
        self, ak: Any, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        code = self._raw_code(symbol)
        vendor_symbol = self._vendor_symbol(symbol)
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        errors: list[str] = []

        loaders: list[tuple[str, Callable[[], pd.DataFrame]]] = [
            (
                "sina",
                lambda: ak.stock_zh_a_daily(
                    symbol=vendor_symbol,
                    start_date=start,
                    end_date=end,
                    adjust=self.adjust,
                ),
            ),
            (
                "eastmoney",
                lambda: ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust=self.adjust,
                    timeout=20,
                ),
            ),
            (
                "tencent",
                lambda: ak.stock_zh_a_hist_tx(
                    symbol=vendor_symbol,
                    start_date=start,
                    end_date=end,
                    adjust=self.adjust,
                    timeout=20,
                ),
            ),
        ]
        raw: pd.DataFrame | None = None
        source = ""
        for source, loader in loaders:
            try:
                raw = self._cached(
                    f"price_{source}_{code}_{start}_{end}_{self.adjust}", loader
                )
                if not raw.empty:
                    break
            except ProviderError as exc:
                errors.append(str(exc))
                raw = None
        if raw is None or raw.empty:
            raise ProviderError(f"{symbol} 行情获取失败: {'; '.join(errors)}")

        frame = raw.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover_rate",
                "outstanding_share": "outstanding_share",
                "turnover": "turnover_rate",
            }
        ).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        numeric = ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]
        for column in numeric:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if source == "eastmoney":
            frame["turnover_rate"] = frame["turnover_rate"] / 100
            frame["outstanding_share"] = frame["volume"] / frame["turnover_rate"].replace(0, np.nan)
        elif source == "tencent":
            frame["volume"] = frame["amount"] * 100
            frame["amount"] = frame["volume"] * frame["close"]
            frame["turnover_rate"] = np.nan
            frame["outstanding_share"] = np.nan

        frame["symbol"] = symbol
        frame["price_source"] = source
        frame["price_adjustment"] = self.adjust
        frame["adjustment_applied"] = True
        columns = [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover_rate",
            "outstanding_share",
            "price_source",
            "price_adjustment",
            "adjustment_applied",
        ]
        return frame[[column for column in columns if column in frame.columns]]

    @staticmethod
    def _first_existing_column(
        frame: pd.DataFrame, candidates: tuple[str, ...] | list[str]
    ) -> str | None:
        for column in candidates:
            if column in frame.columns:
                return column
        return None

    @staticmethod
    def _numeric_series(series: pd.Series) -> pd.Series:
        text = series.astype(str).str.replace(",", "", regex=False).str.strip()
        text = text.replace(
            {
                "None": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "NULL": np.nan,
                "--": np.nan,
                "": np.nan,
            }
        )
        return pd.to_numeric(text, errors="coerce")

    @staticmethod
    def _quarter_report_dates(
        start_date: str, end_date: str, pad_years: int = 2
    ) -> list[str]:
        start = pd.Timestamp(start_date) - pd.DateOffset(years=pad_years)
        end = pd.Timestamp(end_date)
        report_dates: list[str] = []
        for year in range(start.year, end.year + 1):
            for suffix in ("0331", "0630", "0930", "1231"):
                report_date = pd.Timestamp(f"{year}-{suffix[:2]}-{suffix[2:]}")
                if start <= report_date <= end:
                    report_dates.append(report_date.strftime("%Y%m%d"))
        return report_dates

    @staticmethod
    def _date_chunks(
        start_date: str, end_date: str, chunk_days: int = 120
    ) -> list[tuple[str, str]]:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        chunks: list[tuple[str, str]] = []
        current = start
        while current <= end:
            chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end)
            chunks.append(
                (current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d"))
            )
            current = chunk_end + pd.Timedelta(days=1)
        return chunks

    def _statement_snapshot_real(
        self, ak: Any, statement_type: str, report_date: str
    ) -> pd.DataFrame:
        cache_key = (statement_type, report_date)
        if cache_key in self._statement_cache:
            return self._statement_cache[cache_key].copy()

        if statement_type == "balance":
            raw = self._cached(
                f"stock_zcfz_em_{report_date}",
                lambda: ak.stock_zcfz_em(date=report_date),
            )
            mappings = {
                "assets": ["资产-总资产", "资产总计", "总资产"],
                "liabilities": ["负债-总负债", "负债合计", "总负债"],
                "book_value": [
                    "股东权益合计",
                    "所有者权益合计",
                    "归属于母公司股东权益合计",
                    "股东权益(不含少数股东权益)",
                ],
                "current_assets": ["资产-流动资产", "流动资产合计", "流动资产"],
                "current_liabilities": ["负债-流动负债", "流动负债合计", "流动负债"],
            }
        elif statement_type == "income":
            raw = self._cached(
                f"stock_lrb_em_{report_date}",
                lambda: ak.stock_lrb_em(date=report_date),
            )
            mappings = {
                "revenue": ["营业总收入", "营业收入"],
                "net_profit": ["净利润", "归属于母公司股东的净利润", "归母净利润"],
                "operating_cost": ["营业总支出-营业支出", "营业支出", "营业成本"],
            }
        elif statement_type == "cashflow":
            raw = self._cached(
                f"stock_xjll_em_{report_date}",
                lambda: ak.stock_xjll_em(date=report_date),
            )
            mappings = {
                "operating_cashflow": [
                    "经营活动产生的现金流量净额",
                    "经营性现金流-现金流量净额",
                ]
            }
        else:
            raise ProviderError(f"未知财务报表类型: {statement_type}")

        if raw.empty:
            empty = pd.DataFrame()
            self._statement_cache[cache_key] = empty
            return empty.copy()

        code_column = self._first_existing_column(
            raw, ["股票代码", "证券代码", "代码"]
        )
        if code_column is None:
            raise ProviderError(
                f"{statement_type} 报表缺少股票代码列: {raw.columns.tolist()}"
            )
        notice_column = self._first_existing_column(
            raw, ["公告日期", "最新公告日期", "更新日期", "发布日期"]
        )

        frame = pd.DataFrame(
            {
                "symbol": raw[code_column].astype(str).map(self._standard_symbol),
                "report_date": pd.Timestamp(report_date),
            }
        )
        for target, candidates in mappings.items():
            source_column = self._first_existing_column(raw, candidates)
            frame[target] = (
                self._numeric_series(raw[source_column])
                if source_column
                else np.nan
            )
        if statement_type == "income":
            frame["gross_profit"] = frame["revenue"] - frame["operating_cost"]
        frame[f"available_date_{statement_type}"] = (
            pd.to_datetime(raw[notice_column], errors="coerce")
            if notice_column
            else pd.NaT
        )
        frame = frame.drop_duplicates(["symbol", "report_date"], keep="last")
        self._statement_cache[cache_key] = frame
        return frame.copy()

    def _balance_sheet_report_detail_real(self, ak: Any, symbol: str) -> pd.DataFrame:
        vendor_symbol = self._vendor_symbol(symbol)
        raw = self._cached(
            f"stock_balance_sheet_by_report_em_{vendor_symbol}",
            lambda current=vendor_symbol: ak.stock_balance_sheet_by_report_em(
                symbol=current
            ),
        )
        if raw.empty:
            return pd.DataFrame()

        frame = pd.DataFrame(
            {
                "report_date": pd.to_datetime(raw.get("REPORT_DATE"), errors="coerce"),
                "available_date_balance_detail": pd.to_datetime(
                    raw.get("NOTICE_DATE"), errors="coerce"
                ),
            }
        )
        mappings = {
            "assets": ["TOTAL_ASSETS", "ASSET_BALANCE"],
            "liabilities": ["TOTAL_LIABILITIES", "LIAB_BALANCE"],
            "book_value": [
                "TOTAL_PARENT_EQUITY",
                "PARENT_EQUITY_BALANCE",
                "TOTAL_EQUITY",
                "EQUITY_BALANCE",
            ],
            "current_assets": ["TOTAL_CURRENT_ASSETS", "CURRENT_ASSET_BALANCE"],
            "current_liabilities": ["TOTAL_CURRENT_LIAB", "CURRENT_LIAB_BALANCE"],
        }
        for target, candidates in mappings.items():
            source_column = self._first_existing_column(raw, candidates)
            frame[target] = (
                self._numeric_series(raw[source_column])
                if source_column is not None
                else np.nan
            )
        return frame.dropna(subset=["report_date"]).drop_duplicates(
            "report_date", keep="last"
        )

    def _financials_real(
        self, ak: Any, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        quarterly_rows: list[pd.DataFrame] = []
        for report_date in self._quarter_report_dates(start_date, end_date):
            balance = self._statement_snapshot_real(ak, "balance", report_date)
            income = self._statement_snapshot_real(ak, "income", report_date)
            cashflow = self._statement_snapshot_real(ak, "cashflow", report_date)

            quarterly: pd.DataFrame | None = None
            for snapshot in (balance, income, cashflow):
                if snapshot.empty:
                    continue
                row = snapshot.loc[snapshot["symbol"] == symbol].copy()
                if row.empty:
                    continue
                quarterly = (
                    row
                    if quarterly is None
                    else quarterly.merge(row, on=["symbol", "report_date"], how="outer")
                )
            if quarterly is not None and not quarterly.empty:
                quarterly_rows.append(quarterly)

        if not quarterly_rows:
            return pd.DataFrame()

        frame = pd.concat(quarterly_rows, ignore_index=True, sort=False)
        detail_balance = self._balance_sheet_report_detail_real(ak, symbol)
        if not detail_balance.empty:
            frame = frame.merge(detail_balance, on="report_date", how="outer")
            for column in (
                "assets",
                "liabilities",
                "book_value",
                "current_assets",
                "current_liabilities",
            ):
                detail_column = f"{column}_y"
                base_column = f"{column}_x"
                if base_column in frame and detail_column in frame:
                    frame[column] = frame[base_column].fillna(frame[detail_column])
                    frame = frame.drop(columns=[base_column, detail_column])
                elif detail_column in frame:
                    frame[column] = frame[detail_column]
                    frame = frame.drop(columns=[detail_column])
                elif base_column in frame:
                    frame[column] = frame[base_column]
                    frame = frame.drop(columns=[base_column])
            if "available_date_balance_detail" in frame:
                if "available_date_balance" in frame:
                    frame["available_date_balance"] = frame[
                        "available_date_balance"
                    ].fillna(frame["available_date_balance_detail"])
                else:
                    frame["available_date_balance"] = frame[
                        "available_date_balance_detail"
                    ]
                frame = frame.drop(columns=["available_date_balance_detail"])
        available_columns = [
            column for column in frame.columns if column.startswith("available_date_")
        ]
        if self.use_actual_notice_date and available_columns:
            available_frame = frame[available_columns].apply(
                pd.to_datetime, errors="coerce"
            )
            frame["available_date"] = available_frame.max(axis=1)
        else:
            frame["available_date"] = pd.NaT
        fallback_available = frame["report_date"] + pd.to_timedelta(
            self.financial_lag_days, unit="D"
        )
        frame["available_date"] = frame["available_date"].fillna(fallback_available)
        keep_columns = [
            "report_date",
            "available_date",
            "net_profit",
            "revenue",
            "book_value",
            "assets",
            "liabilities",
            "operating_cashflow",
            "gross_profit",
            "current_assets",
            "current_liabilities",
        ]
        return (
            frame[[column for column in keep_columns if column in frame.columns]]
            .drop_duplicates("report_date", keep="last")
            .sort_values("available_date")
        )

    def _sentiment_history(self, ak: Any, symbol: str) -> pd.DataFrame:
        code = self._raw_code(symbol)
        raw = self._cached(
            f"stock_comment_focus_{code}",
            lambda: ak.stock_comment_detail_scrd_focus_em(symbol=code),
        )
        if raw.empty:
            return pd.DataFrame()
        date_column = self._first_existing_column(raw, ["交易日", "日期", "时间"])
        sentiment_column = self._first_existing_column(
            raw, ["用户关注指数", "关注指数", "关注度"]
        )
        if date_column is None or sentiment_column is None:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[date_column], errors="coerce"),
                "sentiment": self._numeric_series(raw[sentiment_column]),
                "sentiment_source": "stock_comment_detail_scrd_focus_em",
            }
        )
        return frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")

    def _northbound_history(
        self, ak: Any, symbols: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        cache_key = tuple(sorted(symbols)) + (start_date, end_date)
        if cache_key in self._northbound_cache:
            return self._northbound_cache[cache_key].copy()

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            code = self._raw_code(symbol)
            try:
                raw = self._cached(
                    f"stock_hsgt_individual_{code}",
                    lambda current=code: ak.stock_hsgt_individual_em(symbol=current),
                )
            except ProviderError:
                continue
            if raw.empty:
                continue
            date_column = self._first_existing_column(
                raw, ["持股日期", "日期", "交易日期"]
            )
            flow_column = self._first_existing_column(
                raw,
                [
                    "今日持股市值变化",
                    "今日增持资金",
                    "持股市值变化-1日",
                    "持股市值变化-1日(元)",
                    "持股市值变化",
                ],
            )
            if date_column is None:
                continue
            frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(raw[date_column], errors="coerce"),
                    "symbol": self._standard_symbol(symbol),
                }
            )
            if flow_column is not None:
                frame["northbound_flow"] = self._numeric_series(raw[flow_column])
            else:
                holding_value_column = self._first_existing_column(
                    raw, ["持股市值", "持股市值(元)"]
                )
                if holding_value_column is None:
                    continue
                frame["northbound_holding_value"] = self._numeric_series(
                    raw[holding_value_column]
                )
                frame = frame.sort_values(["symbol", "date"])
                frame["northbound_flow"] = frame.groupby("symbol")[
                    "northbound_holding_value"
                ].diff()
                frame = frame.drop(columns=["northbound_holding_value"])
            frame = frame.dropna(subset=["date"])
            frame = frame[
                (frame["date"] >= start_ts) & (frame["date"] <= end_ts)
            ]
            if frame.empty:
                continue
            frames.append(frame[["date", "symbol", "northbound_flow"]])

        result = (
            pd.concat(frames, ignore_index=True)
            .dropna(subset=["date"])
            .drop_duplicates(["date", "symbol"], keep="last")
            .sort_values(["date", "symbol"])
            if frames
            else pd.DataFrame(columns=["date", "symbol", "northbound_flow"])
        )
        self._northbound_cache[cache_key] = result
        return result.copy()

    def _official_suspension_flags(
        self, ak: Any, trading_dates: list[pd.Timestamp], symbols: list[str]
    ) -> pd.DataFrame:
        cache_key = tuple(pd.Timestamp(date).strftime("%Y%m%d") for date in trading_dates)
        if cache_key in self._suspension_cache:
            return self._suspension_cache[cache_key].copy()

        symbol_set = set(symbols)
        frames: list[pd.DataFrame] = []
        for date in trading_dates:
            query_date = pd.Timestamp(date).strftime("%Y%m%d")
            try:
                raw = self._cached(
                    f"stock_tfp_em_{query_date}",
                    lambda current_date=query_date: ak.stock_tfp_em(date=current_date),
                )
            except ProviderError:
                continue
            if raw.empty:
                continue
            code_column = self._first_existing_column(
                raw, ["代码", "股票代码", "证券代码"]
            )
            if code_column is None:
                continue
            frame = pd.DataFrame(
                {
                    "date": pd.Timestamp(date).normalize(),
                    "symbol": raw[code_column].astype(str).map(self._standard_symbol),
                    "official_suspension_flag": True,
                }
            )
            frame = frame.loc[frame["symbol"].isin(symbol_set)].drop_duplicates(
                ["date", "symbol"]
            )
            if not frame.empty:
                frames.append(frame)

        result = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["date", "symbol", "official_suspension_flag"])
        )
        self._suspension_cache[cache_key] = result
        return result.copy()

    @staticmethod
    def _combine_suspension_flags(data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        frame["is_suspended_inferred_missing"] = (
            frame.get("is_suspended", pd.Series(False, index=frame.index))
            .fillna(False)
            .astype(bool)
        )
        frame["is_zero_volume"] = (
            frame.get("volume", pd.Series(0, index=frame.index)).fillna(0).eq(0)
        )
        frame["official_suspension_flag"] = (
            frame.get(
                "official_suspension_flag", pd.Series(False, index=frame.index)
            )
            .fillna(False)
            .astype(bool)
        )
        frame["is_suspended"] = (
            frame["is_suspended_inferred_missing"]
            | frame["is_zero_volume"]
            | frame["official_suspension_flag"]
        )
        return frame

    def _macro_pmi_history_legacy_unused(self, ak: Any) -> pd.DataFrame:
        pmi = self._cached("macro_china_pmi", lambda: ak.macro_china_pmi())
        pmi = pmi[["鏈堜唤", "鍒堕€犱笟-鎸囨暟"]].rename(
            columns={"鏈堜唤": "month", "鍒堕€犱笟-鎸囨暟": "macro_pmi"}
        )
        pmi["date"] = pd.to_datetime(
            pmi["month"].astype(str).str.extract(r"(\d{4})")[0]
            + "-"
            + pmi["month"].astype(str).str.extract(r"骞?(\d{2})")[0]
            + "-01",
            errors="coerce",
        ) + pd.offsets.MonthEnd(0)
        pmi["macro_pmi"] = pd.to_numeric(pmi["macro_pmi"], errors="coerce")
        return pmi[["date", "macro_pmi"]]

    @staticmethod
    def _indicator_row(raw: pd.DataFrame, indicator: str) -> pd.Series:
        matches = raw.loc[raw["指标"].astype(str) == indicator]
        if matches.empty:
            return pd.Series(dtype=float)
        return pd.to_numeric(matches.iloc[0, 2:], errors="coerce")

    def _financials(self, ak: Any, symbol: str) -> pd.DataFrame:
        code = self._raw_code(symbol)
        raw = self._cached(
            f"financial_abstract_{code}",
            lambda: ak.stock_financial_abstract(symbol=code),
        )
        if raw.empty or "指标" not in raw:
            return pd.DataFrame()

        date_columns = [
            column for column in raw.columns[2:] if str(column).isdigit() and len(str(column)) == 8
        ]
        frame = pd.DataFrame({"report_date": pd.to_datetime(date_columns, format="%Y%m%d")})
        mappings = {
            "net_profit": "归母净利润",
            "revenue": "营业总收入",
            "operating_cost": "营业成本",
            "book_value": "股东权益合计(净资产)",
            "operating_cashflow": "经营现金流量净额",
            "roa_reported": "总资产报酬率(ROA)",
            "gross_margin_reported": "毛利率",
            "leverage_reported": "资产负债率",
            "current_ratio_reported": "流动比率",
        }
        for target, indicator in mappings.items():
            values = self._indicator_row(raw, indicator)
            frame[target] = [values.get(column, np.nan) for column in date_columns]

        for column in ("roa_reported", "gross_margin_reported", "leverage_reported"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100
        frame["current_ratio_reported"] = pd.to_numeric(
            frame["current_ratio_reported"], errors="coerce"
        )
        frame["assets"] = frame["book_value"] / (1 - frame["leverage_reported"]).replace(
            0, np.nan
        )
        frame["liabilities"] = frame["assets"] - frame["book_value"]
        frame["gross_profit"] = frame["revenue"] - frame["operating_cost"]
        current_liability_share = 0.5
        frame["current_liabilities"] = frame["liabilities"] * current_liability_share
        frame["current_assets"] = (
            frame["current_ratio_reported"] * frame["current_liabilities"]
        )
        frame["available_date"] = frame["report_date"] + pd.to_timedelta(
            self.financial_lag_days, unit="D"
        )
        if self.use_actual_notice_date:
            market_symbol = f"{self._suffix(code)}{code}"
            try:
                statements = self._cached(
                    f"profit_statement_dates_{market_symbol}",
                    lambda: ak.stock_profit_sheet_by_report_em(symbol=market_symbol),
                )
                report_column = next(
                    (
                        column
                        for column in ("REPORT_DATE", "报告日", "报告期")
                        if column in statements.columns
                    ),
                    None,
                )
                notice_column = next(
                    (
                        column
                        for column in ("NOTICE_DATE", "公告日期", "最新公告日期")
                        if column in statements.columns
                    ),
                    None,
                )
                if report_column and notice_column:
                    notices = statements[[report_column, notice_column]].copy()
                    notices[report_column] = pd.to_datetime(
                        notices[report_column], errors="coerce"
                    ).dt.normalize()
                    notices[notice_column] = pd.to_datetime(
                        notices[notice_column], errors="coerce"
                    ).dt.normalize()
                    notices = notices.dropna().drop_duplicates(
                        report_column, keep="last"
                    )
                    notice_lookup = notices.set_index(report_column)[notice_column]
                    actual_notice = frame["report_date"].dt.normalize().map(notice_lookup)
                    frame["available_date"] = actual_notice.fillna(
                        frame["available_date"]
                    )
            except ProviderError:
                # 部分下游站点可能暂时不可用，保守滞后仍可防止报告期前视。
                pass
        return frame.sort_values("available_date")

    def _industry_history(
        self, ak: Any, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        code = self._raw_code(symbol)
        extended_start = (
            pd.Timestamp(start_date) - pd.DateOffset(years=10)
        ).strftime("%Y%m%d")
        raw = self._cached(
            f"industry_{code}_{extended_start}_{end_date}",
            lambda: ak.stock_industry_change_cninfo(
                symbol=code,
                start_date=extended_start,
                end_date=end_date.replace("-", ""),
            ),
        )
        if raw.empty or "变更日期" not in raw:
            return pd.DataFrame()
        selected = raw.copy()
        industry_columns = [
            column
            for column in ("行业门类", "行业大类", "行业中类")
            if column in selected.columns
        ]
        if not industry_columns:
            return pd.DataFrame()
        selected["industry"] = selected[industry_columns].bfill(axis=1).iloc[:, 0]
        result = selected[["变更日期", "industry"]].rename(
            columns={"变更日期": "available_date"}
        )
        result["available_date"] = pd.to_datetime(result["available_date"])
        return result.dropna().sort_values("available_date")

    def _dividend_history(self, ak: Any, symbol: str) -> pd.DataFrame:
        code = self._raw_code(symbol)
        raw = self._cached(
            f"dividend_{code}",
            lambda: ak.stock_history_dividend_detail(symbol=code, indicator="分红"),
        )
        if raw.empty or "除权除息日" not in raw or "派息" not in raw:
            return pd.DataFrame()
        frame = raw[["除权除息日", "派息"]].rename(
            columns={"除权除息日": "date", "派息": "cash_per_10_shares"}
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["cash_per_10_shares"] = pd.to_numeric(
            frame["cash_per_10_shares"], errors="coerce"
        )
        return frame.dropna().sort_values("date")

    def _macro_data_legacy_unused(self, ak: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Legacy compatibility wrapper; the active fetch path no longer uses market-level
        # northbound allocation and instead calls _northbound_history for stock-level data.
        pmi = self._cached("macro_china_pmi", lambda: ak.macro_china_pmi())
        pmi = pmi[["月份", "制造业-指数"]].rename(
            columns={"月份": "month", "制造业-指数": "macro_pmi"}
        )
        pmi["date"] = pd.to_datetime(
            pmi["month"].astype(str).str.extract(r"(\d{4})")[0]
            + "-"
            + pmi["month"].astype(str).str.extract(r"年(\d{2})")[0]
            + "-01",
            errors="coerce",
        ) + pd.offsets.MonthEnd(0)
        pmi["macro_pmi"] = pd.to_numeric(pmi["macro_pmi"], errors="coerce")

        north = self._cached(
            "northbound_history", lambda: ak.stock_hsgt_hist_em(symbol="北向资金")
        )
        north = north[["日期", "当日成交净买额"]].rename(
            columns={"日期": "date", "当日成交净买额": "northbound_market"}
        )
        north["date"] = pd.to_datetime(north["date"])
        north["northbound_market"] = (
            pd.to_numeric(north["northbound_market"], errors="coerce") * 1e8
        )
        return pmi[["date", "macro_pmi"]], north

    def _macro_data(self, ak: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Legacy compatibility wrapper."""

        return self._macro_pmi_history(ak), pd.DataFrame(
            columns=["date", "symbol", "northbound_flow"]
        )

    def _macro_pmi_history(self, ak: Any) -> pd.DataFrame:
        """Load PMI history with robust column parsing and no mojibake-dependent labels."""

        pmi = self._cached("macro_china_pmi", lambda: ak.macro_china_pmi())
        month_column = self._first_existing_column(
            pmi, ["月份", "month", "Month"]
        )
        value_column = self._first_existing_column(
            pmi, ["制造业-指数", "制造业PMI", "PMI", "pmi"]
        )
        if month_column is None or value_column is None:
            raise ProviderError(
                f"macro_china_pmi 缺少关键列: {pmi.columns.tolist()}"
            )
        pmi = pmi[[month_column, value_column]].rename(
            columns={month_column: "month", value_column: "macro_pmi"}
        )
        month_parts = pmi["month"].astype(str).str.extract(r"(\d{4})\D+(\d{1,2})")
        pmi["date"] = pd.to_datetime(
            month_parts[0]
            + "-"
            + month_parts[1].str.zfill(2)
            + "-01",
            errors="coerce",
        ) + pd.offsets.MonthEnd(0)
        pmi["macro_pmi"] = pd.to_numeric(pmi["macro_pmi"], errors="coerce")
        return pmi[["date", "macro_pmi"]]

    @staticmethod
    def _asof_merge(
        daily: pd.DataFrame,
        history: pd.DataFrame,
        history_date: str = "available_date",
    ) -> pd.DataFrame:
        if history.empty:
            return daily
        daily = daily.copy()
        history = history.copy()
        daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
        history[history_date] = pd.to_datetime(history[history_date]).astype(
            "datetime64[ns]"
        )
        return pd.merge_asof(
            daily.sort_values("date"),
            history.sort_values(history_date),
            left_on="date",
            right_on=history_date,
            direction="backward",
        )

    def _complete_calendar(self, data: pd.DataFrame) -> pd.DataFrame:
        trading_dates = pd.DatetimeIndex(sorted(data["date"].unique()))
        completed: list[pd.DataFrame] = []
        for symbol, group in data.groupby("symbol"):
            first_observation = pd.Timestamp(group["date"].min())
            last_observation = pd.Timestamp(group["date"].max())
            eligible_from = pd.to_datetime(
                group.get("universe_eligible_from", pd.Series([first_observation])),
                errors="coerce",
            ).dropna()
            eligible_to = pd.to_datetime(
                group.get("universe_eligible_to", pd.Series([last_observation])),
                errors="coerce",
            ).dropna()
            calendar_start = eligible_from.min() if not eligible_from.empty else first_observation
            calendar_end = eligible_to.max() if not eligible_to.empty else last_observation
            valid_dates = trading_dates[
                (trading_dates >= calendar_start)
                & (trading_dates <= calendar_end)
            ]
            group = group.set_index("date").reindex(valid_dates)
            group.index.name = "date"
            group["symbol"] = symbol
            observed = group["close"].notna()
            group["is_suspended"] = ~observed
            for column in ("close", "outstanding_share"):
                group[column] = group[column].ffill()
            for column in ("open", "high", "low"):
                group[column] = group[column].fillna(group["close"])
            for column in ("volume", "amount", "turnover_rate"):
                group[column] = group[column].fillna(0)
            for column in (
                "industry",
                "price_source",
                "price_adjustment",
                "adjustment_applied",
                "name",
                "universe_eligible_from",
                "universe_eligible_to",
                "in_universe",
            ):
                if column in group:
                    group[column] = group[column].ffill()
            financial_columns = [
                "net_profit",
                "revenue",
                "book_value",
                "assets",
                "liabilities",
                "operating_cashflow",
                "gross_profit",
                "current_assets",
                "current_liabilities",
            ]
            for column in financial_columns:
                if column in group:
                    group[column] = group[column].ffill()
            completed.append(group.reset_index())
        return pd.concat(completed, ignore_index=True)

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取并整合 AkShare 真实数据。"""

        try:
            import akshare as ak
        except ImportError as exc:
            raise ProviderError("使用 AkShareProvider 前请安装 akshare") from exc

        symbols, names, membership_periods = self._universe(start_date)
        if not symbols:
            raise ProviderError("AkShare 股票池为空，请配置 data.symbols 或 universe_index")

        frames: list[pd.DataFrame] = []
        failures: list[str] = []
        for symbol in symbols:
            try:
                daily = self._fetch_price(ak, symbol, start_date, end_date)
                try:
                    financials = self._financials_real(
                        ak, symbol, start_date, end_date
                    )
                except ProviderError as exc:
                    failures.append(f"{symbol} 财务数据: {exc}")
                    financials = pd.DataFrame()
                try:
                    industry = self._industry_history(ak, symbol, start_date, end_date)
                except ProviderError as exc:
                    failures.append(f"{symbol} 行业数据: {exc}")
                    industry = pd.DataFrame()
                try:
                    dividends = self._dividend_history(ak, symbol)
                except ProviderError as exc:
                    failures.append(f"{symbol} 分红数据: {exc}")
                    dividends = pd.DataFrame()
                try:
                    sentiment = self._sentiment_history(ak, symbol)
                except ProviderError as exc:
                    failures.append(f"{symbol} emotion data: {exc}")
                    sentiment = pd.DataFrame()
                daily = self._asof_merge(daily, financials)
                daily = self._asof_merge(daily, industry)
                if not sentiment.empty:
                    daily = daily.merge(sentiment, on="date", how="left")

                dividend_events = (
                    dividends.groupby("date")["cash_per_10_shares"].sum()
                    if not dividends.empty
                    else pd.Series(dtype=float)
                )
                event_series = dividend_events.reindex(
                    pd.DatetimeIndex(daily["date"]), fill_value=0
                )
                daily["dividend_per_share_ttm"] = (
                    event_series.rolling("365D").sum().to_numpy() / 10
                )
                daily["dividend"] = (
                    daily["dividend_per_share_ttm"]
                    * daily["outstanding_share"].ffill()
                )
                daily["name"] = names.get(symbol, "")
                periods = membership_periods.get(
                    symbol, [(pd.Timestamp(start_date), None)]
                )
                daily["universe_eligible_from"] = min(
                    period[0] for period in periods
                )
                daily["universe_eligible_to"] = max(
                    (period[1] if period[1] is not None else pd.Timestamp(end_date))
                    for period in periods
                )
                daily["in_universe"] = False
                for effective_from, effective_to in periods:
                    active = daily["date"] >= effective_from
                    if effective_to is not None:
                        active &= daily["date"] <= effective_to
                    daily["in_universe"] |= active
                frames.append(daily)
            except ProviderError as exc:
                failures.append(f"{symbol}: {exc}")

        if not frames:
            raise ProviderError("全部证券获取失败: " + " | ".join(failures))

        data = pd.concat(frames, ignore_index=True)
        data = self._complete_calendar(data)
        suspension_flags = self._official_suspension_flags(
            ak,
            sorted(pd.to_datetime(data["date"]).unique()),
            sorted(data["symbol"].dropna().astype(str).unique()),
        )
        pmi = self._macro_pmi_history(ak)
        northbound = self._northbound_history(
            ak,
            sorted(data["symbol"].dropna().astype(str).unique()),
            start_date,
            end_date,
        )
        data["date"] = pd.to_datetime(data["date"]).astype("datetime64[ns]")
        if not suspension_flags.empty:
            suspension_flags["date"] = pd.to_datetime(
                suspension_flags["date"]
            ).astype("datetime64[ns]")
            data = data.merge(suspension_flags, on=["date", "symbol"], how="left")
        else:
            data["official_suspension_flag"] = False
        data = self._combine_suspension_flags(data)
        pmi["date"] = pd.to_datetime(pmi["date"]).astype("datetime64[ns]")
        data = pd.merge_asof(
            data.sort_values("date"),
            pmi.sort_values("date"),
            on="date",
            direction="backward",
        )
        if not northbound.empty:
            northbound["date"] = pd.to_datetime(northbound["date"]).astype(
                "datetime64[ns]"
            )
            data = data.merge(northbound, on=["date", "symbol"], how="left")
        else:
            data["northbound_flow"] = np.nan
        data["market_cap"] = data["close"] * data["outstanding_share"]
        if "sentiment" not in data:
            data["sentiment"] = np.nan
        data["risk_free_rate"] = self.risk_free_rate
        data["is_st"] = data["name"].astype(str).str.upper().str.contains("ST")
        data["in_universe"] = data["in_universe"].fillna(False)
        returns = data.groupby("symbol")["close"].pct_change().fillna(0)
        codes = data["symbol"].str[:6]
        growth_board = codes.str.startswith(("300", "301", "688"))
        threshold = np.where(growth_board, 0.195, 0.095)
        threshold = np.where(data["is_st"], 0.047, threshold)
        data["limit_up"] = returns >= threshold
        data["limit_down"] = returns <= -threshold
        data["industry"] = data["industry"].fillna("未知")
        data["data_source"] = "AkShare"

        required_financials = [
            "net_profit",
            "revenue",
            "book_value",
            "assets",
            "liabilities",
            "operating_cashflow",
            "gross_profit",
            "current_assets",
            "current_liabilities",
        ]
        for column in required_financials:
            if column not in data:
                data[column] = np.nan

        if failures:
            failure_path = self.cache_path / "fetch_failures.txt"
            failure_path.write_text("\n".join(failures), encoding="utf-8")
        else:
            (self.cache_path / "fetch_failures.txt").unlink(missing_ok=True)

        return data.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_provider(config: dict[str, Any], random_seed: int) -> MarketDataProvider:
    """根据配置创建数据源。"""

    provider = config["provider"].lower()
    if provider == "synthetic":
        return SyntheticProvider(
            symbol_count=int(config.get("synthetic_symbol_count", 80)),
            random_seed=random_seed,
        )
    if provider == "akshare":
        return AkShareProvider(
            symbols=list(config.get("symbols", [])),
            universe_index=str(config.get("universe_index", "000300")),
            max_symbols=int(config.get("max_symbols", 300)),
            adjust=str(config.get("adjust", "qfq")),
            cache_dir=str(config.get("cache_dir", "data/akshare_cache")),
            refresh_cache=bool(config.get("refresh_cache", False)),
            request_interval=float(config.get("request_interval", 0.25)),
            retries=int(config.get("retries", 3)),
            financial_lag_days=int(config.get("financial_lag_days", 120)),
            risk_free_rate=float(config.get("risk_free_rate", 0.02)),
            require_membership_before_start=bool(
                config.get("require_membership_before_start", True)
            ),
            use_actual_notice_date=bool(config.get("use_actual_notice_date", True)),
            membership_file=str(config.get("membership_file", "")),
        )
    raise ProviderError(f"不支持的数据源: {provider}")
