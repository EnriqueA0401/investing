"""Point-in-time market data container.

All panels are wide DataFrames indexed by trading date (tz-naive, ascending,
unique) with one column per ticker. A row dated ``d`` contains only
information that was observable at the close of ``d``.

``total_return_index`` is used for all return calculations. Vendor
"adjusted close" series are back-adjusted: their *levels* at past dates change
when later dividends/splits occur, so only *ratios* within the series are
point-in-time safe. Features must therefore be built from returns or ratios,
never from price levels of an adjusted series.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PANEL_FIELDS = ("close", "total_return_index", "volume")


@dataclass(frozen=True)
class MarketData:
    close: pd.DataFrame
    total_return_index: pd.DataFrame
    volume: pd.DataFrame
    asset_class: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ref = self.total_return_index
        if not isinstance(ref.index, pd.DatetimeIndex):
            raise TypeError("panels must be indexed by a DatetimeIndex")
        if ref.index.tz is not None:
            raise ValueError("panels must be tz-naive (dates are exchange-local trading dates)")
        if not ref.index.is_monotonic_increasing or ref.index.has_duplicates:
            raise ValueError("index must be strictly increasing")
        for name in PANEL_FIELDS:
            panel = getattr(self, name)
            if not panel.index.equals(ref.index) or list(panel.columns) != list(ref.columns):
                raise ValueError(f"panel '{name}' is not aligned with total_return_index")

    @property
    def dates(self) -> pd.DatetimeIndex:
        idx = self.total_return_index.index
        assert isinstance(idx, pd.DatetimeIndex)
        return idx

    @property
    def tickers(self) -> list[str]:
        return [str(c) for c in self.total_return_index.columns]

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.dates[-1])

    def _map(self, rows: slice | np.ndarray, cols: list[str] | None = None) -> MarketData:
        def sub(df: pd.DataFrame) -> pd.DataFrame:
            out = df.loc[rows]
            return out if cols is None else out.loc[:, cols]

        return MarketData(
            close=sub(self.close),
            total_return_index=sub(self.total_return_index),
            volume=sub(self.volume),
            asset_class=dict(self.asset_class)
            if cols is None
            else {t: self.asset_class[t] for t in cols if t in self.asset_class},
        )

    def asof(self, t: pd.Timestamp | str) -> MarketData:
        """Everything observable at the close of ``t`` (rows dated <= t)."""
        mask = self.dates <= pd.Timestamp(t)
        if not mask.any():
            raise ValueError(f"no data on or before {t}")
        return self._map(mask)

    def between(
        self, start: pd.Timestamp | str | None, end: pd.Timestamp | str | None
    ) -> MarketData:
        mask = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            mask &= self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        return self._map(mask)

    def select(self, tickers: list[str]) -> MarketData:
        return self._map(slice(None), list(tickers))

    def returns(self) -> pd.DataFrame:
        """Simple daily total returns; NaN before an asset's first observation."""
        return self.total_return_index.pct_change(fill_method=None)

    def history_length(self) -> pd.Series:
        """Number of valid observations per ticker up to the last row."""
        return self.total_return_index.notna().sum()

    def investable(self, min_history_days: int) -> list[str]:
        """Tickers with enough history *and* a valid price on the last row.

        Uses only rows already in this (possibly truncated) view, so it is
        causal by construction when called on ``data.asof(t)``.
        """
        last = self.total_return_index.iloc[-1]
        ok = (self.history_length() >= min_history_days) & last.notna()
        return [str(t) for t in ok[ok].index]
