"""Point-in-time market data container.

All panels are wide DataFrames indexed by trading date (tz-naive, ascending,
unique) with one column per ticker. A row dated ``d`` contains only
information that was observable at the close of ``d``.

``total_return_index`` is used for all return calculations. Vendor
"adjusted close" series are back-adjusted: their *levels* at past dates change
when later dividends/splits occur, so only *ratios* within the series are
point-in-time safe. Features must therefore be built from returns or ratios,
never from price levels of an adjusted series.

``cash_return`` (optional) is the simple return earned by uninvested cash from
the previous trading date to ``d``; it is the risk-free rate for excess-return
statistics.

Auxiliary point-in-time frames have their own DatetimeIndex (e.g. month-ends);
a row dated ``d`` holds only what was public at the close of ``d``:
``fundamentals`` (name -> dates x tickers), ``universe`` (dates x tickers bool
membership) and ``extras`` (name -> daily series such as macro prints).
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
    cash_return: pd.Series | None = None
    fundamentals: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    universe: pd.DataFrame | None = None
    extras: Mapping[str, pd.Series] = field(default_factory=dict)

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
        if self.cash_return is not None and not self.cash_return.index.equals(ref.index):
            raise ValueError("cash_return is not aligned with total_return_index")
        aux: list[pd.DataFrame | pd.Series] = [*self.fundamentals.values(), *self.extras.values()]
        if self.universe is not None:
            aux.append(self.universe)
        for frame in aux:
            if (
                not isinstance(frame.index, pd.DatetimeIndex)
                or not frame.index.is_monotonic_increasing
            ):
                raise ValueError("auxiliary frames need an increasing DatetimeIndex")

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

    def _map(
        self,
        rows: slice | np.ndarray,
        cols: list[str] | None = None,
        hi: pd.Timestamp | None = None,
    ) -> MarketData:
        def sub(df: pd.DataFrame) -> pd.DataFrame:
            out = df.loc[rows]
            return out if cols is None else out.loc[:, cols]

        def aux_df(df: pd.DataFrame) -> pd.DataFrame:
            out = df if hi is None else df.loc[df.index <= hi]
            return out if cols is None else out.loc[:, [c for c in cols if c in out.columns]]

        def aux_s(s: pd.Series) -> pd.Series:
            return s if hi is None else s.loc[s.index <= hi]

        return MarketData(
            close=sub(self.close),
            total_return_index=sub(self.total_return_index),
            volume=sub(self.volume),
            asset_class=dict(self.asset_class)
            if cols is None
            else {t: self.asset_class[t] for t in cols if t in self.asset_class},
            cash_return=None if self.cash_return is None else self.cash_return.loc[rows],
            fundamentals={k: aux_df(v) for k, v in self.fundamentals.items()},
            universe=None if self.universe is None else aux_df(self.universe),
            extras={k: aux_s(v) for k, v in self.extras.items()},
        )

    def asof(self, t: pd.Timestamp | str) -> MarketData:
        """Everything observable at the close of ``t`` (rows dated <= t)."""
        mask = self.dates <= pd.Timestamp(t)
        if not mask.any():
            raise ValueError(f"no data on or before {t}")
        return self._map(mask, hi=pd.Timestamp(t))

    def between(
        self, start: pd.Timestamp | str | None, end: pd.Timestamp | str | None
    ) -> MarketData:
        mask = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            mask &= self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        return self._map(mask, hi=None if end is None else pd.Timestamp(end))

    def select(self, tickers: list[str]) -> MarketData:
        return self._map(slice(None), list(tickers))

    def returns(self) -> pd.DataFrame:
        """Simple daily total returns; NaN before an asset's first observation."""
        return self.total_return_index.pct_change(fill_method=None)

    def cash(self) -> pd.Series:
        """Daily cash return; zero if no cash series was loaded."""
        if self.cash_return is None:
            return pd.Series(0.0, index=self.dates, name="cash")
        return self.cash_return

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
