"""Data-source adapters. Real vendors (yfinance, FRED) arrive in Phase 2."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
from pydantic import Field

from quantlab.config import Asset
from quantlab.contracts import ComponentParams, DataSource
from quantlab.data.market_data import MarketData
from quantlab.registry import register


@register("data_source", "synthetic")
class SyntheticSource(DataSource):
    """Seeded one-factor GBM panel with staggered inception dates.

    For tests and plumbing only: it has no economic content, so any "edge"
    found on it is by construction spurious (useful as a false-positive check).
    """

    class Params(ComponentParams):
        seed: int = 7
        end: date = date(2025, 12, 31)
        annual_drift: float = 0.05
        annual_vol: float = 0.15
        market_beta: float = 0.6
        inception: dict[str, date] = Field(default_factory=dict)
        base_volume: float = 5e6

    params: Params

    def fetch(self, assets: Sequence[Asset], start: date, end: date | None) -> MarketData:
        p = self.params
        dates = pd.bdate_range(start, end or p.end, name="date")
        tickers = [a.ticker for a in assets]
        rng = np.random.default_rng(p.seed)
        n, k = len(dates), len(tickers)
        sigma_d = p.annual_vol / np.sqrt(252)
        market = rng.standard_normal(n)
        idio = rng.standard_normal((n, k))
        beta = p.market_beta
        shocks = beta * market[:, None] + np.sqrt(1 - beta**2) * idio
        log_ret = (p.annual_drift / 252 - 0.5 * sigma_d**2) + sigma_d * shocks
        tri = pd.DataFrame(100 * np.exp(np.cumsum(log_ret, axis=0)), index=dates, columns=tickers)
        volume = pd.DataFrame(
            p.base_volume * rng.lognormal(0.0, 0.3, (n, k)), index=dates, columns=tickers
        )
        for ticker, first in p.inception.items():
            if ticker in tri.columns:
                tri.loc[tri.index < pd.Timestamp(first), ticker] = np.nan
                volume.loc[volume.index < pd.Timestamp(first), ticker] = np.nan
        return MarketData(
            close=tri.copy(),
            total_return_index=tri,
            volume=volume,
            asset_class={a.ticker: a.asset_class for a in assets},
        )
