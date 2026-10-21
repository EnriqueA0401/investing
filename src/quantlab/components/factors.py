"""Factors. Each returns a dates x tickers panel where row ``d`` uses data <= ``d`` only.

All factors are built from *ratios* of the total-return index (never levels),
so they are invariant to how the index is normalised. Every factor has an
``assert_causal`` test in ``tests/test_factors.py``.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
from pydantic import Field

from quantlab.contracts import ComponentParams, Factor
from quantlab.data.market_data import MarketData
from quantlab.registry import register

TRADING_DAYS_PER_MONTH = 21


@register("factor", "momentum")
class Momentum(Factor):
    """Cross-sectional momentum: total return from ``t - lookback`` to ``t - skip``.

    Thesis (H-002): slow diffusion of information across markets and slow-moving
    capital make relative winners keep outperforming over the next month.
    Expected sign: positive. Skipping the last month avoids short-term reversal
    (liquidity/microstructure). Known risk: momentum crashes in sharp rebounds.
    """

    hypothesis_id: ClassVar[str] = "H-002"

    class Params(ComponentParams):
        label: str = "mom_12_1"
        lookback_days: int = Field(default=252, gt=0)
        skip_days: int = Field(default=21, ge=0)

    params: Params

    def compute(self, history: MarketData) -> pd.DataFrame:
        p = self.params
        if p.skip_days >= p.lookback_days:
            raise ValueError("skip_days must be shorter than lookback_days")
        tri = history.total_return_index
        return tri.shift(p.skip_days) / tri.shift(p.lookback_days) - 1.0


@register("factor", "excess_momentum")
class ExcessMomentum(Factor):
    """Time-series momentum: trailing total return in excess of cash over the same window.

    Thesis (H-001): under-reaction followed by delayed over-reaction and
    pro-cyclical hedging flows make an asset's own trend persist. Expected sign:
    positive (positive trailing excess return -> positive expected excess
    return). Pays in prolonged crises, loses in V-shaped reversals (whipsaw).
    """

    hypothesis_id: ClassVar[str] = "H-001"

    class Params(ComponentParams):
        label: str = "tsmom_12"
        lookback_days: int = Field(default=252, gt=0)

    params: Params

    def compute(self, history: MarketData) -> pd.DataFrame:
        n = self.params.lookback_days
        tri = history.total_return_index
        asset = tri / tri.shift(n) - 1.0
        cash_index = (1.0 + history.cash()).cumprod()
        cash = cash_index / cash_index.shift(n) - 1.0
        return asset.sub(cash, axis=0)


@register("factor", "realized_vol")
class RealizedVol(Factor):
    """Annualised realised volatility of daily total returns over a trailing window.

    Thesis (H-003): leverage-constrained investors overpay for high-risk
    assets, so low-volatility assets earn higher risk-adjusted returns.
    Expected sign vs. *risk-adjusted* forward return: negative. Used both as a
    risk input (inverse-vol weights) and as a feature (H-006).
    """

    hypothesis_id: ClassVar[str] = "H-003"

    class Params(ComponentParams):
        label: str = "vol_63"
        window_days: int = Field(default=63, ge=2)

    params: Params

    def compute(self, history: MarketData) -> pd.DataFrame:
        w = self.params.window_days
        out: pd.DataFrame = history.returns().rolling(w, min_periods=w).std() * (252**0.5)
        return out
