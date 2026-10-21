"""Risk overlays (``pipeline.risk`` slot): applied in order after portfolio construction.

Overlays only ever *reduce* or *re-time* exposure; whatever they remove goes to
cash. They never lever up (long-only, unlevered mandate).
"""

from __future__ import annotations

from typing import cast

import pandas as pd
from pydantic import Field

from quantlab.contracts import ComponentParams, DecisionContext, RiskManager
from quantlab.registry import register
from quantlab.risk_model import ledoit_wolf_cov, portfolio_vol, trailing_returns


@register("risk", "trend_filter")
class TrendFilter(RiskManager):
    """H-001: zero the slot of any asset whose trend feature is <= threshold (slot -> cash)."""

    class Params(ComponentParams):
        feature: str = "tsmom_12"
        threshold: float = 0.0

    params: Params

    def adjust(self, weights: pd.Series, ctx: DecisionContext) -> pd.Series:
        row = cast(pd.Series, ctx.features[self.params.feature].loc[ctx.as_of])
        trend = row.reindex(weights.index)
        keep = trend > self.params.threshold  # NaN trend -> not kept
        return weights.where(keep, 0.0)


@register("risk", "vol_target")
class VolTarget(RiskManager):
    """H-005: scale exposure by min(1, target / forecast vol); remainder in cash."""

    class Params(ComponentParams):
        target_vol: float = Field(default=0.10, gt=0)
        window_days: int = Field(default=63, ge=20)

    params: Params

    def adjust(self, weights: pd.Series, ctx: DecisionContext) -> pd.Series:
        held = [str(t) for t in weights[weights > 0].index]
        if not held:
            return weights
        cov = ledoit_wolf_cov(trailing_returns(ctx.history, held, self.params.window_days, 20))
        vol = portfolio_vol(weights[held], cov)
        scale = min(1.0, self.params.target_vol / vol) if vol > 0 else 1.0
        return weights * scale


@register("risk", "turnover_cap")
class TurnoverCap(RiskManager):
    """Move only part of the way from current (drifted) holdings to target so that
    one-way turnover <= ``max_turnover``; positions are then clipped to the cap."""

    class Params(ComponentParams):
        max_turnover: float = Field(gt=0, le=1)

    params: Params

    def adjust(self, weights: pd.Series, ctx: DecisionContext) -> pd.Series:
        names = weights.index.union(ctx.current_weights.index)
        tgt = weights.reindex(names).fillna(0.0)
        cur = ctx.current_weights.reindex(names).fillna(0.0)
        turnover = float((tgt - cur).abs().sum() / 2)
        if turnover <= self.params.max_turnover:
            return weights
        lam = self.params.max_turnover / turnover
        out = (cur + lam * (tgt - cur)).clip(lower=0.0, upper=ctx.constraints.max_weight)
        out = out[out.index.isin(ctx.investable)]
        return out[out > 0]
