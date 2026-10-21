"""Transaction-cost models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.contracts import CostContext, CostModel
from quantlab.registry import register


@register("cost_model", "spread_slippage_impact")
class SpreadSlippageImpact(CostModel):
    """Linear (half-spread + commission + slippage) plus square-root market impact.

    cost_i / NAV = |dw_i| * (half_spread_i + commission + slippage) / 1e4
                 + |dw_i| * k * sigma_i * sqrt(|dw_i| * NAV / ADV_i)

    sigma_i is daily return volatility and ADV_i average daily dollar volume,
    both measured strictly before the trade date.
    """

    def cost(self, trades: pd.Series, ctx: CostContext) -> pd.Series:
        cfg = ctx.config
        size = trades.abs()
        spread = pd.Series(
            [
                cfg.half_spread_bps.get(
                    ctx.asset_class.get(str(t), ""), cfg.half_spread_bps["default"]
                )
                for t in size.index
            ],
            index=size.index,
        )
        linear = size * (spread + cfg.commission_bps + cfg.slippage_bps) / 1e4
        if cfg.impact_coefficient == 0:
            return linear
        traded = size[size > 0].index
        adv = ctx.adv_usd.reindex(traded)
        vol = ctx.daily_vol.reindex(traded)
        missing = adv[(adv.isna()) | (adv <= 0) | vol.isna()].index
        if len(missing):
            raise ValueError(
                f"{ctx.trade_date.date()}: no ADV/vol estimate for traded assets {list(missing)}"
            )
        participation = size[traded] * ctx.nav_usd / adv
        impact = size[traded] * cfg.impact_coefficient * vol * np.sqrt(participation)
        return linear.add(impact, fill_value=0.0)
