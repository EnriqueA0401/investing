from __future__ import annotations

import pandas as pd
import pytest

from quantlab.components.costs import SpreadSlippageImpact
from quantlab.config import CostConfig
from quantlab.contracts import CostContext


def _ctx(cfg: CostConfig, adv: float = 1e8) -> CostContext:
    return CostContext(
        trade_date=pd.Timestamp("2024-01-02"),
        nav_usd=1e6,
        adv_usd=pd.Series({"A": adv, "B": adv}),
        daily_vol=pd.Series({"A": 0.01, "B": 0.01}),
        asset_class={"A": "equity", "B": "commodity"},
        config=cfg,
    )


def test_linear_costs_by_asset_class() -> None:
    cfg = CostConfig(
        half_spread_bps={"default": 5, "equity": 1},
        commission_bps=0,
        slippage_bps=0,
        impact_coefficient=0,
    )
    c = SpreadSlippageImpact.from_params({}).cost(pd.Series({"A": 0.1, "B": -0.1}), _ctx(cfg))
    assert c["A"] == pytest.approx(0.1 * 1e-4)
    assert c["B"] == pytest.approx(0.1 * 5e-4)


def test_impact_grows_with_size_and_lower_liquidity() -> None:
    cfg = CostConfig(
        half_spread_bps={"default": 0}, commission_bps=0, slippage_bps=0, impact_coefficient=0.1
    )
    model = SpreadSlippageImpact.from_params({})
    small = model.cost(pd.Series({"A": 0.01}), _ctx(cfg)).sum()
    big = model.cost(pd.Series({"A": 0.10}), _ctx(cfg)).sum()
    illiquid = model.cost(pd.Series({"A": 0.10}), _ctx(cfg, adv=1e6)).sum()
    assert big / small == pytest.approx(10**1.5)  # |dw|^(3/2)
    assert illiquid > big


def test_missing_liquidity_estimate_raises() -> None:
    cfg = CostConfig(impact_coefficient=0.1)
    ctx = _ctx(cfg)
    ctx = CostContext(**{**ctx.__dict__, "adv_usd": pd.Series({"A": float("nan")})})
    with pytest.raises(ValueError, match="no ADV"):
        SpreadSlippageImpact.from_params({}).cost(pd.Series({"A": 0.1}), ctx)
