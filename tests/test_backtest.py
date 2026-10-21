from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.components.backtest import DailyDrift
from quantlab.components.costs import SpreadSlippageImpact
from quantlab.config import ExperimentConfig
from quantlab.data.market_data import MarketData


def _data(returns: list[list[float]]) -> MarketData:
    r = np.array(returns)
    dates = pd.bdate_range("2024-01-01", periods=len(r) + 1)
    tri = pd.DataFrame(
        100 * np.vstack([np.ones(r.shape[1]), np.cumprod(1 + r, axis=0)]),
        index=dates,
        columns=["A", "B"],
    )
    vol = pd.DataFrame(1e9, index=dates, columns=["A", "B"])
    return MarketData(
        close=tri, total_return_index=tri, volume=vol, asset_class={"A": "equity", "B": "equity"}
    )


def _cfg(cfg: ExperimentConfig, spread_bps: float = 0.0) -> ExperimentConfig:
    costs = cfg.costs.model_copy(
        update={
            "half_spread_bps": {"default": spread_bps},
            "commission_bps": 0.0,
            "slippage_bps": 0.0,
            "impact_coefficient": 0.0,
        }
    )
    return cfg.model_copy(update={"costs": costs})


RETS = [[0.01, 0.0], [0.02, -0.01], [-0.01, 0.03], [0.00, 0.01]]


def test_lagged_fill_and_drift_match_hand_calculation(smoke_config: ExperimentConfig) -> None:
    data = _data(RETS)
    d = data.dates
    res = DailyDrift.from_params({}).run(
        {d[0]: pd.Series({"A": 0.5})},
        data,
        SpreadSlippageImpact.from_params({}),
        _cfg(smoke_config),
    )
    # Decided at d0 close, filled at d1 close -> d1 earns nothing.
    assert res.net_returns.index[0] == d[1]
    assert res.net_returns.iloc[0] == 0.0
    # d2: 0.5 * 2% ; then weight drifts to 0.5*1.02/1.01.
    assert res.net_returns.loc[d[2]] == pytest.approx(0.01)
    w_a = 0.5 * 1.02 / 1.01
    assert res.weights.loc[d[2], "A"] == pytest.approx(w_a)
    assert res.net_returns.loc[d[3]] == pytest.approx(w_a * -0.01)


def test_costs_charged_on_fill_day(smoke_config: ExperimentConfig) -> None:
    data = _data(RETS)
    d = data.dates
    res = DailyDrift.from_params({}).run(
        {d[0]: pd.Series({"A": 0.5, "B": 0.5})},
        data,
        SpreadSlippageImpact.from_params({}),
        _cfg(smoke_config, spread_bps=10.0),
    )
    assert res.costs.loc[d[1]] == pytest.approx(1.0 * 10 / 1e4)
    assert res.turnover.loc[d[1]] == pytest.approx(0.5)
    assert (res.net_returns <= res.gross_returns + 1e-15).all()


def test_signal_cannot_trade_on_its_own_close(smoke_config: ExperimentConfig) -> None:
    """A target decided at t must not earn the return of t+1 if lag is 1 (fill at t+1 close)."""
    data = _data(RETS)
    d = data.dates
    res = DailyDrift.from_params({}).run(
        {d[1]: pd.Series({"A": 1.0})},
        data,
        SpreadSlippageImpact.from_params({}),
        _cfg(smoke_config),
    )
    assert res.net_returns.loc[d[2]] == 0.0  # A's +2% on d2 is NOT captured
    assert res.net_returns.loc[d[3]] == pytest.approx(-0.01)


def test_holding_asset_without_price_raises(smoke_config: ExperimentConfig) -> None:
    data = _data(RETS)
    tri = data.total_return_index.copy()
    tri.loc[data.dates[3], "B"] = np.nan
    broken = MarketData(
        close=tri, total_return_index=tri, volume=data.volume, asset_class=data.asset_class
    )
    with pytest.raises(ValueError, match="without a return"):
        DailyDrift.from_params({}).run(
            {data.dates[0]: pd.Series({"B": 0.5})},
            broken,
            SpreadSlippageImpact.from_params({}),
            _cfg(smoke_config),
        )
