"""Every factor must pass the adversarial look-ahead harness; plus value checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import registry
from quantlab.config import ComponentSpec
from quantlab.contracts import Factor
from quantlab.data.market_data import MarketData
from quantlab.testing.leakage import assert_causal
from tests.conftest import make_market_data

FACTOR_SPECS = [
    ComponentSpec(name="momentum", params={"label": "m121"}),
    ComponentSpec(name="momentum", params={"label": "m61", "lookback_days": 126}),
    ComponentSpec(name="momentum", params={"label": "rev", "lookback_days": 21, "skip_days": 0}),
    ComponentSpec(name="excess_momentum", params={"label": "ts12"}),
    ComponentSpec(name="excess_momentum", params={"label": "ts6", "lookback_days": 126}),
    ComponentSpec(name="realized_vol"),
]


def _data_with_cash() -> MarketData:
    d = make_market_data(n_days=700, late_ticker="CCC", late_offset=200)
    cash = pd.Series(0.0001, index=d.dates)
    return MarketData(
        close=d.close,
        total_return_index=d.total_return_index,
        volume=d.volume,
        asset_class=d.asset_class,
        cash_return=cash,
    )


@pytest.mark.leakage
@pytest.mark.parametrize("spec", FACTOR_SPECS, ids=lambda s: str(s.params.get("label", s.name)))
def test_factor_is_causal(spec: ComponentSpec) -> None:
    factor: Factor = registry.build("factor", spec)
    assert_causal(factor.compute, _data_with_cash())


def test_momentum_value() -> None:
    d = _data_with_cash()
    f = registry.build("factor", ComponentSpec(name="momentum"))
    panel = f.compute(d)
    tri = d.total_return_index
    assert panel.iloc[400, 0] == pytest.approx(tri.iloc[400 - 21, 0] / tri.iloc[400 - 252, 0] - 1)


def test_excess_momentum_subtracts_cash() -> None:
    d = _data_with_cash()
    plain = registry.build("factor", ComponentSpec(name="momentum", params={"skip_days": 0}))
    excess = registry.build("factor", ComponentSpec(name="excess_momentum"))
    cash_12m = 1.0001**252 - 1
    diff = plain.compute(d) - excess.compute(d)
    assert np.allclose(diff.dropna().to_numpy(), cash_12m)


def test_factor_invariant_to_index_scaling() -> None:
    """Ratios only: rescaling the TR index (as a vendor re-adjustment would) changes nothing."""
    d = _data_with_cash()
    scaled = MarketData(
        close=d.close,
        total_return_index=d.total_return_index * 3.7,
        volume=d.volume,
        asset_class=d.asset_class,
        cash_return=d.cash_return,
    )
    for spec in FACTOR_SPECS:
        f = registry.build("factor", spec)
        pd.testing.assert_frame_equal(f.compute(d), f.compute(scaled))
