"""The look-ahead harness must catch every classic leak and pass causal code."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from quantlab.data.market_data import MarketData
from quantlab.testing.leakage import LookAheadError, assert_causal
from tests.conftest import make_market_data

pytestmark = pytest.mark.leakage

LEAKY: dict[str, Callable[[MarketData], pd.DataFrame]] = {
    "negative_shift": lambda d: d.returns().shift(-1),
    "centred_window": lambda d: d.returns().rolling(21, center=True).mean(),
    "full_sample_zscore": lambda d: (d.returns() - d.returns().mean()) / d.returns().std(),
    "backfill": lambda d: d.total_return_index.pct_change(21, fill_method=None).bfill(),
    "adjusted_price_level": lambda d: d.total_return_index / d.total_return_index.iloc[-1],
    "future_max": lambda d: d.total_return_index / d.total_return_index.max(),
}

CAUSAL: dict[str, Callable[[MarketData], pd.DataFrame]] = {
    "momentum_12_1": lambda d: d.total_return_index.shift(21) / d.total_return_index.shift(252) - 1,
    "trailing_vol": lambda d: d.returns().rolling(63).std(),
    "expanding_zscore": lambda d: (
        (d.returns() - d.returns().expanding().mean()) / d.returns().expanding().std()
    ),
    "cross_sectional_rank": lambda d: d.returns().rolling(21).sum().rank(axis=1),
    "ffill": lambda d: d.total_return_index.ffill().pct_change(5, fill_method=None),
}


@pytest.mark.parametrize("name", sorted(LEAKY))
def test_harness_detects_leak(name: str, market_data: MarketData) -> None:
    with pytest.raises(LookAheadError):
        assert_causal(LEAKY[name], market_data)


@pytest.mark.parametrize("name", sorted(CAUSAL))
def test_harness_accepts_causal(name: str, market_data: MarketData) -> None:
    assert_causal(CAUSAL[name], market_data)


def test_harness_detects_bfill_at_late_inception() -> None:
    """bfill across an asset's listing date leaks post-inception prices into pre-inception rows."""
    data = make_market_data(late_ticker="CCC", late_offset=300)

    def leaky(d: MarketData) -> pd.DataFrame:
        return d.total_return_index.bfill().pct_change(fill_method=None)

    with pytest.raises(LookAheadError):
        assert_causal(leaky, data, cutoffs=[data.dates[250]])
