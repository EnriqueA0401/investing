from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.market_data import MarketData
from tests.conftest import make_market_data


def test_asof_excludes_future_rows(market_data: MarketData) -> None:
    t = market_data.dates[100]
    view = market_data.asof(t)
    assert view.end == t
    assert len(view.dates) == 101
    assert (view.volume.index <= t).all()


def test_misaligned_panels_rejected(market_data: MarketData) -> None:
    with pytest.raises(ValueError, match="not aligned"):
        MarketData(
            close=market_data.close.iloc[:-1],
            total_return_index=market_data.total_return_index,
            volume=market_data.volume,
        )


def test_tz_aware_index_rejected(market_data: MarketData) -> None:
    tri = market_data.total_return_index.tz_localize("UTC")
    with pytest.raises(ValueError, match="tz-naive"):
        MarketData(close=tri, total_return_index=tri, volume=tri)


def test_investable_requires_history_and_is_causal() -> None:
    data = make_market_data(late_ticker="CCC", late_offset=300)
    # 300 NaN rows then data: CCC needs 252 obs -> investable from row 300+251.
    before = data.asof(data.dates[300 + 250])
    after = data.asof(data.dates[300 + 251])
    assert "CCC" not in before.investable(252)
    assert "CCC" in after.investable(252)


def test_returns_nan_before_inception() -> None:
    data = make_market_data(late_ticker="BBB", late_offset=10)
    r = data.returns()["BBB"]
    assert r.iloc[:11].isna().all()
    assert pd.notna(r.iloc[11])
