"""Vendor adapters, tested offline with fabricated vendor payloads."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.components import vendors
from quantlab.components.backtest import DailyDrift
from quantlab.components.costs import SpreadSlippageImpact
from quantlab.components.vendors import (
    VendorDataError,
    YahooFred,
    cash_return_from_rate,
    reconstruct_total_return,
)
from quantlab.config import Asset, ExperimentConfig
from quantlab.data.market_data import MarketData


def test_total_return_includes_dividend_on_ex_date() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"X": [100.0, 100.0, 99.0, 99.0]}, index=idx)
    div = pd.DataFrame({"X": [0.0, 0.0, 1.0, 0.0]}, index=idx)
    tri = reconstruct_total_return(close, div)["X"]
    assert tri.iloc[0] == 100.0
    assert tri.iloc[2] == pytest.approx(100.0)  # price drop fully offset by the dividend
    assert tri.pct_change().iloc[3] == 0.0


def test_total_return_starts_at_inception() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    close = pd.DataFrame({"X": [np.nan, 50.0, 55.0]}, index=idx)
    tri = reconstruct_total_return(close, pd.DataFrame(0.0, index=idx, columns=["X"]))["X"]
    assert np.isnan(tri.iloc[0])
    assert tri.iloc[1:].tolist() == pytest.approx([100.0, 110.0])


def test_cash_uses_previous_session_rate() -> None:
    days = pd.bdate_range("2024-01-01", periods=4)
    rate = pd.Series([5.0, 5.0, 10.0, 10.0], index=days)
    cash = cash_return_from_rate(rate, days, lag_days=1)
    assert np.isnan(cash.iloc[0])
    assert cash.iloc[2] == pytest.approx(1.05 ** (1 / 252) - 1)  # day-2 rate not yet used
    assert cash.iloc[3] == pytest.approx(1.10 ** (1 / 252) - 1)


def _fake_yahoo(idx: pd.DatetimeIndex, adj_bias: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (len(idx), 2)), axis=0)
    fields = {
        "Close": close,
        "Adj Close": close * (1 + adj_bias * np.arange(len(idx)))[:, None],
        "Volume": np.full_like(close, 1e6),
        "Dividends": np.zeros_like(close),
        "Capital Gains": np.zeros_like(close),
    }
    cols = pd.MultiIndex.from_product([list(fields), ["AAA", "BBB"]])
    data = np.hstack(list(fields.values()))
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> pd.DatetimeIndex:
    idx = pd.bdate_range("2020-01-01", periods=60)
    monkeypatch.setattr(
        vendors, "fetch_fred_series", lambda *a, **k: pd.Series(4.0, index=idx, name="DTB3")
    )
    return idx


ASSETS = [Asset(ticker="AAA", asset_class="equity"), Asset(ticker="BBB", asset_class="credit")]


def test_fetch_builds_aligned_market_data(
    offline: pd.DatetimeIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = YahooFred.from_params({})
    monkeypatch.setattr(src, "_download", lambda *a: _fake_yahoo(offline))
    data = src.fetch(ASSETS, date(2020, 1, 1), None)
    assert data.tickers == ["AAA", "BBB"]
    assert data.cash_return is not None
    assert data.cash_return.iloc[5] == pytest.approx(1.04 ** (1 / 252) - 1)
    assert data.asset_class == {"AAA": "equity", "BBB": "credit"}


def test_fetch_fails_when_vendor_adjustment_disagrees(
    offline: pd.DatetimeIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = YahooFred.from_params({})
    monkeypatch.setattr(src, "_download", lambda *a: _fake_yahoo(offline, adj_bias=0.01))
    with pytest.raises(VendorDataError, match="diverges"):
        src.fetch(ASSETS, date(2020, 1, 1), None)


def test_uninvested_weight_earns_cash(smoke_config: ExperimentConfig) -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    tri = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    vol = pd.DataFrame(1e9, index=idx, columns=["A", "B"])
    cash = pd.Series(0.0001, index=idx)
    data = MarketData(close=tri, total_return_index=tri, volume=vol, cash_return=cash)
    costs = smoke_config.costs.model_copy(
        update={
            "half_spread_bps": {"default": 0.0},
            "commission_bps": 0.0,
            "slippage_bps": 0.0,
            "impact_coefficient": 0.0,
        }
    )
    cfg = smoke_config.model_copy(update={"costs": costs})
    res = DailyDrift.from_params({}).run(
        {idx[0]: pd.Series({"A": 0.25})}, data, SpreadSlippageImpact.from_params({}), cfg
    )
    # Flat prices: return = 100% cash on the fill day, then 75% cash.
    assert res.net_returns.iloc[0] == pytest.approx(0.0001)
    assert res.net_returns.iloc[1] == pytest.approx(0.75 * 0.0001)
    assert res.excess_returns().iloc[1] == pytest.approx(-0.25 * 0.0001)
