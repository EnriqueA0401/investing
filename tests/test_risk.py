from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import risk
from quantlab.data.market_data import MarketData
from quantlab.testing.leakage import assert_causal
from tests.conftest import make_market_data


def test_var_cvar_normal_sample() -> None:
    x = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200_000))
    t = risk.var_cvar(x)
    assert t.loc["95%", "VaR_hist"] == pytest.approx(0.01645, rel=0.02)
    assert t.loc["95%", "VaR_cornish_fisher"] == pytest.approx(t.loc["95%", "VaR_normal"], rel=0.02)
    assert t.loc["99%", "CVaR_hist"] > t.loc["99%", "VaR_hist"]


def test_drawdown_table() -> None:
    idx = pd.bdate_range("2020-01-01", periods=6)
    r = pd.Series([0.1, -0.2, -0.1, 0.5, -0.05, 0.0], index=idx)
    t = risk.drawdown_table(r)
    assert t.loc[0, "depth"] == pytest.approx(0.8 * 0.9 - 1)
    assert str(t.loc[0, "trough"]) == "2020-01-03"


def test_factor_exposures_recover_betas() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2005-01-31", periods=240, freq="ME")
    f = pd.DataFrame(rng.normal(0, 0.03, (240, 2)), index=idx, columns=["equity", "duration"])
    y = 0.6 * f["equity"] + 0.3 * f["duration"] + rng.normal(0, 0.002, 240)
    out = risk.factor_exposures(y, f)
    assert out.loc["equity", "coef"] == pytest.approx(0.6, abs=0.02)
    assert out.loc["duration", "coef"] == pytest.approx(0.3, abs=0.02)
    assert out.loc["R2", "coef"] > 0.95


def test_hypothetical_stress_is_linear_in_weights() -> None:
    w = pd.Series({"SPY": 0.5, "TLT": 0.5})
    out = risk.hypothetical_stress({"x": w})
    assert out.loc["Equity crash (-30%), flight to quality", "x"] == pytest.approx(-0.10)


@pytest.mark.leakage
def test_regime_labels_are_causal() -> None:
    d = make_market_data(n_days=1200, tickers=("SPY", "AGG"))
    data = MarketData(
        close=d.close,
        total_return_index=d.total_return_index,
        volume=d.volume,
        cash_return=pd.Series(0.0001, index=d.dates),
    )

    def labels(x: MarketData) -> pd.DataFrame:
        out = risk.regime_labels(x).astype(float)
        # A label for month m is usable during month m; compare on the last day it is known.
        out.index = out.index - pd.offsets.MonthEnd(1)
        return out

    me = data.total_return_index.resample("ME").last().index
    cutoffs = [pd.Timestamp(data.dates[data.dates <= m][-1]) for m in me[14:-1:6]]
    assert_causal(labels, data, cutoffs=cutoffs)
