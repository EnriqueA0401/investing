from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import ComponentSpec
from quantlab.data.market_data import MarketData
from quantlab.ml_diagnostic import build_panel, cpcv_ic
from quantlab.robustness import random_filter_null
from tests.conftest import make_market_data

TICKERS = tuple("ABCDEFGH")


def _data() -> MarketData:
    d = make_market_data(n_days=2000, tickers=TICKERS, seed=4)
    return MarketData(
        close=d.close,
        total_return_index=d.total_return_index,
        volume=d.volume,
        cash_return=pd.Series(0.0001, index=d.dates),
    )


def _monthly_targets(data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    me = data.total_return_index.resample("ME").last().index
    dates = pd.DatetimeIndex([data.dates[data.dates <= m][-1] for m in me])
    base = pd.DataFrame(1 / 8, index=dates, columns=list(TICKERS))
    fwd = data.total_return_index.loc[dates].shift(-1) / data.total_return_index.loc[dates] - 1
    oracle = base.where(fwd.rank(axis=1) > 4, 0.0)  # keeps the 4 future winners
    rng = np.random.default_rng(0)
    blind = base.where(rng.random(base.shape) > 0.5, 0.0)
    return base, oracle, blind


def test_random_null_flags_oracle_and_not_noise() -> None:
    data = _data()
    base, oracle, blind = _monthly_targets(data)
    assert random_filter_null(base, oracle, data, n_sims=300)["p_value"] < 0.01
    assert random_filter_null(base, blind, data, n_sims=300)["p_value"] > 0.01


def test_ml_cpcv_diagnostic_runs_on_noise() -> None:
    data = _data()
    specs = [
        ComponentSpec(name="momentum", params={"label": "m", "lookback_days": 126}),
        ComponentSpec(name="realized_vol", params={"label": "v"}),
    ]
    signs = {"m": 1.0, "v": -1.0}
    panel = build_panel(data, specs, signs, list(TICKERS), data.dates[300])
    out = cpcv_ic(panel, signs)
    assert len(out["paths"]) == 5
    assert abs(out["median_ic_ridge"]) < 0.2  # pure noise: no real predictability
