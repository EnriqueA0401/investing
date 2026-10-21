"""Validators, metrics, registry and logging."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from quantlab import metrics as m
from quantlab import registry
from quantlab.components.validators import BasicIntegrity
from quantlab.config import ComponentSpec
from quantlab.contracts import Severity
from quantlab.data.market_data import MarketData
from quantlab.logging_setup import JsonFormatter
from tests.conftest import make_market_data


def _with(data: MarketData, row: int, col: str, value: float) -> MarketData:
    tri = data.total_return_index.copy()
    tri.iloc[row, tri.columns.get_loc(col)] = value
    return MarketData(
        close=tri, total_return_index=tri, volume=data.volume, asset_class=data.asset_class
    )


def _checks(data: MarketData) -> dict[str, Severity]:
    return {i.check: i.severity for i in BasicIntegrity.from_params({}).validate(data)}


def test_clean_data_has_no_issues(market_data: MarketData) -> None:
    assert _checks(market_data) == {}


def test_validator_flags_bad_prices_gaps_and_jumps(market_data: MarketData) -> None:
    assert _checks(_with(market_data, 50, "AAA", -1.0))["non_positive"] is Severity.ERROR
    assert _checks(_with(market_data, 50, "AAA", np.nan))["internal_gap"] is Severity.WARNING
    tri = market_data.total_return_index.copy()
    tri.iloc[50:60, 0] = np.nan
    long_gap = MarketData(close=tri, total_return_index=tri, volume=market_data.volume)
    assert _checks(long_gap)["internal_gap"] is Severity.ERROR
    jump = _with(market_data, 50, "BBB", float(market_data.total_return_index.iloc[50, 1]) * 2)
    assert _checks(jump)["extreme_return"] is Severity.WARNING


def test_validator_ignores_pre_inception_nans() -> None:
    assert _checks(make_market_data(late_ticker="CCC", late_offset=100)) == {}


def test_metrics_hand_calculations() -> None:
    idx = pd.bdate_range("2024-01-01", periods=252)
    r = pd.Series(0.001, index=idx)
    assert m.annualized_return(r) == pytest.approx(1.001**252 - 1)
    dd = pd.Series([0.10, -0.50, 0.20], index=idx[:3])
    assert m.max_drawdown(dd) == pytest.approx(-0.5)
    first_day_loss = pd.Series([-0.10, 0.05], index=idx[:2])
    assert m.max_drawdown(first_day_loss) == pytest.approx(-0.10)
    noise = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 252), index=idx)
    assert m.information_ratio(noise, noise) != m.information_ratio(noise, noise * 0.5)
    assert m.beta(noise, noise) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="empty"):
        m.sharpe_ratio(pd.Series(dtype=float))


def test_registry_unknown_component_lists_alternatives() -> None:
    with pytest.raises(registry.UnknownComponentError, match="equal_weight"):
        registry.build("portfolio", ComponentSpec(name="nope"))


def test_component_params_are_strict() -> None:
    with pytest.raises(ValueError, match="extra"):
        registry.build("validator", ComponentSpec(name="basic_integrity", params={"typo": 1}))


def test_json_log_lines() -> None:
    rec = logging.makeLogRecord(
        {"name": "quantlab.x", "msg": "hi", "levelname": "INFO", "run_id": "r1"}
    )
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["msg"] == "hi"
    assert payload["run_id"] == "r1"
