"""End-to-end orchestration: determinism and structural absence of look-ahead."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab import registry
from quantlab.components.portfolios import ConstraintViolationError, check_constraints
from quantlab.config import ComponentSpec, ExperimentConfig
from quantlab.contracts import DecisionContext, SignalGenerator
from quantlab.data.market_data import MarketData
from quantlab.pipeline import load_data, run_experiment
from quantlab.splits import HoldoutLockedError

SEEN: list[tuple[pd.Timestamp, pd.Timestamp]] = []


@registry.register("signal", "_spy_test")
class _Spy(SignalGenerator):
    """Records (as_of, last date visible) at every decision."""

    def generate(self, ctx: DecisionContext) -> pd.Series:
        SEEN.append((ctx.as_of, ctx.history.end))
        return pd.Series(1.0, index=ctx.investable)


def test_run_is_deterministic(smoke_config: ExperimentConfig) -> None:
    a = run_experiment(smoke_config)
    b = run_experiment(smoke_config)
    pd.testing.assert_series_equal(a.strategy.net_returns, b.strategy.net_returns)
    assert a.metrics == b.metrics


@pytest.mark.leakage
def test_components_never_see_the_future(smoke_config: ExperimentConfig) -> None:
    SEEN.clear()
    pipe = smoke_config.pipeline.model_copy(update={"signal": ComponentSpec(name="_spy_test")})
    res = run_experiment(smoke_config.model_copy(update={"pipeline": pipe}))
    assert SEEN, "spy was never called"
    assert all(end <= as_of for as_of, end in SEEN)
    assert res.strategy.net_returns.index.max() <= pd.Timestamp(smoke_config.split.research_end)


@pytest.mark.leakage
def test_research_results_independent_of_holdout_data(smoke_config: ExperimentConfig) -> None:
    """Changing only post-research_end data must not change any research-stage number."""
    data, _ = load_data(smoke_config)
    cut = pd.Timestamp(smoke_config.split.research_end)
    tri = data.total_return_index.copy()
    tri.loc[tri.index > cut] *= 3.0
    altered = MarketData(
        close=tri, total_return_index=tri, volume=data.volume, asset_class=data.asset_class
    )
    a = run_experiment(smoke_config, data=data)
    b = run_experiment(smoke_config, data=altered)
    assert a.metrics == b.metrics


def test_final_stage_requires_unlock(smoke_config: ExperimentConfig) -> None:
    with pytest.raises(HoldoutLockedError):
        run_experiment(smoke_config, stage="final")
    res = run_experiment(smoke_config, stage="final", unlock_reason="test")
    assert res.strategy.net_returns.index.min() >= pd.Timestamp(smoke_config.split.holdout_start)


def test_constraint_check_rejects_breaches(smoke_config: ExperimentConfig) -> None:
    c = smoke_config.constraints
    with pytest.raises(ConstraintViolationError, match="max_weight"):
        check_constraints(pd.Series({"SPY": 0.5}), c, ["SPY"])
    with pytest.raises(ConstraintViolationError, match="short"):
        check_constraints(pd.Series({"SPY": -0.1}), c, ["SPY"])
    with pytest.raises(ConstraintViolationError, match="non-investable"):
        check_constraints(pd.Series({"SPY": 0.1}), c, ["AGG"])


@pytest.mark.leakage
def test_validation_issues_do_not_reveal_holdout(smoke_config: ExperimentConfig) -> None:
    data, _ = load_data(smoke_config)
    tri = data.total_return_index.copy()
    tri.loc[
        tri.index > pd.Timestamp(smoke_config.split.holdout_start), "SPY"
    ] *= -1  # corrupt holdout
    corrupted = MarketData(
        close=tri, total_return_index=tri, volume=data.volume, asset_class=data.asset_class
    )
    res = run_experiment(smoke_config, data=corrupted)  # research stage must not notice
    assert res.issues == []
