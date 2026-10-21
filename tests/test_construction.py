"""Portfolio constructors, overlays and signals (including the ML signal's purging)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import registry
from quantlab.components.portfolios import check_constraints
from quantlab.components.signals import RidgeTopN, ZScoreComboTopN
from quantlab.config import ComponentSpec, PortfolioConstraints
from quantlab.contracts import DecisionContext
from quantlab.data.market_data import MarketData
from quantlab.pipeline import drift_weights
from quantlab.risk_model import cap_and_redistribute, ledoit_wolf_cov, trailing_returns
from tests.conftest import make_market_data

TICKERS = ("A", "B", "C", "D", "E", "F", "G", "H")


def _data(n_days: int = 1500) -> MarketData:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    vols = np.linspace(0.003, 0.02, len(TICKERS))
    common = rng.normal(0, 0.006, n_days)[:, None]
    rets = rng.normal(0.0003, 1.0, (n_days, len(TICKERS))) * vols + common * (vols / 0.01)
    tri = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=list(TICKERS))
    vol = pd.DataFrame(1e7, index=dates, columns=list(TICKERS))
    return MarketData(
        close=tri, total_return_index=tri, volume=vol, cash_return=pd.Series(0.0, index=dates)
    )


def _ctx(
    data: MarketData,
    features: dict[str, pd.DataFrame] | None = None,
    cap: float = 0.2,
    current: pd.Series | None = None,
) -> DecisionContext:
    return DecisionContext(
        as_of=data.end,
        history=data,
        investable=data.tickers,
        previous_weights=pd.Series(dtype=float),
        constraints=PortfolioConstraints(max_weight=cap),
        rng=np.random.default_rng(0),
        features=features or {},
        current_weights=current if current is not None else pd.Series(dtype=float),
    )


ALL = pd.Series(1.0, index=list(TICKERS))


@pytest.mark.parametrize("name", ["equal_weight", "inverse_vol", "erc", "hrp", "min_variance"])
def test_constructors_respect_mandate(name: str) -> None:
    data = _data()
    ctx = _ctx(data)
    w = registry.build("portfolio", ComponentSpec(name=name)).construct(ALL, ctx)
    check_constraints(w, ctx.constraints, ctx.investable)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_risk_based_overweights_low_vol_assets() -> None:
    ctx = _ctx(_data(), cap=1.0)
    for name in ("inverse_vol", "erc", "hrp", "min_variance"):
        w = registry.build("portfolio", ComponentSpec(name=name)).construct(ALL, ctx)
        assert w["A"] > w["H"], name


def test_erc_equalises_risk_contributions() -> None:
    data = _data()
    ctx = _ctx(data, cap=1.0)
    w = registry.build("portfolio", ComponentSpec(name="erc")).construct(ALL, ctx)
    cov = ledoit_wolf_cov(trailing_returns(data, list(TICKERS), 252)).to_numpy()
    rc = w.to_numpy() * (cov @ w.to_numpy())
    assert rc.std() / rc.mean() < 1e-3


def test_robust_mv_tilts_toward_high_scores_but_stays_diversified() -> None:
    ctx = _ctx(_data())
    scores = pd.Series(np.arange(len(TICKERS), dtype=float), index=list(TICKERS))
    w = registry.build("portfolio", ComponentSpec(name="robust_mean_variance")).construct(
        scores, ctx
    )
    check_constraints(w, ctx.constraints, ctx.investable)
    neutral = registry.build("portfolio", ComponentSpec(name="robust_mean_variance")).construct(
        pd.Series(1.0, index=list(TICKERS)), ctx
    )
    assert (w - neutral)["H"] > 0 > (w - neutral)["A"]


def test_cap_and_redistribute() -> None:
    w = cap_and_redistribute(pd.Series({"a": 0.7, "b": 0.2, "c": 0.1}), cap=0.4)
    assert w.max() <= 0.4 + 1e-12
    assert w.sum() == pytest.approx(1.0)
    starved = cap_and_redistribute(pd.Series({"a": 1.0, "b": 1.0}), cap=0.2)
    assert starved.sum() == pytest.approx(0.4)  # residual stays in cash


def test_trend_filter_sends_slot_to_cash() -> None:
    data = _data()
    feat = pd.DataFrame(1.0, index=data.dates, columns=list(TICKERS))
    feat.loc[data.end, ["A", "B"]] = -0.1
    ctx = _ctx(data, {"tsmom_12": feat})
    w = pd.Series(1 / 8, index=list(TICKERS))
    out = registry.build("risk", ComponentSpec(name="trend_filter")).adjust(w, ctx)
    assert out[["A", "B"]].sum() == 0
    assert out.sum() == pytest.approx(0.75)


def test_vol_target_only_scales_down() -> None:
    ctx = _ctx(_data())
    w = pd.Series(1 / 8, index=list(TICKERS))
    low = registry.build("risk", ComponentSpec(name="vol_target", params={"target_vol": 0.01}))
    high = registry.build("risk", ComponentSpec(name="vol_target", params={"target_vol": 5.0}))
    assert low.adjust(w, ctx).sum() < 1.0
    pd.testing.assert_series_equal(high.adjust(w, ctx), w)


def test_turnover_cap_limits_trading() -> None:
    cur = pd.Series({"A": 0.2, "B": 0.2})
    ctx = _ctx(_data(), current=cur)
    tgt = pd.Series({"C": 0.2, "D": 0.2})
    out = registry.build(
        "risk", ComponentSpec(name="turnover_cap", params={"max_turnover": 0.1})
    ).adjust(tgt, ctx)
    names = out.index.union(cur.index)
    assert (
        out.reindex(names).fillna(0) - cur.reindex(names).fillna(0)
    ).abs().sum() / 2 == pytest.approx(0.1)


def test_drift_weights() -> None:
    data = _data()
    t0 = data.dates[-21]
    w = drift_weights(pd.Series({"A": 0.5}), data, t0)
    growth = data.total_return_index["A"].iloc[-1] / data.total_return_index.loc[t0, "A"]
    assert w["A"] == pytest.approx(0.5 * growth / (0.5 * growth + 0.5))


FEATS = {"m": 1.0, "v": -1.0}


def _features(data: MarketData) -> dict[str, pd.DataFrame]:
    tri = data.total_return_index
    return {"m": tri / tri.shift(126) - 1, "v": data.returns().rolling(63).std()}


@pytest.mark.leakage
def test_ridge_signal_ignores_future_rows_in_feature_panels() -> None:
    """Even if handed full-sample feature panels, predictions at t depend only on <= t."""
    data = _data()
    t = data.dates[1100]
    hist = data.asof(t)
    sig = RidgeTopN(RidgeTopN.Params(features=FEATS, n=3, min_train_months=12))
    truncated = sig.predict(_ctx(hist, _features(hist)))
    full_feats = _features(data)  # contains rows after t
    leaky_ctx = DecisionContext(
        as_of=t,
        history=hist,
        investable=hist.tickers,
        previous_weights=pd.Series(dtype=float),
        constraints=PortfolioConstraints(),
        rng=np.random.default_rng(0),
        features=full_feats,
    )
    pd.testing.assert_series_equal(truncated, sig.predict(leaky_ctx))


def test_ridge_training_labels_end_by_as_of() -> None:
    data = _data()
    hist = data.asof(data.dates[900])
    sig = RidgeTopN(RidgeTopN.Params(features=FEATS, n=3, min_train_months=12))
    x, y, months = sig.training_set(_ctx(hist, _features(hist)))
    assert months > 12
    assert len(x) == len(y)


def test_ridge_falls_back_to_combo_without_history() -> None:
    data = make_market_data(n_days=400, tickers=("A", "B", "C", "D"))
    feats = _features(data)
    ctx = _ctx(data, feats)
    ridge = RidgeTopN(RidgeTopN.Params(features=FEATS, n=2)).predict(ctx)
    combo = ZScoreComboTopN(ZScoreComboTopN.Params(features=FEATS, n=2)).combo(ctx)
    pd.testing.assert_series_equal(ridge, combo)
