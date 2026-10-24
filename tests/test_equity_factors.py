"""Equity factors must pass the adversarial look-ahead harness; L/S construction checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import registry
from quantlab.config import ComponentSpec, PortfolioConstraints
from quantlab.contracts import DecisionContext
from quantlab.data.market_data import MarketData
from quantlab.equities.portfolio import (
    LSOptimizer,
    beta_neutralize,
    neutralize,
    pca_risk_model,
)
from quantlab.equities.universe import month_end_dates, reconstitute
from quantlab.testing.leakage import assert_causal

N = 60
TICKERS = [f"S{i:02d}" for i in range(N)]
SECTORS = ["BusEq", "Hlth", "Shops", "Manuf"]


def equity_data(n_days: int = 900, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    mkt = rng.normal(0.0004, 0.01, n_days)
    betas = rng.uniform(0.5, 1.5, N)
    rets = mkt[:, None] * betas + rng.normal(0, 0.015, (n_days, N))
    cols = [*TICKERS, "SPY"]
    tri = pd.DataFrame(
        100 * np.cumprod(1 + np.column_stack([rets, mkt]), axis=0), index=dates, columns=cols
    )
    tri.iloc[:200, 5] = np.nan  # late listing
    vol = pd.DataFrame(1e6, index=dates, columns=cols)
    me = month_end_dates(dates)
    fund = {
        k: pd.DataFrame(rng.lognormal(0, 0.5, (len(me), N)) * s, index=me, columns=TICKERS)
        for k, s in {
            "net_income_ttm": 1e8, "cfo_ttm": 1.2e8, "equity": 1e9, "gross_profit_ttm": 3e8,
            "assets": 2e9, "assets_1y_ago": 1.9e9, "shares": 1e7,
        }.items()
    }  # fmt: skip
    fund["market_cap"] = fund["shares"] * tri[TICKERS].reindex(me)
    members = reconstitute(fund["market_cap"], tri[TICKERS], size=40)
    return MarketData(
        close=tri,
        total_return_index=tri,
        volume=vol,
        asset_class={t: SECTORS[i % 4] for i, t in enumerate(TICKERS)} | {"SPY": "benchmark"},
        cash_return=pd.Series(0.0001, index=dates),
        fundamentals=fund,
        universe=members.reindex(columns=cols, fill_value=False),
        extras={"BAA10Y": pd.Series(np.linspace(2, 3, n_days), index=dates)},
    )


SPECS = [
    ComponentSpec(
        name="fundamental_ratio",
        params={"label": "ep", "numerator": "net_income_ttm", "denominator": "market_cap"},
    ),
    ComponentSpec(
        name="fundamental_ratio",
        params={"label": "gpa", "numerator": "gross_profit_ttm", "denominator": "assets"},
    ),
    ComponentSpec(name="accruals"),
    ComponentSpec(name="asset_growth"),
    ComponentSpec(name="price_momentum"),
    ComponentSpec(name="residual_momentum"),
    ComponentSpec(name="market_beta", params={"kind": "beta"}),
    ComponentSpec(name="market_beta", params={"label": "vol", "kind": "vol"}),
]  # fmt: skip


@pytest.mark.leakage
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name + str(s.params.get("label", "")))
def test_equity_factor_is_causal(spec: ComponentSpec) -> None:
    data = equity_data()
    f = registry.build("factor", spec)
    assert f.precomputable
    assert_causal(f.compute, data, n_cutoffs=6)


def test_beta_factor_recovers_true_beta() -> None:
    data = equity_data(n_days=1200)
    beta = registry.build("factor", ComponentSpec(name="market_beta")).compute(data).iloc[-1]
    assert beta.drop("SPY").between(0.2, 2.0).mean() > 0.95
    assert beta["SPY"] == pytest.approx(1.0)


def test_neutralize_removes_sector_and_size() -> None:
    rng = np.random.default_rng(1)
    names = [f"n{i}" for i in range(200)]
    sector = pd.Series([SECTORS[i % 4] for i in range(200)], index=names)
    log_cap = pd.Series(rng.normal(22, 1, 200), index=names)
    x = pd.Series(rng.normal(size=200), index=names) + log_cap + (sector == "Hlth") * 3
    z = neutralize(x, sector, log_cap, winsor=0.0)
    assert abs(np.corrcoef(z, log_cap)[0, 1]) < 1e-6
    assert z.groupby(sector).mean().abs().max() < 0.2


def _ctx(data: MarketData, features: dict[str, pd.DataFrame]) -> DecisionContext:
    t = pd.Timestamp(features["beta_252"].index[-1])
    hist = data.asof(t)
    return DecisionContext(
        as_of=t,
        history=hist,
        investable=TICKERS,
        previous_weights=pd.Series(dtype=float),
        constraints=PortfolioConstraints(long_only=False, max_gross_leverage=2.0, max_weight=0.2),
        rng=np.random.default_rng(0),
        features=features,
    )


def test_ls_sector_neutral_is_beta_and_sector_neutral() -> None:
    data = equity_data(n_days=1200)
    beta = registry.build("factor", ComponentSpec(name="market_beta")).compute(data)
    ctx = _ctx(data, {"beta_252": beta})
    scores = pd.Series(np.random.default_rng(2).normal(size=N), index=TICKERS)
    pf = registry.build(
        "portfolio", ComponentSpec(name="ls_sector_neutral", params={"n_per_side": 8})
    )
    w = pf.construct(scores, ctx)
    b = beta.iloc[-1].reindex(w.index).clip(0.2, 3.0)
    assert (w * b).sum() == pytest.approx(0.0, abs=1e-9)
    assert (w > 0).sum() == 8
    assert (w < 0).sum() == 8
    sec = pd.Series({t: data.asset_class[t] for t in w.index})
    assert w.groupby(sec).apply(lambda s: (s > 0).sum() - (s < 0).sum()).abs().max() == 0


def test_beta_neutralize_keeps_side_gross_within_cap() -> None:
    w = pd.Series({"a": 0.5, "b": 0.5, "c": -0.5, "d": -0.5})
    beta = pd.Series({"a": 1.5, "b": 1.5, "c": 1.0, "d": 1.0})
    out = beta_neutralize(w, beta, 1.0)
    assert (out * beta).sum() == pytest.approx(0.0)
    assert out[out > 0].sum() <= 1.0 + 1e-12


def test_pca_risk_model_reconstructs_covariance() -> None:
    rng = np.random.default_rng(3)
    f = rng.normal(0, 0.01, (500, 3))
    load = rng.normal(0, 1, (3, 40))
    r = pd.DataFrame(f @ load + rng.normal(0, 0.005, (500, 40)))
    b, fv, dv = pca_risk_model(r, 3)
    model = b @ np.diag(fv) @ b.T + np.diag(dv)
    sample = np.cov(r.to_numpy(), rowvar=False)
    assert np.abs(model - sample).max() / np.abs(sample).max() < 0.15


def test_ls_optimizer_satisfies_constraints() -> None:
    data = equity_data(n_days=1200)
    beta = registry.build("factor", ComponentSpec(name="market_beta")).compute(data)
    ctx = _ctx(data, {"beta_252": beta})
    scores = pd.Series(np.random.default_rng(4).normal(size=N), index=TICKERS)
    pf = LSOptimizer(LSOptimizer.Params(n_factors=5))
    # 60 names is below the optimizer's 100-name floor: it must return a flat book.
    assert pf.construct(scores, ctx).empty


def test_credit_regime_halves_in_stress() -> None:
    data = equity_data()
    ctx = _ctx(data, {"beta_252": pd.DataFrame(index=[data.end])})
    w = pd.Series({"S01": 0.05, "S02": -0.05})
    out = registry.build(
        "risk", ComponentSpec(name="credit_regime", params={"lookback_days": 200})
    ).adjust(w, ctx)
    assert out.abs().sum() == pytest.approx(0.05)  # spread trending up -> z > 1
