"""Single-stock factors. Each returns a (month-end dates x tickers) panel whose row
``d`` uses only data public at the close of ``d``.

Fundamental panels in ``MarketData.fundamentals`` are already point-in-time
(first-reported, usable the day after filing; see ``equities.fundamentals``).
Every factor here has an ``assert_causal`` test.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from quantlab.contracts import ComponentParams, Factor
from quantlab.data.market_data import MarketData
from quantlab.registry import register


def _month_end_index(history: MarketData) -> pd.DatetimeIndex:
    if "market_cap" in history.fundamentals:
        return pd.DatetimeIndex(history.fundamentals["market_cap"].index)
    d = history.dates
    s = pd.Series(d, index=d)
    return pd.DatetimeIndex(s.groupby([d.year, d.month]).max().to_numpy())


@register("factor", "fundamental_ratio")
class FundamentalRatio(Factor):
    """numerator / denominator from point-in-time fundamentals (denominator > 0).

    Used for (pre-registered theses in docs/hypotheses/equities/):
    * Value (S-VAL): earnings, cash-flow and book yield = NI_ttm, CFO_ttm, equity
      over market cap. Thesis: cheap stocks earn a premium for distress risk and
      investor over-extrapolation of poor fundamentals. Expected sign: +.
    * Quality (S-QUAL): gross profit / assets (Novy-Marx 2013) and ROE. Thesis:
      the market under-prices the persistence of profitability. Expected sign: +.
    """

    hypothesis_id: ClassVar[str] = "S-VAL/S-QUAL"
    precomputable: ClassVar[bool] = True

    class Params(ComponentParams):
        label: str
        numerator: str
        denominator: str

    params: Params

    def compute(self, history: MarketData) -> pd.DataFrame:
        f = history.fundamentals
        num, den = f[self.params.numerator], f[self.params.denominator]
        num, den = num.align(den, join="inner")
        return num / den.where(den > 0)


@register("factor", "accruals")
class Accruals(Factor):
    """(NI_ttm - CFO_ttm) / assets (Sloan 1996). Thesis: earnings backed by accruals
    rather than cash are less persistent and investors fixate on headline earnings.
    Expected sign vs forward returns: negative (used with sign -1)."""

    hypothesis_id: ClassVar[str] = "S-QUAL"
    precomputable: ClassVar[bool] = True

    class Params(ComponentParams):
        label: str = "accruals"

    params: Params

    def compute(self, history: MarketData) -> pd.DataFrame:
        f = history.fundamentals
        assets = f["assets"]
        return (f["net_income_ttm"] - f["cfo_ttm"]) / assets.where(assets > 0)


@register("factor", "asset_growth")
class AssetGrowth(Factor):
    """Year-on-year total-asset growth (Cooper, Gulen & Schill 2008; FF 'CMA').
    Thesis: aggressive investors/empire builders earn lower subsequent returns
    (q-theory: high investment <-> low cost of capital). Expected sign: negative."""

    hypothesis_id: ClassVar[str] = "S-INV"
    precomputable: ClassVar[bool] = True

    class Params(ComponentParams):
        label: str = "asset_growth"

    params: Params

    def compute(self, history: MarketData) -> pd.DataFrame:
        f = history.fundamentals
        prev = f["assets_1y_ago"]
        return f["assets"] / prev.where(prev > 0) - 1.0


def _monthly_returns(history: MarketData, idx: pd.DatetimeIndex) -> pd.DataFrame:
    tri = history.total_return_index.reindex(idx)
    return tri.pct_change(fill_method=None)


@register("factor", "residual_momentum")
class ResidualMomentum(Factor):
    """Blitz, Huij & Martens (2011): 12-1 month cumulative residual return from a
    36-month market regression, scaled by residual volatility. Thesis: same
    under-reaction as momentum, minus the time-varying market beta that drives
    momentum crashes. Expected sign: +."""

    hypothesis_id: ClassVar[str] = "S-MOM"
    precomputable: ClassVar[bool] = True

    class Params(ComponentParams):
        label: str = "resid_mom"
        market: str = "SPY"
        window_months: int = Field(default=36, ge=24)
        min_months: int = Field(default=24, ge=12)

    params: Params

    def compute(self, history: MarketData) -> pd.DataFrame:
        p = self.params
        idx = _month_end_index(history)
        idx = idx[idx <= history.end]
        r = _monthly_returns(history, idx)
        mkt = r.pop(p.market) if p.market in r else None
        if mkt is None:
            raise KeyError(f"market series {p.market} missing from data")
        out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
        rv, mv = r.to_numpy(), mkt.to_numpy()
        for i in range(p.window_months, len(idx)):
            y = rv[i - p.window_months + 1 : i + 1]
            x = mv[i - p.window_months + 1 : i + 1]
            ok = ~np.isnan(y) & ~np.isnan(x)[:, None]
            n = ok.sum(axis=0)
            xm = np.where(ok, x[:, None], np.nan)
            ym = np.where(ok, y, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                xbar, ybar = np.nanmean(xm, axis=0), np.nanmean(ym, axis=0)
                beta = np.nansum((xm - xbar) * (ym - ybar), axis=0) / np.nansum(
                    (xm - xbar) ** 2, axis=0
                )
                resid = ym - (ybar + beta * (xm - xbar))
                score = np.nansum(resid[-12:-1], axis=0) / np.nanstd(resid, axis=0, ddof=1)
            score[n < p.min_months] = np.nan
            out.iloc[i] = score
        return out


@register("factor", "market_beta")
class MarketBeta(Factor):
    """Trailing daily beta to the market (also the BAB/low-risk signal, S-LVOL:
    leverage-constrained investors bid up high-beta stocks; expected sign -)."""

    hypothesis_id: ClassVar[str] = "S-LVOL"
    precomputable: ClassVar[bool] = True

    class Params(ComponentParams):
        label: str = "beta_252"
        market: str = "SPY"
        window_days: int = Field(default=252, ge=60)
        min_days: int = Field(default=200, ge=40)
        kind: Literal["beta", "vol"] = "beta"

    params: Params

    def compute(self, history: MarketData) -> pd.DataFrame:
        p = self.params
        rets = history.returns()
        mkt = rets[p.market]
        idx = _month_end_index(history)
        idx = idx[idx <= history.end]
        pos = rets.index.get_indexer(idx)
        out = pd.DataFrame(np.nan, index=idx, columns=rets.columns)
        rv, mv = rets.to_numpy(), mkt.to_numpy()
        for row, j in enumerate(pos):
            if j < 0 or j + 1 < p.min_days:
                continue
            y = rv[max(0, j - p.window_days + 1) : j + 1]
            x = mv[max(0, j - p.window_days + 1) : j + 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                if p.kind == "vol":
                    val = np.nanstd(y, axis=0, ddof=1) * np.sqrt(252)
                else:
                    ok = ~np.isnan(y) & ~np.isnan(x)[:, None]
                    xm = np.where(ok, x[:, None], np.nan)
                    ym = np.where(ok, y, np.nan)
                    xbar, ybar = np.nanmean(xm, axis=0), np.nanmean(ym, axis=0)
                    val = np.nansum((xm - xbar) * (ym - ybar), axis=0) / np.nansum(
                        (xm - xbar) ** 2, axis=0
                    )
            n = (~np.isnan(y)).sum(axis=0)
            val = np.where(n >= p.min_days, val, np.nan)
            out.iloc[row] = val
        return out


@register("factor", "price_momentum")
class PriceMomentum(Factor):
    """12-1 month total-return momentum (Jegadeesh & Titman 1993) at month-ends.
    Thesis: under-reaction to firm news. Expected sign: +. Known risk: crashes in
    sharp market rebounds (Daniel & Moskowitz 2016)."""

    hypothesis_id: ClassVar[str] = "S-MOM"
    precomputable: ClassVar[bool] = True

    class Params(ComponentParams):
        label: str = "mom_12_1"
        lookback_months: int = 12
        skip_months: int = 1

    params: Params

    def compute(self, history: MarketData) -> pd.DataFrame:
        p = self.params
        idx = _month_end_index(history)
        idx = idx[idx <= history.end]
        tri = history.total_return_index.reindex(idx)
        return tri.shift(p.skip_months) / tri.shift(p.lookback_months) - 1.0
