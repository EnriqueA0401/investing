"""Long/short equity: neutralized signals, constructors, risk model and overlays."""

from __future__ import annotations

from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import Field

from quantlab.contracts import (
    ComponentParams,
    CostContext,
    CostModel,
    DecisionContext,
    PortfolioConstructor,
    RiskManager,
    SignalGenerator,
)
from quantlab.logging_setup import get_logger
from quantlab.registry import register

log = get_logger(__name__)


# ------------------------------------------------------------------ signals


def _feature_row(ctx: DecisionContext, label: str) -> pd.Series:
    panel = ctx.features[label]
    if ctx.as_of not in panel.index:
        return pd.Series(np.nan, index=ctx.investable)
    return cast(pd.Series, panel.loc[ctx.as_of]).reindex(ctx.investable).astype(float)


def neutralize(
    x: pd.Series,
    sector: pd.Series,
    log_cap: pd.Series | None,
    winsor: float = 0.01,
) -> pd.Series:
    """Winsorize, demean within sector, scale to unit cross-sectional std, then
    remove the linear size exposure (residual on log market cap)."""
    x = x.dropna()
    if len(x) < 10:
        return pd.Series(np.nan, index=x.index)
    lo, hi = float(x.quantile(winsor)), float(x.quantile(1 - winsor))
    x = x.clip(lower=lo, upper=hi)
    x = x - x.groupby(sector.reindex(x.index).fillna("Other")).transform("mean")
    sd = float(x.std(ddof=0))
    x = x / sd if sd > 0 else x * 0.0
    if log_cap is not None:
        lc = log_cap.reindex(x.index)
        ok = lc.notna()
        if int(ok.sum()) > 10:
            a = np.column_stack([np.ones(int(ok.sum())), lc[ok].to_numpy()])
            beta, *_ = np.linalg.lstsq(a, x[ok].to_numpy(), rcond=None)
            x.loc[ok] = x[ok] - a @ beta
    return x


def _sectors(ctx: DecisionContext, names: list[str]) -> pd.Series:
    return pd.Series({t: ctx.history.asset_class.get(t, "Other") for t in names}, dtype=object)


def composite_scores(
    ctx: DecisionContext,
    features: dict[str, float],
    min_features: int,
    size_neutral: bool,
) -> pd.Series:
    sector = _sectors(ctx, ctx.investable)
    log_cap = None
    if size_neutral and "market_cap" in ctx.history.fundamentals:
        cap = ctx.history.fundamentals["market_cap"]
        if ctx.as_of in cap.index:
            row = cast(pd.Series, cap.loc[ctx.as_of]).reindex(ctx.investable)
            log_cap = np.log(row.where(row > 0))
    parts = {f: neutralize(_feature_row(ctx, f), sector, log_cap) for f in features}
    z = pd.DataFrame(parts).reindex(ctx.investable)
    w = pd.Series(features, dtype=float)
    avail = z.notna()
    # Weighted average over the features available for each name: sum(w*z) / sum(|w|).
    num = (z.fillna(0.0) * w).sum(axis=1)
    den = (avail * w.abs()).sum(axis=1)
    score = num / den.where(den > 0)
    return score.where(avail.sum(axis=1) >= min_features)


@register("signal", "neutral_composite")
class NeutralComposite(SignalGenerator):
    """Equal-weight average of sector- and size-neutralized z-scores (signed)."""

    class Params(ComponentParams):
        features: dict[str, float]
        min_features: int = Field(default=1, ge=1)
        size_neutral: bool = True

    params: Params

    def generate(self, ctx: DecisionContext) -> pd.Series:
        p = self.params
        return composite_scores(ctx, p.features, p.min_features, p.size_neutral).dropna()


@register("signal", "ml_rank")
class MLRank(SignalGenerator):
    """Walk-forward model on neutralized features (S-ML).

    Training rows are past month-ends ``s`` (features neutralized as in the
    composite); label = cross-sectional percentile of the return from ``s`` to
    the next month-end, both <= ``as_of`` (purging is structural). Retrains every
    ``retrain_months``; before ``min_train_months`` falls back to the composite.
    Hyper-parameters are fixed ex-ante (no tuning).
    """

    class Params(ComponentParams):
        features: dict[str, float]
        model: Literal["gbm", "ridge"] = "gbm"
        min_train_months: int = Field(default=36, ge=12)
        retrain_months: int = Field(default=12, ge=1)
        min_features: int = 3
        seed: int = 0

    params: Params

    def __init__(self, params: ComponentParams) -> None:
        super().__init__(params)
        self._model: Any = None
        self._trained_at: pd.Timestamp | None = None

    def _x_at(self, ctx: DecisionContext, when: pd.Timestamp, names: list[str]) -> pd.DataFrame:
        """Neutralized feature matrix at ``when`` (<= as_of) for ``names``."""
        sector = _sectors(ctx, names)
        cols = {}
        for f, sign in self.params.features.items():
            panel = ctx.features[f]
            row = (
                cast(pd.Series, panel.loc[when]).reindex(names).astype(float)
                if when in panel.index
                else pd.Series(np.nan, index=names)
            )
            cols[f] = sign * neutralize(row, sector, None)
        x = pd.DataFrame(cols).reindex(names)
        return x[x.notna().sum(axis=1) >= self.params.min_features].fillna(0.0)

    def _xy_at(
        self, ctx: DecisionContext, s: pd.Timestamp, s_next: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.Series]:
        hist = ctx.history
        names = list(ctx.investable)
        if hist.universe is not None:
            past = hist.universe.loc[hist.universe.index <= s]
            if len(past):
                row = past.iloc[-1]
                names = [str(n) for n in row[row.astype(bool)].index]
        names = [n for n in names if n in hist.total_return_index.columns]
        x = self._x_at(ctx, s, names)
        tri = hist.total_return_index
        fwd = (tri.loc[s_next] / tri.loc[s] - 1.0).reindex(x.index).dropna()
        y = cast(pd.Series, fwd.rank(pct=True) - 0.5)
        return x.loc[fwd.index], y

    def _train(self, ctx: DecisionContext) -> None:
        idx = ctx.features[next(iter(self.params.features))].index
        months = [pd.Timestamp(d) for d in idx if d <= ctx.as_of]
        xs, ys = [], []
        for s, s_next in pairwise(months):
            if s_next > ctx.as_of:
                raise AssertionError("label ends after as_of")
            x, y = self._xy_at(ctx, s, s_next)
            if len(y) >= 50:
                xs.append(x)
                ys.append(y)
        if len(xs) < self.params.min_train_months:
            self._model = None
            return
        x_all, y_all = pd.concat(xs).to_numpy(), pd.concat(ys).to_numpy()
        model: Any
        if self.params.model == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

            model = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_depth=3,
                l2_regularization=1.0,
                min_samples_leaf=200,
                random_state=self.params.seed,
            )
        else:
            from sklearn.linear_model import Ridge  # noqa: PLC0415

            model = Ridge(alpha=1.0, fit_intercept=False)
        self._model = model.fit(x_all, y_all)
        self._trained_at = ctx.as_of

    def generate(self, ctx: DecisionContext) -> pd.Series:
        p = self.params
        last = self._trained_at
        months_since = (
            None
            if last is None
            else (ctx.as_of.year - last.year) * 12 + ctx.as_of.month - last.month
        )
        if months_since is None or months_since < 0 or months_since >= p.retrain_months:
            self._train(ctx)
        if self._model is None:
            return composite_scores(ctx, p.features, p.min_features, size_neutral=True).dropna()
        x = self._x_at(ctx, ctx.as_of, list(ctx.investable))
        return pd.Series(self._model.predict(x.to_numpy()), index=x.index)


# ------------------------------------------------------------ construction


def _beta_row(ctx: DecisionContext, label: str, names: pd.Index) -> pd.Series:
    if label not in ctx.features or ctx.as_of not in ctx.features[label].index:
        return pd.Series(1.0, index=names)
    b = cast(pd.Series, ctx.features[label].loc[ctx.as_of]).reindex(names)
    return b.fillna(b.median()).clip(lower=0.2, upper=3.0)


def beta_neutralize(w: pd.Series, beta: pd.Series, gross_cap: float) -> pd.Series:
    """Rescale one side so long beta == short beta, keeping each side's gross <= cap."""
    long_, short = w[w > 0], w[w < 0]
    bl = float((long_ * beta[long_.index]).sum())
    bs = float((-short * beta[short.index]).sum())
    if bl <= 0 or bs <= 0:
        return w
    out = w.copy()
    if bl > bs:
        out[long_.index] *= bs / bl  # shrink the higher-beta side
    else:
        out[short.index] *= bl / bs
    side_gross = max(float(out[out > 0].sum()), float(-out[out < 0].sum()))
    return out * min(1.0, gross_cap / side_gross) if side_gross > 0 else out


def cap_net_exposure(w: pd.Series, max_net: float | None) -> pd.Series:
    """Mandate limits prevail over beta neutrality: if |net| exceeds the cap, shrink
    the larger side until it fits (accepting a residual beta). Only binds when beta
    neutralization would otherwise breach the limit (e.g. low-beta vs high-beta)."""
    if max_net is None:
        return w
    long_g, short_g = float(w[w > 0].sum()), float(-w[w < 0].sum())
    net = long_g - short_g
    out = w.copy()
    if net > max_net and long_g > 0:
        out[out > 0] *= (short_g + max_net) / long_g
    elif -net > max_net and short_g > 0:
        out[out < 0] *= (long_g + max_net) / short_g
    return out


@register("portfolio", "ls_sector_neutral")
class LSSectorNeutral(PortfolioConstructor):
    """Top/bottom ``n`` per side, allocated across sectors in proportion to sector
    size (each sector net ~0), equal weight within side, then the higher-beta
    side is scaled down so the book's ex-ante beta is zero."""

    class Params(ComponentParams):
        n_per_side: int = Field(default=40, ge=5)
        gross_per_side: float = Field(default=1.0, gt=0, le=1.0)
        beta_feature: str = "beta_252"

    params: Params

    def construct(self, scores: pd.Series, ctx: DecisionContext) -> pd.Series:
        p = self.params
        s = scores.dropna()
        if len(s) < 4 * p.n_per_side:
            return pd.Series(dtype=float)
        sector = _sectors(ctx, [str(n) for n in s.index])
        counts = sector.value_counts()
        raw = counts / counts.sum() * p.n_per_side
        k = pd.Series(np.floor(raw.to_numpy()).astype(int), index=raw.index)
        for sec in (raw - k).sort_values(ascending=False).index[: p.n_per_side - int(k.sum())]:
            k[sec] += 1
        longs: list[str] = []
        shorts: list[str] = []
        for sec, k_alloc in k.items():
            names = s[(sector == sec).to_numpy()].sort_values()
            ks = int(min(int(k_alloc), len(names) // 2))
            if ks:
                longs += [str(n) for n in names.index[-ks:]]
                shorts += [str(n) for n in names.index[:ks]]
        if not longs or not shorts:
            return pd.Series(dtype=float)
        w = pd.Series(0.0, index=[*longs, *shorts])
        w[longs] = p.gross_per_side / len(longs)
        w[shorts] = -p.gross_per_side / len(shorts)
        w = beta_neutralize(w, _beta_row(ctx, p.beta_feature, w.index), p.gross_per_side)
        w = cap_net_exposure(w, ctx.constraints.max_net_exposure)
        cap = ctx.constraints.max_weight
        return w[w != 0].clip(lower=-cap, upper=cap)


def pca_risk_model(rets: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Statistical factor model of daily returns: loadings (n x k), factor
    variances (k), specific variances (n); covariance = B diag(f) B' + diag(d)."""
    x = rets.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    t = x.shape[0]
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    k = min(k, len(sv))
    b = vt[:k].T
    fvar = sv[:k] ** 2 / (t - 1)
    resid = x - (u[:, :k] * sv[:k]) @ vt[:k]
    dvar = np.maximum(resid.var(axis=0, ddof=1), 1e-8)
    return b, fvar, dvar


@register("portfolio", "ls_optimizer")
class LSOptimizer(PortfolioConstructor):
    """Mean-variance L/S with a statistical (PCA) risk model, solved with cvxpy.

    max  a'w - tc * ||w - w_cur||_1
    s.t. sum(w) = 0, beta'w = 0, sector nets = 0, |w_i| <= cap, ||w||_1 <= gross,
         w' (B F B' + D) w <= target_vol^2 / 252
    a = IC * sigma_i * z_i (Grinold), z = standardized composite score.
    """

    class Params(ComponentParams):
        information_coefficient: float = Field(default=0.03, gt=0, le=0.2)
        target_vol: float = Field(default=0.10, gt=0)
        gross: float = Field(default=2.0, gt=0)
        n_factors: int = Field(default=15, ge=1)
        window_days: int = Field(default=252, ge=60)
        tc_bps: float = Field(default=10.0, ge=0)
        beta_feature: str = "beta_252"

    params: Params

    def construct(self, scores: pd.Series, ctx: DecisionContext) -> pd.Series:
        import cvxpy as cp  # noqa: PLC0415 - heavy optional import

        p = self.params
        s = scores.dropna()
        names = [str(n) for n in s.index]
        rets = ctx.history.returns()[names].iloc[-p.window_days :]
        keep = rets.notna().mean() >= 0.8
        names = [n for n in names if bool(keep[n])]
        if len(names) < 100:
            return pd.Series(dtype=float)
        b, fvar, dvar = pca_risk_model(rets[names].fillna(0.0), p.n_factors)
        sig = np.sqrt(np.einsum("ik,k,ik->i", b, fvar, b) + dvar) * np.sqrt(252)
        sv = s[names]
        z = ((sv - sv.mean()) / sv.std(ddof=0)).to_numpy()
        alpha = p.information_coefficient * sig * z
        beta = _beta_row(ctx, p.beta_feature, pd.Index(names)).to_numpy()
        sectors = np.array([ctx.history.asset_class.get(n, "Other") for n in names])
        cur = ctx.current_weights.reindex(names).fillna(0.0).to_numpy()
        w = cp.Variable(len(names))
        cap = ctx.constraints.max_weight
        risk = cp.sum_squares(cp.multiply(np.sqrt(fvar), b.T @ w)) + cp.sum_squares(
            cp.multiply(np.sqrt(dvar), w)
        )
        cons = [
            cp.sum(w) == 0,
            beta @ w == 0,
            cp.abs(w) <= cap,
            cp.norm1(w) <= p.gross,
            risk <= p.target_vol**2 / 252,
        ]
        for sec in np.unique(sectors):
            idx = np.flatnonzero(sectors == sec)
            if len(idx) >= 2:
                cons.append(cp.sum(w[idx]) == 0)
        prob = cp.Problem(cp.Maximize(alpha @ w - p.tc_bps / 1e4 * cp.norm1(w - cur)), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            prob.solve(solver=cp.SCS)
        if w.value is None:
            log.warning("optimizer infeasible; flat book", extra={"as_of": str(ctx.as_of.date())})
            return pd.Series(dtype=float)
        out = pd.Series(np.asarray(w.value).ravel(), index=names)
        out = out.where(out.abs() > 1e-5, 0.0).clip(lower=-cap, upper=cap)
        return out[out != 0]


# ---------------------------------------------------------------- overlays


@register("risk", "credit_regime")
class CreditRegime(RiskManager):
    """S-MACRO: scale gross exposure when the Baa-Treasury spread is more than
    ``z_threshold`` standard deviations above its trailing 3-year mean.

    Thesis: factor premia (notably momentum and low-risk) suffer crashes and
    de-leveraging spirals in credit-stress regimes; cutting exposure then lowers
    tail risk more than it costs in expected return. Uses the previous
    session's print (one-day publication lag)."""

    class Params(ComponentParams):
        series: str = "BAA10Y"
        lookback_days: int = Field(default=756, ge=120)
        z_threshold: float = 1.0
        scale: float = Field(default=0.5, ge=0, le=1)

    params: Params

    def adjust(self, weights: pd.Series, ctx: DecisionContext) -> pd.Series:
        p = self.params
        s = ctx.history.extras[p.series].dropna().iloc[:-1]  # previous session only
        window = s.iloc[-p.lookback_days :]
        if len(window) < 120:
            return weights
        z = (float(window.iloc[-1]) - float(window.mean())) / float(window.std(ddof=1))
        return weights * p.scale if z > p.z_threshold else weights


# ------------------------------------------------------------------- costs


@register("cost_model", "equity_liquidity")
class EquityLiquidityCost(CostModel):
    """Half-spread by liquidity bucket + commission + slippage + square-root impact.

    Half-spread (bps): ADV >= $500M -> 1.5; >= $100M -> 3; else 6 (large-cap US
    quotes; conservative versus TAQ estimates)."""

    class Params(ComponentParams):
        tiers_usd: tuple[float, float] = (5e8, 1e8)
        tiers_bps: tuple[float, float, float] = (1.5, 3.0, 6.0)

    params: Params

    def cost(self, trades: pd.Series, ctx: CostContext) -> pd.Series:
        cfg, p = ctx.config, self.params
        size = trades.abs()
        adv = ctx.adv_usd.reindex(size.index)
        hs = np.where(
            adv >= p.tiers_usd[0],
            p.tiers_bps[0],
            np.where(adv >= p.tiers_usd[1], p.tiers_bps[1], p.tiers_bps[2]),
        )
        linear = size * (hs + cfg.commission_bps + cfg.slippage_bps) / 1e4
        if cfg.impact_coefficient == 0:
            return linear
        vol = ctx.daily_vol.reindex(size.index).fillna(float(ctx.daily_vol.median()))
        adv_f = adv.fillna(float(adv.median())).clip(lower=1e6)
        impact = size * cfg.impact_coefficient * vol * np.sqrt(size * ctx.nav_usd / adv_f)
        return linear + impact.fillna(0.0)
