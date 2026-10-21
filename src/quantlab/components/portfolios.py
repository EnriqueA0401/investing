"""Portfolio constructors and the always-on constraint check."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from quantlab.config import PortfolioConstraints
from quantlab.contracts import ComponentParams, DecisionContext, PortfolioConstructor
from quantlab.registry import register
from quantlab.risk_model import cap_and_redistribute, ledoit_wolf_cov, trailing_returns

_TOL = 1e-9


class ConstraintViolationError(ValueError):
    pass


def check_constraints(
    weights: pd.Series, constraints: PortfolioConstraints, investable: list[str]
) -> None:
    """Raise if final target weights breach the mandate. Called by the orchestrator."""
    if weights.isna().any():
        raise ConstraintViolationError("NaN weights")
    if constraints.long_only and (weights < -_TOL).any():
        raise ConstraintViolationError(
            f"short positions in long-only mandate: {weights[weights < 0]}"
        )
    if weights.abs().sum() > constraints.max_gross_leverage + _TOL:
        raise ConstraintViolationError(f"gross leverage {weights.abs().sum():.4f} above limit")
    if (weights.abs() > constraints.max_weight + _TOL).any():
        raise ConstraintViolationError(f"position above max_weight: {weights.abs().max():.4f}")
    held = weights[weights.abs() > _TOL].index
    not_investable = set(held) - set(investable)
    if not_investable:
        raise ConstraintViolationError(
            f"weights on non-investable assets: {sorted(not_investable)}"
        )


@register("portfolio", "equal_weight")
class EqualWeight(PortfolioConstructor):
    """1/N over assets with a valid score, capped at ``max_weight`` (excess stays in cash)."""

    def construct(self, scores: pd.Series, ctx: DecisionContext) -> pd.Series:
        names = scores.dropna().index
        if len(names) == 0:
            return pd.Series(dtype=float)
        w = min(1.0 / len(names), ctx.constraints.max_weight)
        return pd.Series(w, index=names, dtype=float)


@register("portfolio", "fixed_weights")
class FixedWeights(PortfolioConstructor):
    """Static policy weights (benchmarks). Weights of not-yet-investable assets are
    redistributed pro-rata over the investable ones."""

    class Params(ComponentParams):
        weights: dict[str, float]

    params: Params

    def construct(self, scores: pd.Series, ctx: DecisionContext) -> pd.Series:
        w = pd.Series(self.params.weights, dtype=float)
        w = w[w.index.isin(ctx.investable)]
        if w.empty:
            return w
        return w * (sum(self.params.weights.values()) / w.sum())


# ---------------------------------------------------------------- risk-based


class _RiskBased(PortfolioConstructor):
    """Shared plumbing: names with a valid score, Ledoit-Wolf covariance, 20% cap."""

    class Params(ComponentParams):
        cov_window_days: int = Field(default=252, ge=20)

    params: Params

    def _cov(self, names: list[str], ctx: DecisionContext) -> pd.DataFrame:
        rets = trailing_returns(ctx.history, names, self.params.cov_window_days)
        return ledoit_wolf_cov(rets) * 252  # annualised

    @abstractmethod
    def raw_weights(self, cov: pd.DataFrame, ctx: DecisionContext) -> pd.Series: ...

    def construct(self, scores: pd.Series, ctx: DecisionContext) -> pd.Series:
        names = [str(n) for n in scores.dropna().index]
        if not names:
            return pd.Series(dtype=float)
        if len(names) == 1:
            return pd.Series(min(1.0, ctx.constraints.max_weight), index=names)
        raw = self.raw_weights(self._cov(names, ctx), ctx)
        return cap_and_redistribute(raw, ctx.constraints.max_weight)


@register("portfolio", "inverse_vol")
class InverseVol(_RiskBased):
    """w_i ∝ 1/σ_i, σ from a shorter window (default 63d) than the covariance."""

    class Params(_RiskBased.Params):
        vol_window_days: int = Field(default=63, ge=2)

    params: Params

    def raw_weights(self, cov: pd.DataFrame, ctx: DecisionContext) -> pd.Series:
        names = [str(c) for c in cov.columns]
        rets = trailing_returns(ctx.history, names, self.params.vol_window_days, min_rows=20)
        inv = 1.0 / rets.std(ddof=1)
        return inv / inv.sum()


@register("portfolio", "erc")
class EqualRiskContribution(_RiskBased):
    """Equal risk contribution via Spinu's convex program:
    min ½ y'Σy − Σ b_i ln y_i, y > 0, then w = y / Σy."""

    def raw_weights(self, cov: pd.DataFrame, ctx: DecisionContext) -> pd.Series:
        sigma = cov.to_numpy()
        n = len(sigma)
        b = np.full(n, 1.0 / n)

        def f(y: np.ndarray) -> tuple[float, np.ndarray]:
            sy = sigma @ y
            return float(0.5 * y @ sy - b @ np.log(y)), sy - b / y

        y0 = 1.0 / np.sqrt(np.diag(sigma))
        res = minimize(f, y0, jac=True, method="L-BFGS-B", bounds=[(1e-10, None)] * n)
        if not res.success:
            raise RuntimeError(f"ERC optimisation failed: {res.message}")
        y = np.asarray(res.x)
        return pd.Series(y / y.sum(), index=cov.index)


@register("portfolio", "hrp")
class HierarchicalRiskParity(_RiskBased):
    """López de Prado (2016): single-linkage tree on correlation distance,
    quasi-diagonal ordering, recursive bisection with inverse-variance splits."""

    def raw_weights(self, cov: pd.DataFrame, ctx: DecisionContext) -> pd.Series:
        vols = np.sqrt(np.diag(cov.to_numpy()))
        corr = cov.to_numpy() / np.outer(vols, vols)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        np.fill_diagonal(dist, 0.0)
        order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
        w = pd.Series(1.0, index=cov.index)
        clusters = [[cov.index[i] for i in order]]
        while clusters:
            nxt = []
            for c in clusters:
                if len(c) < 2:
                    continue
                left, right = c[: len(c) // 2], c[len(c) // 2 :]
                v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
                alpha = 1.0 - v_l / (v_l + v_r)
                w[left] *= alpha
                w[right] *= 1.0 - alpha
                nxt += [left, right]
            clusters = nxt
        return w / w.sum()


def _cluster_var(cov: pd.DataFrame, names: list[str]) -> float:
    sub = cov.loc[names, names].to_numpy()
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def _solve_long_only(
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]], n: int, cap: float
) -> np.ndarray:
    """Minimise a smooth objective over {0 <= w <= cap, sum w = min(1, n*cap)}."""
    budget = min(1.0, n * cap)
    res = minimize(
        objective,
        np.full(n, budget / n),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - budget, "jac": lambda w: np.ones(n)}
        ],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"long-only optimisation failed: {res.message}")
    return np.clip(np.asarray(res.x), 0.0, cap)


@register("portfolio", "min_variance")
class MinVariance(_RiskBased):
    """Long-only minimum variance with the position cap enforced inside the optimiser."""

    def raw_weights(self, cov: pd.DataFrame, ctx: DecisionContext) -> pd.Series:
        sigma = cov.to_numpy()

        def f(w: np.ndarray) -> tuple[float, np.ndarray]:
            return float(w @ sigma @ w), 2.0 * sigma @ w

        w = _solve_long_only(f, len(sigma), ctx.constraints.max_weight)
        return pd.Series(w, index=cov.index)

    def construct(self, scores: pd.Series, ctx: DecisionContext) -> pd.Series:
        names = [str(n) for n in scores.dropna().index]
        if len(names) < 2:
            return super().construct(scores, ctx)
        return self.raw_weights(self._cov(names, ctx), ctx)


@register("portfolio", "robust_mean_variance")
class RobustMeanVariance(_RiskBased):
    """Mean-variance with heavily shrunk alphas (Grinold): α_i = IC · σ_i · z_i.

    ``scores`` are raw signal values; they are z-scored cross-sectionally, so
    only their ranking/dispersion matters. Maximises α'w − (λ/2) w'Σw subject to
    the long-only budget and position cap.
    """

    class Params(_RiskBased.Params):
        information_coefficient: float = Field(default=0.05, gt=0, le=0.2)
        risk_aversion: float = Field(default=10.0, gt=0)

    params: Params

    def raw_weights(self, cov: pd.DataFrame, ctx: DecisionContext) -> pd.Series:
        raise NotImplementedError  # needs scores; see construct

    def construct(self, scores: pd.Series, ctx: DecisionContext) -> pd.Series:
        s = scores.dropna()
        names = [str(n) for n in s.index]
        if len(names) < 2:
            return super().construct(scores, ctx)
        cov = self._cov(names, ctx)
        sigma = cov.to_numpy()
        vol = np.sqrt(np.diag(sigma))
        z = (
            ((s - s.mean()) / s.std(ddof=0)).fillna(0.0).to_numpy()
            if s.std(ddof=0) > 0
            else 0 * vol
        )
        alpha = self.params.information_coefficient * vol * z
        lam = self.params.risk_aversion

        def f(w: np.ndarray) -> tuple[float, np.ndarray]:
            sw = sigma @ w
            return float(-(alpha @ w) + 0.5 * lam * w @ sw), -alpha + lam * sw

        w = _solve_long_only(f, len(names), ctx.constraints.max_weight)
        return pd.Series(w, index=names)
