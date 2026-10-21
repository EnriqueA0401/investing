"""Portfolio constructors and the always-on constraint check."""

from __future__ import annotations

import pandas as pd

from quantlab.config import PortfolioConstraints
from quantlab.contracts import ComponentParams, DecisionContext, PortfolioConstructor
from quantlab.registry import register

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
