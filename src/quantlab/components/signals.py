"""Signal generators. Scores are indexed by investable tickers; NaN/absent = not selected."""

from __future__ import annotations

from itertools import pairwise
from typing import cast

import numpy as np
import pandas as pd
from pydantic import Field

from quantlab.calendar import rebalance_dates
from quantlab.contracts import ComponentParams, DecisionContext, SignalGenerator
from quantlab.registry import register


def _row(ctx: DecisionContext, label: str) -> pd.Series:
    if label not in ctx.features:
        raise KeyError(f"feature '{label}' not computed; add it to pipeline.factors")
    panel = ctx.features[label]
    # .loc[as_of], not iloc[-1]: correct even if a panel were handed rows past as_of.
    row = cast(pd.Series, panel.loc[ctx.as_of])
    return row.reindex(ctx.investable).astype(float)


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def _top(scores: pd.Series, n: int) -> pd.Series:
    """Top-``n`` names (ties broken by ticker for determinism), returned as score 1.0."""
    s = scores.dropna()
    ranked = s.sort_index().sort_values(ascending=False, kind="stable").index[:n]
    return pd.Series(1.0, index=ranked)


@register("signal", "uniform")
class Uniform(SignalGenerator):
    """No view: every investable asset scores 1. The 'dumb' baseline signal."""

    def generate(self, ctx: DecisionContext) -> pd.Series:
        return pd.Series(1.0, index=ctx.investable, dtype=float)


@register("signal", "feature_score")
class FeatureScore(SignalGenerator):
    """Raw value of one feature at ``as_of`` (for constructors that use magnitudes)."""

    class Params(ComponentParams):
        feature: str

    params: Params

    def generate(self, ctx: DecisionContext) -> pd.Series:
        return _row(ctx, self.params.feature).dropna()


@register("signal", "top_n")
class TopN(SignalGenerator):
    """Select the ``n`` investable assets with the highest value of one feature."""

    class Params(ComponentParams):
        feature: str
        n: int = Field(default=6, ge=1)

    params: Params

    def generate(self, ctx: DecisionContext) -> pd.Series:
        return _top(_row(ctx, self.params.feature), self.params.n)


@register("signal", "zscore_combo_top_n")
class ZScoreComboTopN(SignalGenerator):
    """Equal-weight average of cross-sectional z-scores (with theoretical signs), top ``n``."""

    class Params(ComponentParams):
        features: dict[str, float] = Field(description="feature label -> sign (+1 / -1)")
        n: int = Field(default=6, ge=1)

    params: Params

    def combo(self, ctx: DecisionContext) -> pd.Series:
        parts = [sign * _zscore(_row(ctx, f)) for f, sign in self.params.features.items()]
        return pd.concat(parts, axis=1).mean(axis=1, skipna=False)

    def generate(self, ctx: DecisionContext) -> pd.Series:
        return _top(self.combo(ctx), self.params.n)


@register("signal", "ridge_top_n")
class RidgeTopN(SignalGenerator):
    """Walk-forward ridge on cross-sectional z-scored features (H-006).

    Training samples are past month-ends ``s`` with label = return from ``s`` to
    the next month-end minus the cross-sectional mean. Only labels that end on
    or before ``as_of`` are used (purging is structural: the history ends at
    ``as_of``). Before ``min_train_months`` the z-score combo is used instead.
    """

    class Params(ComponentParams):
        features: dict[str, float]
        n: int = Field(default=6, ge=1)
        alpha: float = Field(default=1.0, gt=0)
        min_train_months: int = Field(default=36, ge=12)

    params: Params

    def training_set(self, ctx: DecisionContext) -> tuple[np.ndarray, np.ndarray, int]:
        tri = ctx.history.total_return_index
        dates = [*rebalance_dates(ctx.history.dates, "ME"), ctx.as_of]
        xs, ys, months = [], [], 0
        for s, s_next in pairwise(dates):
            if s_next > ctx.as_of:  # defensive: never use a label ending after as_of
                raise AssertionError("label ends after as_of")
            feats = pd.concat(
                {f: ctx.features[f].loc[s] for f in self.params.features}, axis=1
            ).dropna()
            fwd = (tri.loc[s_next] / tri.loc[s] - 1.0).reindex(feats.index).dropna()
            feats = feats.loc[fwd.index]
            if len(feats) < 3:
                continue
            xs.append(feats.apply(_zscore).to_numpy())
            ys.append((fwd - fwd.mean()).to_numpy())
            months += 1
        if not xs:
            return np.empty((0, len(self.params.features))), np.empty(0), 0
        return np.vstack(xs), np.concatenate(ys), months

    def predict(self, ctx: DecisionContext) -> pd.Series:
        x, y, months = self.training_set(ctx)
        if months < self.params.min_train_months:
            return ZScoreComboTopN(
                ZScoreComboTopN.Params(features=self.params.features, n=self.params.n)
            ).combo(ctx)
        k = x.shape[1]
        beta = np.linalg.solve(x.T @ x + self.params.alpha * np.eye(k), x.T @ y)
        now = pd.concat({f: _row(ctx, f) for f in self.params.features}, axis=1).dropna()
        return pd.Series(now.apply(_zscore).to_numpy() @ beta, index=now.index)

    def generate(self, ctx: DecisionContext) -> pd.Series:
        return _top(self.predict(ctx), self.params.n)
