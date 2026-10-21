"""Signal generators."""

from __future__ import annotations

import pandas as pd

from quantlab.contracts import DecisionContext, SignalGenerator
from quantlab.registry import register


@register("signal", "uniform")
class Uniform(SignalGenerator):
    """No view: every investable asset scores 1. The 'dumb' baseline signal."""

    def generate(self, ctx: DecisionContext) -> pd.Series:
        return pd.Series(1.0, index=ctx.investable, dtype=float)
