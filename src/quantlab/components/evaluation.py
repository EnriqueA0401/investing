"""Evaluators. Deflated Sharpe / PBO / CPCV statistics arrive in Phase 5."""

from __future__ import annotations

from pydantic import Field

from quantlab import metrics as m
from quantlab.contracts import BacktestResult, ComponentParams, Evaluator
from quantlab.registry import register


@register("evaluator", "standard")
class StandardEvaluator(Evaluator):
    """Net-of-cost performance, cost drag, turnover and benchmark-relative stats."""

    class Params(ComponentParams):
        risk_free_annual: float = Field(
            default=0.0, description="Placeholder until a T-bill series is wired in (Phase 2)."
        )

    params: Params

    def evaluate(
        self, result: BacktestResult, benchmark: BacktestResult | None = None
    ) -> dict[str, float]:
        net, gross = result.net_returns, result.gross_returns
        rf_d = (1 + self.params.risk_free_annual) ** (1 / m.PERIODS_PER_YEAR) - 1
        years = len(net) / m.PERIODS_PER_YEAR
        out = {
            "n_days": float(len(net)),
            "cagr": m.annualized_return(net),
            "vol": m.annualized_vol(net),
            "sharpe": m.sharpe_ratio(net, rf_d),
            "sortino": m.sortino_ratio(net, rf_d),
            "max_drawdown": m.max_drawdown(net),
            "calmar": m.calmar_ratio(net),
            "hit_rate_monthly": m.monthly_hit_rate(net),
            "skew": m.as_float(net.skew()),
            "excess_kurtosis": m.as_float(net.kurt()),
            "gross_cagr": m.annualized_return(gross),
            "cost_drag_annual": float(result.costs.sum() / years),
            "turnover_annual": float(result.turnover.sum() / years),
        }
        if benchmark is not None:
            common = net.index.intersection(benchmark.net_returns.index)
            s, b = net.loc[common], benchmark.net_returns.loc[common]
            out |= {
                "bench_cagr": m.annualized_return(b),
                "bench_sharpe": m.sharpe_ratio(b, rf_d),
                "bench_max_drawdown": m.max_drawdown(b),
                "excess_cagr": m.annualized_return(s) - m.annualized_return(b),
                "tracking_error": m.tracking_error(s, b),
                "information_ratio": m.information_ratio(s, b),
                "beta_to_bench": m.beta(s, b),
            }
        return out
