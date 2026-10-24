"""Experiment orchestrator.

The orchestrator is the only place that touches the full dataset. At every
decision date it hands components ``data.asof(t)``, so look-ahead through the
data path is impossible by construction; leakage *within* a component
(e.g. a negative shift) is caught by ``quantlab.testing.assert_causal`` in
that component's unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from quantlab import registry
from quantlab.calendar import rebalance_dates
from quantlab.components.portfolios import FixedWeights, check_constraints
from quantlab.config import ComponentSpec, ExperimentConfig
from quantlab.contracts import (
    Backtester,
    BacktestResult,
    CostModel,
    DataSource,
    DataValidator,
    DecisionContext,
    Evaluator,
    Factor,
    PortfolioConstructor,
    RiskManager,
    Severity,
    SignalGenerator,
    ValidationIssue,
)
from quantlab.data.market_data import MarketData
from quantlab.data.store import SnapshotStore
from quantlab.logging_setup import get_logger
from quantlab.reproducibility import seed_everything
from quantlab.splits import HoldoutGuard
from quantlab.tracking import RunTracker, Stage

log = get_logger(__name__)


BASELINE_KEYS = ("cagr", "vol", "sharpe", "sortino", "max_drawdown", "turnover_annual")


class DataValidationError(RuntimeError):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        lines = "; ".join(f"{i.ticker}:{i.check}:{i.message}" for i in issues)
        super().__init__(f"data validation failed: {lines}")


@dataclass
class Components:
    validators: list[DataValidator]
    factors: list[Factor]
    signal: SignalGenerator
    portfolio: PortfolioConstructor
    risk: list[RiskManager]
    cost_model: CostModel
    backtester: Backtester
    evaluator: Evaluator

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> Components:
        p = config.pipeline
        if p.model is not None:
            raise NotImplementedError("model slot is wired in Phase 3 (needs purged CV)")
        return cls(
            validators=[registry.build("validator", s) for s in p.validators],
            factors=[registry.build("factor", s) for s in p.factors],
            signal=registry.build("signal", p.signal),
            portfolio=registry.build("portfolio", p.portfolio),
            risk=[registry.build("risk", s) for s in p.risk],
            cost_model=registry.build("cost_model", p.cost_model),
            backtester=registry.build("backtester", p.backtester),
            evaluator=registry.build("evaluator", p.evaluator),
        )


@dataclass
class ExperimentResult:
    run_id: str
    metrics: dict[str, float]
    strategy: BacktestResult
    benchmark: BacktestResult
    baseline: BacktestResult
    issues: list[ValidationIssue] = field(default_factory=list)


def load_data(config: ExperimentConfig) -> tuple[MarketData, str]:
    """Load the pinned snapshot, or fetch live and freeze it. Returns (data, snapshot_id)."""
    store = SnapshotStore(config.tracking.snapshot_dir)
    if config.data_snapshot is not None:
        return store.load(config.data_snapshot), config.data_snapshot
    source = cast(DataSource, registry.build("data_source", config.pipeline.data_source))
    data = source.fetch(config.universe.assets, config.split.data_start, config.split.holdout_end)
    snap_id = store.save(data, {"source": config.pipeline.data_source.model_dump(mode="json")})
    log.info("data frozen to snapshot", extra={"snapshot_id": snap_id})
    return data, snap_id


def validate_data(data: MarketData, validators: list[DataValidator]) -> list[ValidationIssue]:
    issues = [i for v in validators for i in v.validate(data)]
    for issue in issues:
        log.log(
            40 if issue.severity is Severity.ERROR else 30,
            "data issue",
            extra={"check": issue.check, "ticker": issue.ticker, "detail": issue.message},
        )
    errors = [i for i in issues if i.severity is Severity.ERROR]
    if errors:
        raise DataValidationError(errors)
    return issues


def generate_targets(
    data: MarketData,
    config: ExperimentConfig,
    *,
    signal: SignalGenerator,
    portfolio: PortfolioConstructor,
    factors: list[Factor],
    risk: list[RiskManager],
    rng: np.random.Generator,
    start: pd.Timestamp,
    respect_universe: bool = True,
) -> dict[pd.Timestamp, pd.Series]:
    """Walk the rebalance schedule, deciding each target from ``data.asof(t)`` only.

    ``respect_universe=False`` (reference portfolios such as the benchmark) skips
    tradability, membership and liquidity screens."""
    schedule = rebalance_dates(data.dates, config.execution.rebalance_frequency)
    schedule = schedule[schedule >= start]
    targets: dict[pd.Timestamp, pd.Series] = {}
    prev = pd.Series(dtype=float)
    prev_t: pd.Timestamp | None = None
    cache = {f.label: f.compute(data) for f in factors if f.precomputable}
    for ts in schedule:
        t = pd.Timestamp(ts)
        hist = data.asof(t)
        investable = (
            investable_names(hist, config)
            if respect_universe
            else hist.investable(config.universe.min_history_days)
        )
        features: dict[str, pd.DataFrame] = {}
        for f in factors:
            panel = cache[f.label].loc[:t] if f.label in cache else f.compute(hist)
            if panel.index.max() > t:
                raise RuntimeError(f"factor {f.label} emitted rows after {t.date()}")
            if f.label in features:
                raise ValueError(f"duplicate factor label '{f.label}'")
            features[f.label] = panel
        ctx = DecisionContext(
            as_of=t,
            history=hist,
            investable=investable,
            previous_weights=prev,
            constraints=config.constraints,
            rng=rng,
            features=features,
            current_weights=drift_weights(prev, hist, prev_t),
        )
        weights = portfolio.construct(signal.generate(ctx), ctx)
        for rm in risk:
            weights = rm.adjust(weights, ctx)
        weights = weights[weights.abs() > 1e-12].astype(float)
        check_constraints(weights, config.constraints, investable)
        targets[t] = weights
        prev, prev_t = weights, t
    return targets


def investable_names(hist: MarketData, config: ExperimentConfig) -> list[str]:
    """Names a strategy may hold at ``hist.end`` (all inputs observable by then)."""
    u = config.universe
    names = hist.investable(u.min_history_days)
    if u.source == "data":
        blocked = {a.ticker for a in u.assets if not a.tradable}
        names = [n for n in names if n not in blocked]
    else:
        allowed = set(u.tradable_tickers)
        names = [n for n in names if n in allowed]
    if u.use_membership:
        if hist.universe is None or hist.universe.empty:
            return []
        row = hist.universe.iloc[-1]
        names = [n for n in names if bool(row.get(n, False))]
    if u.exclude_asset_classes:
        names = [n for n in names if hist.asset_class.get(n) not in u.exclude_asset_classes]
    if u.min_price > 0 or u.min_adv_usd > 0:
        px = hist.close[names].iloc[-1]
        adv = (hist.close[names] * hist.volume[names]).iloc[-63:].mean()
        ok = (px >= u.min_price) & (adv >= u.min_adv_usd)
        names = [n for n in names if bool(ok.get(n, False))]
    return names


def drift_weights(weights: pd.Series, hist: MarketData, since: pd.Timestamp | None) -> pd.Series:
    """Weights set at ``since`` drifted with total returns (and cash) to ``hist.end``.

    Approximation used for turnover control: ignores the one-session fill lag.
    """
    if since is None or weights.empty:
        return pd.Series(dtype=float)
    tri = hist.total_return_index
    growth = (tri.iloc[-1] / tri.loc[since]).reindex(weights.index).fillna(1.0)
    cash_idx = (1.0 + hist.cash()).cumprod()
    cash_growth = float(cash_idx.iloc[-1] / cash_idx.loc[since])
    risky: pd.Series = weights * growth
    total = float(risky.sum()) + (1.0 - float(weights.sum())) * cash_growth
    return risky / total


def run_experiment(
    config: ExperimentConfig,
    stage: Stage = "research",
    data: MarketData | None = None,
    unlock_reason: str | None = None,
) -> ExperimentResult:
    rng = seed_everything(config.seed)
    tracker = RunTracker(config)
    guard = HoldoutGuard(config, tracker.ledger)
    if stage == "final" and unlock_reason:
        guard.unlock(unlock_reason)

    with tracker.start(stage) as run:
        if data is None:
            data, snap_id = load_data(config)
            run.set_tag("data_snapshot_id", snap_id)
        comps = Components.from_config(config)
        run.set_tag("holdout_unlocks_total", len(tracker.ledger.holdout_unlocks()))

        view = guard.view(data, stage)
        # Validate only what this stage may see: issue reports must not leak holdout facts.
        issues = validate_data(view, comps.validators)
        run.set_tag("n_data_warnings", len(issues))
        start, _ = guard.evaluation_window(stage)
        start = pd.Timestamp(start)

        targets = generate_targets(
            view,
            config,
            signal=comps.signal,
            portfolio=comps.portfolio,
            factors=comps.factors,
            risk=comps.risk,
            rng=rng,
            start=start,
        )
        strategy = comps.backtester.run(targets, view, comps.cost_model, config, start)

        bench_pf = FixedWeights.from_params({"weights": config.benchmark.weights})
        # The benchmark is a policy portfolio, not bound by the strategy's position caps.
        bench_cfg = config.model_copy(
            update={
                "execution": config.execution.model_copy(
                    update={"rebalance_frequency": config.benchmark.rebalance_frequency}
                ),
                "constraints": config.constraints.model_copy(update={"max_weight": 1.0}),
            }
        )
        bench_targets = generate_targets(
            view,
            bench_cfg,
            signal=registry.build("signal", ComponentSpec(name="uniform")),
            portfolio=bench_pf,
            factors=[],
            risk=[],
            rng=rng,
            start=start,
            respect_universe=False,
        )
        benchmark = comps.backtester.run(bench_targets, view, comps.cost_model, bench_cfg, start)

        base_targets = generate_targets(
            view,
            config,
            signal=registry.build("signal", ComponentSpec(name="uniform")),
            portfolio=registry.build("portfolio", config.baseline),
            factors=[],
            risk=[],
            rng=rng,
            start=start,
        )
        baseline = comps.backtester.run(base_targets, view, comps.cost_model, config, start)

        metrics = comps.evaluator.evaluate(strategy, benchmark)
        base_metrics = comps.evaluator.evaluate(baseline, benchmark)
        metrics |= {f"base_{k}": base_metrics[k] for k in BASELINE_KEYS}
        run.log_metrics(metrics)
        run.set_tag(
            "n_trials_to_date",
            len(tracker.ledger.trial_hashes(stage=stage) | {config.config_hash()}),
        )
        run.log_frame("net_returns", strategy.net_returns)
        run.log_frame("weights", strategy.weights)
        run.log_frame("benchmark_net_returns", benchmark.net_returns)
        run.log_frame("baseline_net_returns", baseline.net_returns)
        _log_summary(metrics)

    return ExperimentResult(run.run_id, metrics, strategy, benchmark, baseline, issues)


def _log_summary(metrics: dict[str, Any]) -> None:
    keys = (
        "cagr",
        "vol",
        "sharpe",
        "max_drawdown",
        "turnover_annual",
        "bench_sharpe",
        "base_sharpe",
    )
    log.info("metrics", extra={k: round(metrics[k], 4) for k in keys if k in metrics})
