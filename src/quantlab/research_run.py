"""End-to-end research steps after the study: robustness (research window), the
single holdout evaluation, and the current proposed portfolio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from quantlab import registry, risk, stats
from quantlab.config import ExperimentConfig
from quantlab.contracts import DecisionContext, Factor, PortfolioConstructor, SignalGenerator
from quantlab.data.market_data import MarketData
from quantlab.logging_setup import get_logger
from quantlab.ml_diagnostic import build_panel, cpcv_ic
from quantlab.pipeline import ExperimentResult, load_data, run_experiment
from quantlab.robustness import random_filter_null, run_variants, standard_variants
from quantlab.study import Study

log = get_logger(__name__)

ROBUSTNESS_DIR = "robustness"
HOLDOUT_DIR = "holdout"


def _metrics_frame(results: dict[str, ExperimentResult]) -> pd.DataFrame:
    df = pd.DataFrame({k: r.metrics for k, r in results.items()}).T
    df.index.name = "variant"
    return df


def risk_pack(
    streams: dict[str, pd.Series], data: MarketData, out: Path, prefix: str
) -> dict[str, Any]:
    """Tail risk, drawdowns, stress, factor exposures and regimes for return streams."""
    out.mkdir(parents=True, exist_ok=True)
    cash = data.cash()
    pack: dict[str, Any] = {}
    for name, r in streams.items():
        pack[f"var_{name}"] = risk.var_cvar(r)
        pack[f"var_monthly_{name}"] = risk.var_cvar(cast(pd.Series, stats.to_monthly(r)))
        pack[f"drawdowns_{name}"] = risk.drawdown_table(r)
    pack["historical_stress"] = risk.historical_stress(streams)
    factors = risk.factor_returns(data)
    for name, r in streams.items():
        ex = stats.to_monthly(r - cash.reindex(r.index))
        pack[f"exposures_{name}"] = risk.factor_exposures(cast(pd.Series, ex), factors)
    monthly = pd.DataFrame({k: stats.to_monthly(v) for k, v in streams.items()})
    pack["regimes"] = risk.performance_by_regime(monthly, risk.regime_labels(data))
    for key, frame in pack.items():
        frame.to_csv(out / f"{prefix}{key}.csv", float_format="%.6g")
    return pack


def run_robustness(study: Study, selected: str, out_root: Path) -> dict[str, Any]:
    out = out_root / ROBUSTNESS_DIR
    out.mkdir(parents=True, exist_ok=True)
    data, _ = load_data(study.configs[selected])
    research = data.asof(pd.Timestamp(study.configs[selected].split.research_end))

    sel_cfg = study.configs[selected]
    trend = next(
        (f.params.get("label") for f in sel_cfg.pipeline.factors if f.name == "excess_momentum"),
        None,
    )
    sel = run_variants(sel_cfg, standard_variants(sel_cfg, trend), data)
    _metrics_frame(sel).to_csv(out / f"variants_{selected}.csv", float_format="%.6g")
    ew_cfg = study.configs["B-EW"]
    ew = run_variants(ew_cfg, standard_variants(ew_cfg, None), data)
    _metrics_frame(ew).to_csv(out / "variants_B-EW.csv", float_format="%.6g")

    # Same configs as in the study -> same hashes -> not new trials.
    base_id = "H3-IV" if "IV" in selected else "H3-ERC"
    base_run = run_experiment(study.configs[base_id], data=data)
    sel_run = run_experiment(sel_cfg, data=data)
    null = random_filter_null(base_run.strategy.targets, sel_run.strategy.targets, research)
    (out / "random_filter_null.json").write_text(json.dumps(null, indent=2))

    ml_cfg = study.configs["H6-RIDGE"]
    signs = dict(ml_cfg.pipeline.signal.params["features"])
    panel = build_panel(
        research,
        list(ml_cfg.pipeline.factors),
        signs,
        list(ml_cfg.universe.tradable_tickers),
        pd.Timestamp(ml_cfg.split.research_start),
    )
    ml = cpcv_ic(panel, signs, alpha=float(ml_cfg.pipeline.signal.params["alpha"]))
    (out / "ml_cpcv.json").write_text(json.dumps(ml, indent=2))

    streams = {
        selected: sel_run.strategy.net_returns,
        "B-EW": sel_run.baseline.net_returns,
        "60/40": sel_run.benchmark.net_returns,
    }
    risk_pack(streams, research, out, "research_")
    return {"random_filter_null": null, "ml_cpcv": ml}


def run_holdout(study: Study, selected: str, reason: str, out_root: Path) -> ExperimentResult:
    """THE single holdout evaluation (ledgered unlock)."""
    cfg = study.configs[selected]
    data, _ = load_data(cfg)
    res = run_experiment(cfg, stage="final", data=data, unlock_reason=reason)
    out = out_root / HOLDOUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    pd.Series(res.metrics).to_csv(out / "metrics.csv", float_format="%.6g", header=["value"])
    pd.DataFrame(
        {
            selected: res.strategy.net_returns,
            "B-EW": res.baseline.net_returns,
            "60/40": res.benchmark.net_returns,
            "cash": res.strategy.cash_returns,
        }
    ).to_parquet(out / "returns_net.parquet")
    streams = {
        selected: res.strategy.net_returns,
        "B-EW": res.baseline.net_returns,
        "60/40": res.benchmark.net_returns,
    }
    risk_pack(streams, data, out, "holdout_")
    return res


def current_portfolio(config: ExperimentConfig, data: MarketData) -> pd.Series:
    """Target weights decided at the last available close (2026-09-30 = last session of
    September per the exchange calendar), using the same components as the pipeline."""
    t = data.end
    hist = data.asof(t)
    tradable = set(config.universe.tradable_tickers)
    investable = [a for a in hist.investable(config.universe.min_history_days) if a in tradable]
    factors = [cast(Factor, registry.build("factor", s)) for s in config.pipeline.factors]
    ctx = DecisionContext(
        as_of=t,
        history=hist,
        investable=investable,
        previous_weights=pd.Series(dtype=float),
        constraints=config.constraints,
        rng=np.random.default_rng(config.seed),
        features={f.label: f.compute(hist) for f in factors},
    )
    signal = cast(SignalGenerator, registry.build("signal", config.pipeline.signal))
    pf = cast(PortfolioConstructor, registry.build("portfolio", config.pipeline.portfolio))
    w = pf.construct(signal.generate(ctx), ctx)
    for spec in config.pipeline.risk:
        w = registry.build("risk", spec).adjust(w, ctx)
    return w[w > 1e-9].sort_values(ascending=False)
