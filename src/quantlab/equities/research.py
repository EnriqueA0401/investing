"""Post-selection steps of the equity study: robustness (research window), ML
IC diagnostic, the single holdout evaluation and the comparison against SPY."""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any, cast

import numpy as np
import pandas as pd

from quantlab import metrics as m
from quantlab import stats
from quantlab.config import ExperimentConfig
from quantlab.data.market_data import MarketData
from quantlab.equities.analysis import ff_alpha, survivorship_drag
from quantlab.equities.portfolio import neutralize
from quantlab.ml_diagnostic import cpcv_ic, walk_forward_ic
from quantlab.pipeline import ExperimentResult, load_data, run_experiment
from quantlab.registry import build
from quantlab.risk import drawdown_table, var_cvar
from quantlab.robustness import variant
from quantlab.study import Study


def equity_variants(cfg: ExperimentConfig) -> dict[str, dict[str, Any]]:
    """Robustness grid fixed before running it (research log, Fase 6 acciones)."""
    pipe = cfg.pipeline.model_dump(mode="json")
    c = cfg.costs
    tiers = pipe["cost_model"]["params"].get("tiers_bps", [1.5, 3.0, 6.0])
    out: dict[str, dict[str, Any]] = {
        "costs_x3": {
            "costs": {
                "commission_bps": c.commission_bps * 3,
                "slippage_bps": c.slippage_bps * 3,
                "impact_coefficient": c.impact_coefficient * 3,
            },
            "pipeline": {
                "cost_model": {
                    "name": pipe["cost_model"]["name"],
                    "params": {"tiers_bps": [t * 3 for t in tiers]},
                }
            },
        },
        "borrow_x3": {"costs": {"borrow_fee_bps_annual": c.borrow_fee_bps_annual * 3}},
        "borrow_x6": {"costs": {"borrow_fee_bps_annual": c.borrow_fee_bps_annual * 6}},
        "lag_5d": {"execution": {"execution_lag_days": 5}},
        "quarterly": {"execution": {"rebalance_frequency": "QE"}},
        "aum_1e8": {"fund": {"aum_usd": 1e8}},
        "aum_1e9": {"fund": {"aum_usd": 1e9}},
        "no_vol_target": {
            "pipeline": {"risk": [r for r in pipe["risk"] if r["name"] != "vol_target"]}
        },
    }
    if pipe["portfolio"]["name"] == "ls_sector_neutral":
        for n in (25, 60):
            params = {**pipe["portfolio"]["params"], "n_per_side": n}
            out[f"n_per_side_{n}"] = {
                "pipeline": {"portfolio": {"name": "ls_sector_neutral", "params": params}}
            }
    return out


def ml_ic_diagnostic(cfg: ExperimentConfig, data: MarketData) -> dict[str, Any]:
    """CPCV and strict walk-forward IC of ridge vs the equal-weight composite on the
    research window (same neutralized features as the strategies)."""
    research = data.asof(pd.Timestamp(cfg.split.research_end))
    weights = dict(cfg.pipeline.signal.params["features"])
    panels = {}
    for spec in cfg.pipeline.factors:
        f = build("factor", spec)
        panels[f.label] = f.compute(research)
    months = [
        d for d in panels[next(iter(weights))].index if d >= pd.Timestamp(cfg.split.research_start)
    ]
    tri = research.total_return_index
    rows = []
    for s, s_next in pairwise(months):
        u = research.universe
        assert u is not None
        row = u.loc[u.index <= s].iloc[-1]
        names = [
            str(n)
            for n in row[row.astype(bool)].index
            if research.asset_class.get(str(n)) not in ("Money", "benchmark")
        ]
        sector = pd.Series({n: research.asset_class.get(n, "Other") for n in names})
        x = pd.DataFrame(
            {
                f: neutralize(
                    cast(pd.Series, panels[f].loc[s]).reindex(names).astype(float), sector, None
                )
                for f in weights
            }
        )
        x = x[x.notna().sum(axis=1) >= 6].fillna(0.0)
        fwd = (tri.loc[s_next] / tri.loc[s] - 1).reindex(x.index).dropna()
        if len(fwd) < 50:
            continue
        z = x.loc[fwd.index]
        z["label"] = fwd.rank(pct=True) - 0.5
        z["t0"], z["t1"] = s, s_next
        rows.append(z.reset_index(names="ticker"))
    panel = pd.concat(rows, ignore_index=True)
    signs = {k: float(np.sign(v)) * abs(v) for k, v in weights.items()}
    return {"cpcv": cpcv_ic(panel, signs), "walk_forward": walk_forward_ic(panel, signs)}


def run_equity_robustness(study: Study, selected: str) -> dict[str, Any]:
    cfg = study.configs[selected]
    data, _ = load_data(cfg)
    out = study.output_dir / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    results = {
        name: run_experiment(variant(cfg, f"{selected}:{name}", ov), data=data)
        for name, ov in equity_variants(cfg).items()
    }
    pd.DataFrame({k: r.metrics for k, r in results.items()}).T.to_csv(
        out / f"variants_{selected}.csv", float_format="%.6g"
    )
    ml = ml_ic_diagnostic(study.configs["S-ML-RIDGE"], data)
    (out / "ml_ic.json").write_text(json.dumps(ml, indent=2, default=float))
    return {"variants": list(results), "ml": ml}


def comparison_vs_benchmark(
    df: pd.DataFrame, cash: pd.Series, bench: str, french: pd.DataFrame | None
) -> pd.DataFrame:
    mo = cast(pd.DataFrame, stats.to_monthly(df))
    b = mo[bench].to_numpy()
    rows = {}
    for c in df:
        r = df[c]
        mc = mo[c].to_numpy()
        row: dict[str, float] = {
            "retorno_total": float(np.prod(1 + r.to_numpy()) - 1),
            "cagr": m.annualized_return(r),
            "volatilidad": m.annualized_vol(r),
            "sharpe": m.sharpe_ratio(r, cash),
            "sortino": m.sortino_ratio(r, cash),
            "max_drawdown": m.max_drawdown(r),
            "corr_diaria": float(r.corr(df[bench])),
            "beta": m.beta(r, df[bench]),
            "up_capture": float(mc[b > 0].mean() / b[b > 0].mean()),
            "down_capture": float(mc[b < 0].mean() / b[b < 0].mean()),
            "var95_mensual": float(var_cvar(mo[c])["VaR_hist"].iloc[0]),
            "cvar95_mensual": float(var_cvar(mo[c])["CVaR_hist"].iloc[0]),
        }
        if french is not None:
            ff = ff_alpha(r - cash, french)
            row |= {
                "ff_alpha_anual": ff["alpha_annual"],
                "ff_alpha_t": ff["alpha_t"],
                "ff_r2": ff["r2"],
            }
        rows[str(c)] = row
    return pd.DataFrame(rows).T


def run_equity_holdout(
    study: Study, selected: str, reason: str, french: pd.DataFrame, coverage: pd.DataFrame
) -> ExperimentResult:
    """THE single holdout evaluation of the pre-registered selection."""
    cfg = study.configs[selected]
    data, _ = load_data(cfg)
    res = run_experiment(cfg, stage="final", data=data, unlock_reason=reason)
    out = study.output_dir / "holdout"
    out.mkdir(parents=True, exist_ok=True)
    idx = res.strategy.net_returns.index
    df = pd.DataFrame(
        {
            selected: res.strategy.net_returns,
            "EW-universo (long-only)": res.baseline.net_returns.reindex(idx),
            "SPY": res.benchmark.net_returns.reindex(idx),
        }
    ).dropna()
    cash = res.strategy.cash_returns.reindex(df.index)
    df.assign(cash=cash).to_parquet(out / "returns_net.parquet")
    pd.Series(res.metrics).to_csv(out / "metrics.csv", header=["value"], float_format="%.6g")
    comparison_vs_benchmark(df, cash, "SPY", french).to_csv(out / "vs_spy.csv", float_format="%.6g")
    drawdown_table(df[selected]).to_csv(out / "drawdowns.csv", index=False)
    drag = survivorship_drag(coverage, pd.DatetimeIndex(df.index))
    ex = df[selected] - cash
    tests = {
        "psr": stats.probabilistic_sharpe_ratio(ex),
        "sharpe": stats.per_period_sharpe(ex) * np.sqrt(252),
        "sharpe_after_drag": stats.per_period_sharpe(ex - drag) * np.sqrt(252),
        "n_days": len(ex),
    }
    (out / "holdout_tests.json").write_text(json.dumps(tests, indent=2, default=float))
    return res
