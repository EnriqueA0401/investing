"""Post-selection robustness on the research window (cannot change the selection).

* ``run_variants``: re-run a config with overrides (parameters, costs, lag,
  frequency, AUM). Every run is recorded in the trial ledger.
* ``random_filter_null``: is the *choice* of which assets the trend filter
  switches off better than switching off the same number at random?
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import ExperimentConfig, _deep_merge
from quantlab.data.market_data import MarketData
from quantlab.pipeline import ExperimentResult, run_experiment


def variant(config: ExperimentConfig, name: str, override: dict[str, Any]) -> ExperimentConfig:
    raw = _deep_merge(config.model_dump(mode="json"), {**override, "name": name})
    return ExperimentConfig.model_validate(raw)


def cost_override(config: ExperimentConfig, mult: float) -> dict[str, Any]:
    c = config.costs
    return {
        "costs": {
            "half_spread_bps": {k: v * mult for k, v in c.half_spread_bps.items()},
            "commission_bps": c.commission_bps * mult,
            "slippage_bps": c.slippage_bps * mult,
            "impact_coefficient": c.impact_coefficient * mult,
        }
    }


def standard_variants(
    config: ExperimentConfig, trend_feature: str | None
) -> dict[str, dict[str, Any]]:
    """The robustness grid fixed in the research log before running it."""
    out: dict[str, dict[str, Any]] = {}
    pipe = config.pipeline.model_dump(mode="json")
    if trend_feature is not None:
        for lb in (126, 189, 252, 315):
            for vw in (21, 63, 126):
                factors = [
                    {
                        "name": "excess_momentum",
                        "params": {"label": trend_feature, "lookback_days": lb},
                    }
                ]
                port = {
                    **pipe["portfolio"],
                    "params": {**pipe["portfolio"]["params"], "vol_window_days": vw},
                }
                out[f"param_trend{lb}_vol{vw}"] = {
                    "pipeline": {"factors": factors, "portfolio": port}
                }
    for mult in (2.0, 3.0, 5.0):
        out[f"costs_x{mult:g}"] = cost_override(config, mult)
    for lag in (2, 5):
        out[f"lag_{lag}d"] = {"execution": {"execution_lag_days": lag}}
    out["quarterly"] = {"execution": {"rebalance_frequency": "QE"}}
    for aum in (1e7, 1e8, 1e9, 1e10):
        out[f"aum_{aum:.0e}"] = {"fund": {"aum_usd": aum}}
    out["turnover_cap_10pct"] = {
        "pipeline": {
            "risk": [*pipe["risk"], {"name": "turnover_cap", "params": {"max_turnover": 0.10}}]
        }
    }
    return out


def run_variants(
    config: ExperimentConfig, variants: dict[str, dict[str, Any]], data: MarketData
) -> dict[str, ExperimentResult]:
    return {
        name: run_experiment(variant(config, f"{config.name}:{name}", ov), data=data)
        for name, ov in variants.items()
    }


def random_filter_null(
    base_targets: pd.DataFrame,
    strategy_targets: pd.DataFrame,
    data: MarketData,
    n_sims: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Monthly, gross-of-cost comparison of the strategy's filter vs random filters.

    ``base_targets``: unfiltered weights per decision date (e.g. H3-IV).
    ``strategy_targets``: filtered weights (e.g. H4-IV-TS12) on the same dates.
    Each month the random filter keeps the same *number* of assets as the
    strategy, chosen uniformly among those the base portfolio holds.
    """
    dates = base_targets.index.intersection(strategy_targets.index)
    tri = data.total_return_index
    cash_idx = (1 + data.cash()).cumprod()
    # Holding-period returns between consecutive fill dates (targets are indexed by fill).
    levels = tri.loc[dates].to_numpy(dtype=float)
    asset_r = levels[1:] / levels[:-1] - 1
    cash_levels = cash_idx.loc[dates].to_numpy(dtype=float)
    cash_r = cash_levels[1:] / cash_levels[:-1] - 1
    base = base_targets.reindex(index=dates, columns=tri.columns).fillna(0.0).to_numpy()[:-1]
    strat = strategy_targets.reindex(index=dates, columns=tri.columns).fillna(0.0).to_numpy()[:-1]
    asset_r = np.nan_to_num(asset_r)

    def excess_sharpe(w: np.ndarray) -> float:
        port = (w * asset_r).sum(axis=-1) + (1 - w.sum(axis=-1)) * cash_r
        ex = port - cash_r
        return float(ex.mean(axis=-1) / ex.std(axis=-1, ddof=1) * np.sqrt(12))

    observed = excess_sharpe(strat)
    rng = np.random.default_rng(seed)
    held = base > 0
    k = (strat > 0).sum(axis=1)
    sims = np.empty(n_sims)
    for i in range(n_sims):
        keys = np.where(held, rng.random(held.shape), -1.0)
        thresh = -np.sort(-keys, axis=1)[np.arange(len(k)), np.maximum(k - 1, 0)]
        mask = (keys >= thresh[:, None]) & held & (k[:, None] > 0)
        sims[i] = excess_sharpe(base * mask)
    return {
        "observed_sharpe_monthly_gross": observed,
        "random_mean": float(sims.mean()),
        "random_p95": float(np.quantile(sims, 0.95)),
        "p_value": float((1 + (sims >= observed).sum()) / (n_sims + 1)),
        "n_sims": n_sims,
    }
