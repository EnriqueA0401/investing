"""Run a pre-registered set of experiments and persist their return streams.

The study file lists experiment overrides on top of a base config. Every
experiment run is recorded in the trial ledger. Outputs (committed, small):

* ``returns_net.parquet``   daily net returns, one column per experiment
* ``returns_excess.parquet`` daily net returns minus cash
* ``reference.parquet``     benchmark (60/40), baseline (EW) and cash returns
* ``metrics.csv``           headline metrics per experiment
* ``weights_last.csv``      last target weights per experiment
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantlab.config import ExperimentConfig, _deep_merge, _load_raw
from quantlab.logging_setup import get_logger
from quantlab.pipeline import ExperimentResult, load_data, run_experiment
from quantlab.tracking import Stage

log = get_logger(__name__)


@dataclass
class Study:
    configs: dict[str, ExperimentConfig]
    output_dir: Path


def load_study(path: str | Path) -> Study:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    base = _load_raw(path.parent / raw["base"])
    configs: dict[str, Any] = {}
    for exp_id, override in raw["experiments"].items():
        merged = _deep_merge(base, {**override, "name": exp_id})
        configs[exp_id] = ExperimentConfig.model_validate(merged)
    hashes = [c.config_hash() for c in configs.values()]
    if len(set(hashes)) != len(hashes):
        raise ValueError("two study experiments resolve to the same configuration")
    return Study(configs=configs, output_dir=(path.parent / raw["output_dir"]).resolve())


def run_study(
    study: Study, stage: Stage = "research", only: list[str] | None = None
) -> dict[str, ExperimentResult]:
    ids = only or list(study.configs)
    unknown = set(ids) - set(study.configs)
    if unknown:
        raise KeyError(f"not in the pre-registered study: {sorted(unknown)}")
    data, _ = load_data(study.configs[ids[0]])
    results = {}
    for exp_id in ids:
        log.info("study experiment", extra={"experiment": exp_id})
        results[exp_id] = run_experiment(study.configs[exp_id], stage=stage, data=data)
    if stage == "research":
        save_results(results, study.output_dir)
    return results


def save_results(results: dict[str, ExperimentResult], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    net = pd.DataFrame({k: r.strategy.net_returns for k, r in results.items()})
    excess = pd.DataFrame({k: r.strategy.excess_returns() for k, r in results.items()})
    any_run = next(iter(results.values()))
    reference = pd.DataFrame(
        {
            "benchmark_60_40": any_run.benchmark.net_returns,
            "baseline_ew": any_run.baseline.net_returns,
            "cash": any_run.strategy.cash_returns,
        }
    )
    net.to_parquet(out / "returns_net.parquet")
    excess.to_parquet(out / "returns_excess.parquet")
    reference.to_parquet(out / "reference.parquet")
    metrics = pd.DataFrame({k: r.metrics for k, r in results.items()}).T
    metrics.index.name = "experiment"
    metrics.to_csv(out / "metrics.csv", float_format="%.6g")
    last = pd.DataFrame({k: r.strategy.targets.iloc[-1] for k, r in results.items()}).T.fillna(0.0)
    last.index.name = "experiment"
    last.to_csv(out / "weights_last.csv", float_format="%.4f")
    turnover = pd.DataFrame({k: r.strategy.turnover for k, r in results.items()})
    turnover.to_parquet(out / "turnover.parquet")
