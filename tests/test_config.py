from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quantlab.config import ExperimentConfig, load_config
from quantlab.study import load_study
from tests.conftest import ROOT


def test_base_and_smoke_configs_load() -> None:
    base = load_config(ROOT / "configs" / "base.yaml")
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    assert base.universe.tickers == smoke.universe.tickers
    assert smoke.pipeline.data_source.name == "synthetic"
    assert base.constraints.long_only
    assert base.constraints.max_gross_leverage == 1.0


def test_hash_ignores_name_but_not_parameters() -> None:
    cfg = load_config(ROOT / "configs" / "base.yaml")
    renamed = cfg.model_copy(update={"name": "other"})
    assert renamed.config_hash() == cfg.config_hash()
    tweaked = cfg.model_copy(update={"seed": cfg.seed + 1})
    assert tweaked.config_hash() != cfg.config_hash()


def _raw(tmp_path: Path, text: str) -> Path:
    base = (ROOT / "configs" / "base.yaml").read_text()
    p = tmp_path / "c.yaml"
    p.write_text(base + "\n" + text)
    return p


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="extra"):
        load_config(_raw(tmp_path, "typo_field: 1"))


def test_split_must_be_ordered() -> None:
    cfg = load_config(ROOT / "configs" / "base.yaml")
    data = cfg.model_dump()
    data["split"]["research_end"] = data["split"]["holdout_start"]
    with pytest.raises(ValidationError, match="research_end < holdout_start"):
        ExperimentConfig.model_validate(data)


def test_zero_execution_lag_rejected() -> None:
    data = load_config(ROOT / "configs" / "base.yaml").model_dump()
    data["execution"]["execution_lag_days"] = 0
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


def test_long_only_cannot_be_levered() -> None:
    data = load_config(ROOT / "configs" / "base.yaml").model_dump()
    data["constraints"]["max_gross_leverage"] = 1.5
    with pytest.raises(ValidationError, match="unlevered"):
        ExperimentConfig.model_validate(data)


def test_benchmark_must_be_in_universe() -> None:
    data = load_config(ROOT / "configs" / "base.yaml").model_dump()
    data["benchmark"]["weights"] = {"QQQ": 1.0}
    with pytest.raises(ValidationError, match="not in universe"):
        ExperimentConfig.model_validate(data)


def test_circular_extends_detected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_ledgered_study_hashes_are_stable() -> None:
    """Adding config fields must not change hashes already recorded in the trial ledger."""
    ledger = (ROOT / "research" / "trials.jsonl").read_text().splitlines()
    known = {
        r["name"]: r["config_hash"]
        for r in map(json.loads, ledger)
        if r.get("event") == "trial" and r.get("stage") == "research"
    }
    study = load_study(ROOT / "configs" / "study.yaml")
    assert {k: c.config_hash() for k, c in study.configs.items()} == {
        k: known[k] for k in study.configs
    }
