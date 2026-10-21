"""Holdout sealing, trial ledger and data snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantlab.config import ExperimentConfig
from quantlab.data.market_data import MarketData
from quantlab.data.store import SnapshotIntegrityError, SnapshotStore
from quantlab.splits import HoldoutGuard, HoldoutLockedError
from quantlab.tracking import RunTracker, TrialLedger
from tests.conftest import make_market_data


def test_research_view_ends_at_research_end(smoke_config: ExperimentConfig, tmp_path: Path) -> None:
    data = make_market_data(n_days=3000, start="2010-01-01")
    guard = HoldoutGuard(smoke_config, TrialLedger(tmp_path / "l.jsonl"))
    view = guard.view(data, "research")
    assert view.end <= pd.Timestamp(smoke_config.split.research_end)


def test_holdout_sealed_until_ledgered_unlock(
    smoke_config: ExperimentConfig, tmp_path: Path
) -> None:
    data = make_market_data(n_days=3000, start="2010-01-01")
    ledger = TrialLedger(tmp_path / "l.jsonl")
    guard = HoldoutGuard(smoke_config, ledger)
    with pytest.raises(HoldoutLockedError):
        guard.view(data, "final")
    guard.unlock("final evaluation of H1")
    assert guard.view(data, "final").end > pd.Timestamp(smoke_config.split.research_end)
    guard.unlock("re-run, same config")  # idempotent for the same config
    assert len(ledger.holdout_unlocks()) == 1


def test_second_config_on_holdout_is_recorded_as_contamination(
    smoke_config: ExperimentConfig, tmp_path: Path
) -> None:
    ledger = TrialLedger(tmp_path / "l.jsonl")
    HoldoutGuard(smoke_config, ledger).unlock("first")
    other = smoke_config.model_copy(update={"seed": 1})
    HoldoutGuard(other, ledger).unlock("second")
    unlocks = ledger.holdout_unlocks()
    assert [u["n_prior"] for u in unlocks] == [0, 1]


def test_ledger_counts_distinct_successful_configs(smoke_config: ExperimentConfig) -> None:
    tracker = RunTracker(smoke_config)
    for _ in range(2):  # same config twice = one trial
        with tracker.start("research") as run:
            run.log_metrics({"sharpe": 0.1})
    other = RunTracker(smoke_config.model_copy(update={"seed": 2}))
    with pytest.raises(RuntimeError), other.start("research"):
        raise RuntimeError("boom")
    assert tracker.ledger.n_trials() == 1
    statuses = [r["status"] for r in tracker.ledger.records("trial")]
    assert statuses == ["ok", "ok", "failed"]


def test_ledger_is_append_only_jsonl(smoke_config: ExperimentConfig) -> None:
    ledger = TrialLedger(smoke_config.tracking.ledger_path)
    ledger.append("trial", {"config_hash": "x", "status": "ok", "stage": "research"})
    ledger.append("trial", {"config_hash": "y", "status": "ok", "stage": "research"})
    lines = smoke_config.tracking.ledger_path.read_text().splitlines()
    assert [json.loads(line)["config_hash"] for line in lines] == ["x", "y"]


def test_snapshot_roundtrip_is_content_addressed(tmp_path: Path, market_data: MarketData) -> None:
    store = SnapshotStore(tmp_path)
    sid = store.save(market_data, {"source": "test"})
    assert store.save(market_data, {"source": "again"}) == sid
    loaded = store.load(sid)
    pd.testing.assert_frame_equal(
        loaded.total_return_index, market_data.total_return_index, check_freq=False
    )
    other = make_market_data(seed=99)
    assert store.save(other, {}) != sid


def test_snapshot_tampering_detected(tmp_path: Path, market_data: MarketData) -> None:
    store = SnapshotStore(tmp_path)
    sid = store.save(market_data, {})
    path = tmp_path / sid / "total_return_index.parquet"
    df = pd.read_parquet(path)
    df.iloc[10, 0] *= 1.01
    df.to_parquet(path)
    with pytest.raises(SnapshotIntegrityError):
        store.load(sid)
