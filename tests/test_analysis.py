from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.analysis import evaluate_study
from quantlab.tracking import TrialLedger


def _setup(tmp_path: Path, edge: float) -> tuple[Path, TrialLedger, set[str]]:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2005-01-03", "2022-12-30")
    cols = ["B-EW", "H1-TS6", "H3-IV"]
    data = rng.normal(0.0002, 0.006, (len(idx), 3))
    data[:, 1] += edge
    pd.DataFrame(data, index=idx, columns=cols).to_parquet(tmp_path / "returns_excess.parquet")
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    hashes = {"a", "b", "c"}
    for h in [*hashes, "robustness-x"]:
        ledger.append("trial", {"config_hash": h, "status": "ok", "stage": "research"})
    return tmp_path, ledger, hashes


def test_noise_candidates_are_not_validated(tmp_path: Path) -> None:
    out, ledger, hashes = _setup(tmp_path, edge=0.0)
    v = evaluate_study(out, ledger, hashes, n_boot=300)
    assert v.n_trials == 3  # robustness trial excluded from the selection N
    assert v.validated == []
    assert not v.selected_is_validated


def test_strong_edge_is_validated(tmp_path: Path) -> None:
    out, ledger, hashes = _setup(tmp_path, edge=0.0008)
    v = evaluate_study(out, ledger, hashes, n_boot=300)
    assert v.validated == ["H1-TS6"]
    assert v.selected == "H1-TS6"


def test_ledger_out_of_sync_raises(tmp_path: Path) -> None:
    out, ledger, _ = _setup(tmp_path, edge=0.0)
    with pytest.raises(RuntimeError, match="out of sync"):
        evaluate_study(out, ledger, {"a"}, n_boot=100)
