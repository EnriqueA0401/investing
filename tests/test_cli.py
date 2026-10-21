from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from quantlab.cli import main
from tests.conftest import ROOT


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    cfg = {
        "extends": str(ROOT / "configs" / "smoke.yaml"),
        "split": {
            "data_start": "2010-01-01",
            "research_start": "2012-01-01",
            "research_end": "2014-12-31",
            "holdout_start": "2015-01-01",
            "holdout_end": "2015-12-31",
        },
        "tracking": {
            "runs_dir": str(tmp_path / "runs"),
            "ledger_path": str(tmp_path / "trials.jsonl"),
            "snapshot_dir": str(tmp_path / "snap"),
        },
    }
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_validate_reports_unresolved_components(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "extends": str(ROOT / "configs" / "base.yaml"),
                "pipeline": {"signal": {"name": "nope"}},
            }
        )
    )
    assert main(["validate", str(p)]) == 1
    assert "signal:nope" in capsys.readouterr().out


def test_run_then_trials(
    cfg_path: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["validate", str(cfg_path)]) == 0
    assert main(["run", str(cfg_path)]) == 0
    capsys.readouterr()
    assert main(["trials", "--ledger", str(tmp_path / "trials.jsonl")]) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts == {"n_trials_research": 1, "n_trials_final": 0, "holdout_unlocks": 0}


def test_final_stage_needs_explicit_reason(cfg_path: Path) -> None:
    with pytest.raises(SystemExit, match="unlock-holdout"):
        main(["run", str(cfg_path), "--stage", "final"])


def test_snapshot_is_reproducible(cfg_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["snapshot", str(cfg_path)])
    first = capsys.readouterr().out.strip()
    main(["snapshot", str(cfg_path)])
    assert capsys.readouterr().out.strip() == first
