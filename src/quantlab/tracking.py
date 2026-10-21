"""Lightweight experiment tracking and the append-only trial ledger.

* ``runs/<run_id>/`` (git-ignored) holds the resolved config, metrics,
  environment fingerprint and artifacts of each run.
* ``research/trials.jsonl`` (committed) is the append-only ledger of every
  configuration ever evaluated. It is the source of truth for the number of
  trials used in multiple-testing corrections (Deflated Sharpe, PBO) and for
  auditing holdout access. Never edit or truncate it by hand.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from quantlab.config import ExperimentConfig
from quantlab.logging_setup import get_logger
from quantlab.reproducibility import environment_fingerprint, git_state

log = get_logger(__name__)

Stage = Literal["research", "final"]
EventType = Literal["trial", "holdout_unlock"]


class TrialLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: EventType, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "ts": datetime.now(UTC).isoformat(), **payload}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def records(self, event: EventType | None = None) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"corrupt ledger line {lineno} in {self.path}") from exc
                if event is None or rec.get("event") == event:
                    yield rec

    def trial_hashes(self, hypothesis_id: str | None = None, stage: Stage = "research") -> set[str]:
        """Distinct configurations that completed evaluation (failed runs excluded)."""
        return {
            r["config_hash"]
            for r in self.records("trial")
            if r.get("status") == "ok"
            and r.get("stage") == stage
            and (hypothesis_id is None or hypothesis_id in r.get("hypothesis_ids", []))
        }

    def n_trials(self, hypothesis_id: str | None = None, stage: Stage = "research") -> int:
        return len(self.trial_hashes(hypothesis_id, stage))

    def holdout_unlocks(self) -> list[dict[str, Any]]:
        return list(self.records("holdout_unlock"))


@dataclass
class Run:
    config: ExperimentConfig
    stage: Stage
    run_dir: Path
    ledger: TrialLedger
    run_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self.metrics.update({k: float(v) for k, v in metrics.items()})

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def log_frame(self, name: str, frame: pd.DataFrame | pd.Series) -> Path:
        out = self.run_dir / "artifacts" / f"{name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df = frame.to_frame() if isinstance(frame, pd.Series) else frame
        df.columns = [str(c) for c in df.columns]
        df.to_parquet(out)
        return out

    def _finish(self, status: Literal["ok", "failed"], error: str | None = None) -> None:
        summary = {
            "run_id": self.run_id,
            "name": self.config.name,
            "config_hash": self.config.config_hash(),
            "hypothesis_ids": list(self.config.hypothesis_ids),
            "stage": self.stage,
            "status": status,
            "error": error,
            "data_snapshot": self.config.data_snapshot,
            "git": git_state(),
            "metrics": self.metrics,
            "tags": self.tags,
        }
        (self.run_dir / "run.json").write_text(
            json.dumps({**summary, "env": environment_fingerprint()}, indent=2, default=str)
        )
        self.ledger.append("trial", summary)
        log.info("run finished", extra={"run_id": self.run_id, "status": status})


class RunTracker:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.runs_dir = config.tracking.runs_dir
        self.ledger = TrialLedger(config.tracking.ledger_path)

    def start(self, stage: Stage) -> RunContext:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{self.config.config_hash()[:8]}-{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=False)
        )
        return RunContext(Run(self.config, stage, run_dir, self.ledger, run_id))


class RunContext:
    """Context manager that always records the run in the ledger, even on failure."""

    def __init__(self, run: Run) -> None:
        self.run = run

    def __enter__(self) -> Run:
        return self.run

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        self.run._finish("failed" if exc else "ok", repr(exc) if exc else None)
