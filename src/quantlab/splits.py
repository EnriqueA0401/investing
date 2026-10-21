"""Holdout sealing.

Research-stage code only ever sees data up to ``split.research_end``. The
holdout becomes visible only through an explicit, ledgered unlock. Unlocking
for a second, *different* configuration is allowed (refusing would just push
people to work around the tool) but it is recorded and surfaced as holdout
contamination in every subsequent report.
"""

from __future__ import annotations

import pandas as pd

from quantlab.config import ExperimentConfig
from quantlab.data.market_data import MarketData
from quantlab.logging_setup import get_logger
from quantlab.tracking import Stage, TrialLedger

log = get_logger(__name__)


class HoldoutLockedError(PermissionError):
    pass


class HoldoutGuard:
    def __init__(self, config: ExperimentConfig, ledger: TrialLedger) -> None:
        self.config = config
        self.split = config.split
        self.ledger = ledger

    def unlock(self, reason: str) -> None:
        cfg_hash = self.config.config_hash()
        prior = {u["config_hash"] for u in self.ledger.holdout_unlocks()}
        if cfg_hash in prior:
            log.info("holdout already unlocked for this config", extra={"config_hash": cfg_hash})
            return
        if prior:
            log.warning(
                "HOLDOUT CONTAMINATION: holdout previously evaluated with other configs",
                extra={"previous": sorted(prior), "config_hash": cfg_hash},
            )
        self.ledger.append(
            "holdout_unlock",
            {
                "config_hash": cfg_hash,
                "name": self.config.name,
                "reason": reason,
                "n_prior": len(prior),
            },
        )

    def is_unlocked(self) -> bool:
        cfg_hash = self.config.config_hash()
        return any(u["config_hash"] == cfg_hash for u in self.ledger.holdout_unlocks())

    def view(self, data: MarketData, stage: Stage) -> MarketData:
        """Data visible to the given stage. Pre-``research_start`` rows are kept as warm-up."""
        if stage == "research":
            return data.asof(pd.Timestamp(self.split.research_end))
        if not self.is_unlocked():
            raise HoldoutLockedError(
                "holdout is sealed; call HoldoutGuard.unlock(reason) (CLI: --unlock-holdout) first"
            )
        end = self.split.holdout_end
        return data.asof(pd.Timestamp(end)) if end is not None else data

    def evaluation_window(self, stage: Stage) -> tuple[pd.Timestamp, pd.Timestamp | None]:
        if stage == "research":
            return pd.Timestamp(self.split.research_start), pd.Timestamp(self.split.research_end)
        end = self.split.holdout_end
        return pd.Timestamp(self.split.holdout_start), (pd.Timestamp(end) if end else None)
