from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.config import ExperimentConfig, load_config
from quantlab.data.market_data import MarketData

ROOT = Path(__file__).resolve().parents[1]


def make_market_data(
    *,
    n_days: int = 600,
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    seed: int = 0,
    start: str = "2015-01-01",
    late_ticker: str | None = None,
    late_offset: int = 300,
) -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    rets = rng.normal(0.0003, 0.01, size=(n_days, len(tickers)))
    tri = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=list(tickers))
    vol = pd.DataFrame(rng.lognormal(15, 0.2, size=tri.shape), index=dates, columns=list(tickers))
    if late_ticker:
        tri.iloc[:late_offset, tri.columns.get_loc(late_ticker)] = np.nan
        vol.iloc[:late_offset, vol.columns.get_loc(late_ticker)] = np.nan
    return MarketData(
        close=tri.copy(),
        total_return_index=tri,
        volume=vol,
        asset_class=dict.fromkeys(tickers, "equity"),
    )


@pytest.fixture
def market_data() -> MarketData:
    return make_market_data()


@pytest.fixture
def smoke_config(tmp_path: Path) -> ExperimentConfig:
    """Smoke config with all tracking redirected to a temp dir and a shorter window."""
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    return cfg.model_copy(
        update={
            "split": cfg.split.model_copy(
                update={
                    "data_start": date(2010, 1, 1),
                    "research_start": date(2012, 1, 1),
                    "research_end": date(2016, 12, 31),
                    "holdout_start": date(2017, 1, 1),
                    "holdout_end": date(2018, 12, 31),
                }
            ),
            "tracking": cfg.tracking.model_copy(
                update={
                    "runs_dir": tmp_path / "runs",
                    "ledger_path": tmp_path / "trials.jsonl",
                    "snapshot_dir": tmp_path / "snapshots",
                }
            ),
        }
    )
