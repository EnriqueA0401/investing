"""Pre-registered acceptance criteria for the equity study
(docs/hypotheses/equities/STUDY_PLAN.md §5-6)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from quantlab import stats
from quantlab.metrics import as_float
from quantlab.risk import factor_exposures
from quantlab.tracking import TrialLedger

SUBPERIODS = [
    ("2012-01-01", "2014-12-31"),
    ("2015-01-01", "2017-12-31"),
    ("2018-01-01", "2020-12-31"),
    ("2021-01-01", "2022-12-31"),
]
DSR_THRESHOLD = 0.95
MIN_POSITIVE_SUBPERIODS = 3
PBO_THRESHOLD = 0.5
DRAG_EVENT_RETURN = 0.30
DRAG_HORIZON_END = 2026.75
ALPHA_T = 2.0
FF_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
COMPLEXITY = {
    "S-VAL": 3, "S-QUAL": 3, "S-MOM": 1, "S-RMOM": 1, "S-LVOL": 2, "S-INV": 1,
    "S-COMP": 11, "S-COMP-OPT": 15, "S-COMP-MACRO": 13, "S-ML-GBM": 20, "S-ML-RIDGE": 15,
}  # fmt: skip


def survivorship_drag(coverage: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Daily pessimistic drag: for universe year y (Jul y .. Jun y+1),
    2 x 0.30 x (1 - coverage_y) / (2026.75 - y), spread over 252 sessions."""
    cov = coverage.assign(year=pd.to_datetime(coverage["rank_date"]).dt.year).set_index("year")
    drag = pd.Series(0.0, index=index)
    for y, row in cov.iterrows():
        yy = int(str(y))
        annual = (
            2 * DRAG_EVENT_RETURN * (1 - float(row["coverage_count"])) / (DRAG_HORIZON_END - yy)
        )
        mask = (index >= pd.Timestamp(yy, 7, 1)) & (index <= pd.Timestamp(yy + 1, 6, 30))
        drag[mask] = annual / 252
    return drag


def ff_alpha(excess_daily: pd.Series, french: pd.DataFrame, lags: int = 5) -> dict[str, float]:
    f = french[FF_COLUMNS].reindex(excess_daily.index).dropna()
    y = excess_daily.reindex(f.index)
    out = factor_exposures(y, f, lags=lags)
    return {
        "alpha_annual": as_float(out.loc["alpha_monthly", "coef"]) * 252,
        "alpha_t": as_float(out.loc["alpha_monthly", "t_stat"]),
        "r2": as_float(out.loc["R2", "coef"]),
        **{f"beta_{c}": as_float(out.loc[c, "coef"]) for c in FF_COLUMNS},
    }


@dataclass
class EquityVerdict:
    sharpe: float
    dsr: float
    dsr_hurdle_annual: float
    subperiod_sharpes: dict[str, float]
    n_positive_subperiods: int
    sharpe_after_drag: float
    drag_annual_mean: float
    ff: dict[str, float]
    passes_dsr: bool
    passes_stability: bool
    passes_survivorship: bool
    is_alpha: bool

    @property
    def passes_individual(self) -> bool:
        return self.passes_dsr and self.passes_stability and self.passes_survivorship


@dataclass
class EquityStudyVerdict:
    n_trials: int
    pbo: float
    pbo_prob_oos_loss: float
    candidates: dict[str, EquityVerdict]
    correlations: dict[str, dict[str, float]]
    validated: list[str]
    selected: str
    selected_is_validated: bool

    def to_json(self) -> str:
        payload: dict[str, Any] = asdict(self)
        for k, v in self.candidates.items():
            payload["candidates"][k]["passes_individual"] = v.passes_individual
        return json.dumps(payload, indent=2, default=float)


def evaluate_equity_study(
    results_dir: Path,
    ledger: TrialLedger,
    selection_hashes: set[str],
    coverage: pd.DataFrame,
    french: pd.DataFrame,
) -> EquityStudyVerdict:
    excess = pd.read_parquet(results_dir / "returns_excess.parquet").dropna()
    n_trials = len(ledger.trial_hashes(stage="research") & selection_hashes)
    if n_trials != excess.shape[1]:
        raise RuntimeError(f"{n_trials} ledgered selection trials vs {excess.shape[1]} columns")
    trial_srs = excess.apply(stats.per_period_sharpe).to_numpy()
    monthly = cast(pd.DataFrame, stats.to_monthly(excess))
    drag = survivorship_drag(coverage, pd.DatetimeIndex(excess.index))
    ann = np.sqrt(252)
    cands: dict[str, EquityVerdict] = {}
    for name in excess.columns:
        x = excess[name]
        dsr, hurdle = stats.deflated_sharpe_ratio(x, trial_srs, n_trials)
        sub = stats.subperiod_sharpes(x, SUBPERIODS)
        after = stats.per_period_sharpe(x - drag) * ann
        ff = ff_alpha(x, french)
        n_pos = int((sub > 0).sum())
        cands[str(name)] = EquityVerdict(
            sharpe=stats.per_period_sharpe(x) * ann,
            dsr=dsr,
            dsr_hurdle_annual=hurdle * ann,
            subperiod_sharpes={str(k): float(v) for k, v in sub.items()},
            n_positive_subperiods=n_pos,
            sharpe_after_drag=after,
            drag_annual_mean=float(drag.mean() * 252),
            ff=ff,
            passes_dsr=dsr >= DSR_THRESHOLD,
            passes_stability=n_pos >= MIN_POSITIVE_SUBPERIODS,
            passes_survivorship=after > 0,
            is_alpha=ff["alpha_t"] >= ALPHA_T,
        )
    pbo = stats.probability_of_backtest_overfitting(monthly, n_blocks=16)
    validated = (
        [k for k, v in cands.items() if v.passes_individual] if pbo.pbo < PBO_THRESHOLD else []
    )
    pool = validated or list(cands)
    best = max(cands[k].dsr for k in pool)
    tied = [k for k in pool if cands[k].dsr >= best - 0.01]
    selected = min(tied, key=lambda k: (COMPLEXITY.get(k, 99), -cands[k].dsr))
    corr = monthly.corr().round(3)
    return EquityStudyVerdict(
        n_trials=n_trials,
        pbo=pbo.pbo,
        pbo_prob_oos_loss=pbo.prob_oos_loss,
        candidates=cands,
        correlations={str(k): {str(c): float(v) for c, v in r.items()} for k, r in corr.iterrows()},
        validated=validated,
        selected=selected,
        selected_is_validated=bool(validated),
    )
