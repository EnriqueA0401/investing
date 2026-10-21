"""Apply the pre-registered acceptance criteria (STUDY_PLAN.md §4-5) to study results.

Choices fixed in the plan (or, where the plan was silent, fixed here *before*
the first evaluation and documented in the research log):
* DSR on daily net excess returns; N = distinct research configs in the ledger;
  V[SR] = variance of the per-period Sharpe across those configs.
* Sharpe difference vs B-EW: monthly excess returns, circular block bootstrap
  (block 6, 10 000 draws), one-sided, p < 0.05.
* Stability: excess Sharpe > 0 in >= 3 of the 4 sub-periods.
* PBO: CSCV with 16 blocks on *monthly* excess returns of all candidates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab import stats
from quantlab.tracking import TrialLedger

SUBPERIODS = [
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2022-12-31"),
]
DSR_THRESHOLD = 0.95
P_THRESHOLD = 0.05
MIN_POSITIVE_SUBPERIODS = 3
PBO_THRESHOLD = 0.5
BASELINE_ID = "B-EW"
# Number of tunable parameters, for the pre-registered "simplest wins ties" rule.
COMPLEXITY = {
    "B-EW": 0, "H1-TS6": 1, "H1-TS12": 1, "H2-XS12": 2, "H2-XS6": 2, "H2-XS12-RMV": 4,
    "H3-IV": 1, "H3-ERC": 1, "H3-HRP": 1, "H3-MV": 1, "H4-IV-TS12": 2, "H4-ERC-TS12": 2,
    "H5-EW-VT": 2, "H6-COMBO": 6, "H6-RIDGE": 8,
}  # fmt: skip


@dataclass
class CandidateVerdict:
    sharpe: float
    dsr: float
    dsr_hurdle_annual: float
    diff_vs_ew: float
    p_vs_ew: float
    ci_vs_ew: tuple[float, float]
    subperiod_sharpes: dict[str, float]
    n_positive_subperiods: int
    passes_dsr: bool
    passes_vs_ew: bool
    passes_stability: bool

    @property
    def passes_individual(self) -> bool:
        return self.passes_dsr and self.passes_vs_ew and self.passes_stability


@dataclass
class StudyVerdict:
    n_trials: int
    pbo: float
    pbo_prob_oos_loss: float
    pbo_median_oos_sharpe_of_is_best: float
    candidates: dict[str, CandidateVerdict]
    walk_forward: list[dict[str, Any]]
    walk_forward_summary: dict[str, float]
    validated: list[str]
    selected: str
    selected_is_validated: bool

    def to_json(self) -> str:
        payload = asdict(self)
        for k, v in self.candidates.items():
            payload["candidates"][k]["passes_individual"] = v.passes_individual
        return json.dumps(payload, indent=2, default=float)


def evaluate_study(results_dir: Path, ledger: TrialLedger, n_boot: int = 10_000) -> StudyVerdict:
    excess = pd.read_parquet(results_dir / "returns_excess.parquet").dropna(how="all")
    excess = excess.dropna()  # common window for all candidates
    n_trials = ledger.n_trials(stage="research")
    if n_trials < excess.shape[1]:
        raise RuntimeError("ledger has fewer trials than study columns: ledger out of sync")
    trial_srs = excess.apply(stats.per_period_sharpe).to_numpy()
    monthly = stats.to_monthly(excess)
    assert isinstance(monthly, pd.DataFrame)

    candidates: dict[str, CandidateVerdict] = {}
    for name in excess.columns:
        x = excess[name]
        dsr, hurdle = stats.deflated_sharpe_ratio(x, trial_srs, n_trials)
        sub = stats.subperiod_sharpes(x, SUBPERIODS)
        if name == BASELINE_ID:
            diff = stats.SharpeDiffTest(0.0, 1.0, 0.0, 0.0)
        else:
            diff = stats.sharpe_difference_test(
                monthly[name], monthly[BASELINE_ID], block=6, n_boot=n_boot, seed=0
            )
        n_pos = int((sub > 0).sum())
        candidates[str(name)] = CandidateVerdict(
            sharpe=stats.per_period_sharpe(x) * np.sqrt(252),
            dsr=dsr,
            dsr_hurdle_annual=hurdle * np.sqrt(252),
            diff_vs_ew=diff.diff_annual,
            p_vs_ew=diff.p_value,
            ci_vs_ew=(diff.ci_low, diff.ci_high),
            subperiod_sharpes={str(k): float(v) for k, v in sub.items()},
            n_positive_subperiods=n_pos,
            passes_dsr=dsr >= DSR_THRESHOLD,
            passes_vs_ew=diff.p_value < P_THRESHOLD and diff.diff_annual > 0,
            passes_stability=n_pos >= MIN_POSITIVE_SUBPERIODS,
        )

    pbo = stats.probability_of_backtest_overfitting(monthly, n_blocks=16)
    wf = stats.walk_forward_selection(excess, min_years=3)
    picked = wf["picked_return"]
    stitched = stats.walk_forward_stitched(excess, min_years=3)
    window = excess.loc[stitched.index]
    ann = np.sqrt(252)
    wf_summary = {
        "sharpe_walk_forward_pick": stats.per_period_sharpe(stitched) * ann,
        "sharpe_ew_same_window": stats.per_period_sharpe(window[BASELINE_ID]) * ann,
        "sharpe_equal_blend_of_candidates": stats.per_period_sharpe(window.mean(axis=1)) * ann,
        "mean_picked_return": float(picked.mean()),
        "mean_median_candidate_return": float(wf["median_candidate_return"].mean()),
        "share_years_picked_beats_median": float((picked > wf["median_candidate_return"]).mean()),
    }

    validated = (
        [k for k, v in candidates.items() if v.passes_individual and k != BASELINE_ID]
        if pbo.pbo < PBO_THRESHOLD
        else []
    )
    pool = validated or [k for k in candidates if k != BASELINE_ID]
    best_dsr = max(candidates[k].dsr for k in pool)
    tied = [k for k in pool if candidates[k].dsr >= best_dsr - 0.01]
    selected = min(tied, key=lambda k: (COMPLEXITY.get(k, 99), -candidates[k].dsr))

    return StudyVerdict(
        n_trials=n_trials,
        pbo=pbo.pbo,
        pbo_prob_oos_loss=pbo.prob_oos_loss,
        pbo_median_oos_sharpe_of_is_best=float(np.median(pbo.is_best_oos_sharpe) * np.sqrt(12)),
        candidates=candidates,
        walk_forward=[
            {str(k): v for k, v in r.items()} for r in wf.reset_index().to_dict(orient="records")
        ],
        walk_forward_summary=wf_summary,
        validated=validated,
        selected=selected,
        selected_is_validated=bool(validated),
    )
