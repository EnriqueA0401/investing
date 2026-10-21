"""Overfitting-aware statistics: PSR/DSR, PBO (CSCV), block bootstrap, walk-forward selection.

References
----------
* Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR).
* Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".
* Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of Backtest
  Overfitting" (CSCV / PBO).
* Politis & Romano (1992) circular block bootstrap.

Sharpe ratios inside PSR/DSR are *per period* (not annualised), as in the papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


def per_period_sharpe(r: pd.Series | np.ndarray) -> float:
    x = np.asarray(r, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe_ratio(excess: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark) given the sample SR, skew and kurtosis of ``excess``."""
    x = excess.dropna().to_numpy(dtype=float)
    n = len(x)
    sr = per_period_sharpe(x)
    skew = float(st.skew(x))
    kurt = float(st.kurtosis(x, fisher=False))  # raw kurtosis (normal = 3)
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    return float(st.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(trial_sharpes: np.ndarray, n_trials: int) -> float:
    """Expected max per-period SR among ``n_trials`` skill-less trials (False Strategy Thm)."""
    if n_trials <= 1:
        return 0.0
    v = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    z1 = st.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = st.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(v) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    excess: pd.Series, trial_sharpes: np.ndarray, n_trials: int
) -> tuple[float, float]:
    """(DSR, SR*) where SR* is the per-period hurdle implied by the trial count."""
    hurdle = expected_max_sharpe(trial_sharpes, n_trials)
    return probabilistic_sharpe_ratio(excess, hurdle), hurdle


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    is_best_oos_sharpe: np.ndarray
    prob_oos_loss: float
    n_combinations: int


def probability_of_backtest_overfitting(returns: pd.DataFrame, n_blocks: int = 16) -> PBOResult:
    """CSCV: split T rows into ``n_blocks`` contiguous blocks; for every half/half
    combination pick the in-sample best (Sharpe) and record its out-of-sample rank.
    PBO = share of combinations where the IS-best ranks below the OOS median."""
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    m = returns.dropna().to_numpy(dtype=float)
    t, n = m.shape
    if n < 2:
        raise ValueError("need at least two strategies")
    blocks = np.array_split(np.arange(t), n_blocks)
    logits, best_oos = [], []
    for is_blocks in combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in range(n_blocks) if b not in is_blocks])
        sr_is = _col_sharpe(m[is_idx])
        sr_oos = _col_sharpe(m[oos_idx])
        best = int(np.nanargmax(sr_is))
        rank = st.rankdata(sr_oos)[best]  # 1 = worst, n = best
        omega = rank / (n + 1.0)
        logits.append(np.log(omega / (1.0 - omega)))
        best_oos.append(sr_oos[best])
    lg = np.asarray(logits)
    bo = np.asarray(best_oos)
    return PBOResult(
        pbo=float((lg <= 0).mean()),
        logits=lg,
        is_best_oos_sharpe=bo,
        prob_oos_loss=float((bo < 0).mean()),
        n_combinations=len(lg),
    )


def _col_sharpe(m: np.ndarray) -> np.ndarray:
    sd = m.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, m.mean(axis=0) / sd, np.nan)


def circular_block_bootstrap_indices(
    n: int, block: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


@dataclass(frozen=True)
class SharpeDiffTest:
    diff_annual: float
    p_value: float
    ci_low: float
    ci_high: float


def sharpe_difference_test(
    a_excess_monthly: pd.Series,
    b_excess_monthly: pd.Series,
    block: int = 6,
    n_boot: int = 10_000,
    seed: int = 0,
) -> SharpeDiffTest:
    """One-sided test of H0: SR_a <= SR_b with a paired circular block bootstrap.

    p-value from the bootstrap distribution centred on the observed difference
    (so it approximates the null), as in Ledoit & Wolf (2008)."""
    joined = pd.concat([a_excess_monthly, b_excess_monthly], axis=1).dropna().to_numpy()
    n = len(joined)
    rng = np.random.default_rng(seed)
    idx = circular_block_bootstrap_indices(n, block, n_boot, rng)
    ann = np.sqrt(12.0)
    obs = (per_period_sharpe(joined[:, 0]) - per_period_sharpe(joined[:, 1])) * ann
    sample = joined[idx]  # (n_boot, n, 2)
    mu = sample.mean(axis=1)
    sd = sample.std(axis=1, ddof=1)
    boot = (mu[:, 0] / sd[:, 0] - mu[:, 1] / sd[:, 1]) * ann
    p = (1.0 + np.sum(boot - obs >= obs)) / (n_boot + 1.0)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return SharpeDiffTest(float(obs), float(p), float(lo), float(hi))


def to_monthly(daily: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return (1.0 + daily).groupby(pd.Grouper(freq="ME")).prod() - 1.0


def subperiod_sharpes(excess_daily: pd.Series, periods: list[tuple[str, str]]) -> pd.Series:
    out = {}
    for start, end in periods:
        x = excess_daily.loc[start:end].dropna()
        out[f"{start[:4]}-{end[:4]}"] = (
            per_period_sharpe(x) * np.sqrt(252) if len(x) > 20 else np.nan
        )
    return pd.Series(out)


def walk_forward_selection(excess_daily: pd.DataFrame, min_years: int = 3) -> pd.DataFrame:
    """Each calendar year, pick the candidate with the best expanding-window Sharpe
    to date and record its *next-year* excess return. Measures whether 'pick the
    backtest winner' is itself a strategy that works out of sample."""
    idx = pd.DatetimeIndex(excess_daily.index)
    years = sorted(set(idx.year))
    rows = []
    for y in years[min_years:]:
        past = excess_daily.loc[: f"{y - 1}-12-31"]
        cur = excess_daily.loc[f"{y}-01-01" : f"{y}-12-31"]
        if cur.empty:
            continue
        sr = past.apply(per_period_sharpe)
        pick = sr.idxmax()
        picked_ret = float(np.prod(1.0 + cur[pick].to_numpy(dtype=float)) - 1.0)
        rows.append(
            {
                "year": y,
                "picked": str(pick),
                "picked_return": picked_ret,
                "median_candidate_return": float(((1 + cur).prod() - 1).median()),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def walk_forward_stitched(excess_daily: pd.DataFrame, min_years: int = 3) -> pd.Series:
    """Daily excess returns of the 'each year hold last year's Sharpe winner' policy."""
    wf = walk_forward_selection(excess_daily, min_years)
    parts = [excess_daily.loc[f"{y}-01-01" : f"{y}-12-31", row.picked] for y, row in wf.iterrows()]
    out: pd.Series = pd.concat(parts).rename("walk_forward_pick")
    return out
