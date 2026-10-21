"""Pure performance-metric functions on daily simple-return series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def as_float(x: Any) -> float:
    """pandas reductions are typed as broad scalar unions; coerce explicitly."""
    return float(x)


def _clean(r: pd.Series) -> pd.Series:
    r = r.dropna()
    if r.empty:
        raise ValueError("empty return series")
    return r


def annualized_return(r: pd.Series) -> float:
    r = _clean(r)
    growth = as_float((1 + r).prod())
    return float(growth ** (PERIODS_PER_YEAR / len(r)) - 1)


def annualized_vol(r: pd.Series) -> float:
    return float(_clean(r).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def sharpe_ratio(r: pd.Series, rf_daily: float = 0.0) -> float:
    ex = _clean(r) - rf_daily
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else float("nan")


def sortino_ratio(r: pd.Series, rf_daily: float = 0.0) -> float:
    ex = _clean(r) - rf_daily
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(PERIODS_PER_YEAR)) if downside > 0 else float("nan")


def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1 + _clean(r)).cumprod()
    return wealth / wealth.cummax().clip(lower=1.0) - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def calmar_ratio(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return annualized_return(r) / abs(mdd) if mdd < 0 else float("nan")


def monthly_hit_rate(r: pd.Series) -> float:
    m = (1 + _clean(r)).groupby(pd.Grouper(freq="ME")).prod() - 1
    return float((m > 0).mean())


def tracking_error(r: pd.Series, b: pd.Series) -> float:
    active = (_clean(r) - b).dropna()
    return float(active.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def information_ratio(r: pd.Series, b: pd.Series) -> float:
    active = (_clean(r) - b).dropna()
    sd = active.std(ddof=1)
    return float(active.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else float("nan")


def beta(r: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([r, b], axis=1).dropna()
    var = joined.iloc[:, 1].var(ddof=1)
    return as_float(joined.cov().iloc[0, 1]) / var if var > 0 else float("nan")
