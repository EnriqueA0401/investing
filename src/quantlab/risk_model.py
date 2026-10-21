"""Covariance estimation and weight post-processing shared by constructors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from quantlab.data.market_data import MarketData


def trailing_returns(
    history: MarketData, tickers: list[str], window: int, min_rows: int = 60
) -> pd.DataFrame:
    """Last ``window`` daily returns for ``tickers``, keeping only rows where all are observed.

    A newly investable asset can shorten the common window by a few rows; fewer
    than ``min_rows`` complete rows is an error.
    """
    rets = history.returns()[tickers].iloc[-window:].dropna(how="any")
    if len(rets) < min_rows:
        raise ValueError(f"only {len(rets)} complete return rows for {tickers}")
    return rets


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance of daily returns (daily units)."""
    lw = LedoitWolf().fit(returns.to_numpy(dtype=float))
    cov = np.asarray(lw.covariance_, dtype=float)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def cap_and_redistribute(weights: pd.Series, cap: float, budget: float = 1.0) -> pd.Series:
    """Scale to ``budget``, then cap each weight and redistribute the excess pro-rata
    over uncapped names (water-filling). If every name is capped, the residual
    stays in cash, so the result can sum to less than ``budget``."""
    w = weights.clip(lower=0.0).astype(float)
    if w.sum() <= 0:
        return w * 0.0
    w = w / w.sum() * budget
    for _ in range(len(w) + 1):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over & (w < cap - 1e-12)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w.clip(upper=cap)


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame, periods: int = 252) -> float:
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    return float(np.sqrt(w @ cov.to_numpy() @ w * periods))
