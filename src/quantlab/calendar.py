"""Rebalance schedule derived from the observed trading calendar."""

from __future__ import annotations

from typing import Literal, cast

import pandas as pd

Frequency = Literal["W", "ME", "QE"]


def _period_key(dates: pd.DatetimeIndex, freq: Frequency) -> pd.Index:
    if freq == "W":
        iso = dates.isocalendar()
        return cast(pd.Index, pd.Index(list(zip(iso["year"], iso["week"], strict=True))))
    if freq == "ME":
        return pd.Index(list(zip(dates.year, dates.month, strict=True)))
    return pd.Index(list(zip(dates.year, dates.quarter, strict=True)))


def rebalance_dates(dates: pd.DatetimeIndex, freq: Frequency) -> pd.DatetimeIndex:
    """Last trading date of each period.

    A date qualifies only if the *next* observed trading date falls in a new
    period, so the trailing, possibly incomplete period is never treated as
    closed. (Exchange holiday calendars are published in advance, so knowing
    that ``d`` is the last session of its month is not look-ahead.)
    """
    if len(dates) < 2:
        return pd.DatetimeIndex([])
    keys = _period_key(dates, freq)
    is_last = [keys[i] != keys[i + 1] for i in range(len(dates) - 1)] + [False]
    return dates[is_last]


def shift_trading_days(dates: pd.DatetimeIndex, t: pd.Timestamp, lag: int) -> pd.Timestamp | None:
    """Trading date ``lag`` sessions after ``t`` (``t`` must be a trading date)."""
    pos = dates.get_loc(t)
    if not isinstance(pos, int):
        raise ValueError(f"{t} is not a unique trading date")
    target = pos + lag
    return pd.Timestamp(dates[target]) if target < len(dates) else None
