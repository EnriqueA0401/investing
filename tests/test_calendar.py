from __future__ import annotations

import pandas as pd

from quantlab.calendar import rebalance_dates, shift_trading_days


def test_month_ends_are_last_sessions_and_trailing_month_excluded() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-04-10")
    out = rebalance_dates(dates, "ME")
    assert list(out) == [pd.Timestamp(d) for d in ("2024-01-31", "2024-02-29", "2024-03-29")]


def test_month_end_respects_missing_sessions() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-03-05").drop(pd.Timestamp("2024-01-31"))
    assert pd.Timestamp("2024-01-30") in rebalance_dates(dates, "ME")


def test_shift_trading_days() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    assert shift_trading_days(dates, dates[0], 2) == dates[2]
    assert shift_trading_days(dates, dates[4], 1) is None
