"""Point-in-time large-cap universe ("Russell-1000-like") and survivorship coverage.

Reconstitution mimics Russell's annual cycle: rank by market cap on the last
trading day of May (using only shares filed by then), membership effective
from July through the following June. Market cap = cover-page shares (in
today's split units) x split-adjusted close.

Survivorship: prices exist only for companies still listed today. To measure
what that removes, the *true* top-N is approximated from EDGAR's
``EntityPublicFloat`` across ALL filers (dead ones included), and compared with
the names we can price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def month_end_dates(trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading date of each calendar month present in the data."""
    s = pd.Series(trading_dates, index=trading_dates)
    return pd.DatetimeIndex(s.groupby([trading_dates.year, trading_dates.month]).max().to_numpy())


def monthly_market_cap(shares_today_units: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    px = close.reindex(shares_today_units.index)
    return shares_today_units.reindex(columns=px.columns) * px


def reconstitute(
    mcap: pd.DataFrame,
    close: pd.DataFrame,
    size: int = 1000,
    min_price: float = 1.0,
    rank_month: int = 5,
) -> pd.DataFrame:
    """Monthly bool membership (month-end index x tickers)."""
    members = pd.DataFrame(False, index=mcap.index, columns=mcap.columns)
    rank_dates = [d for d in mcap.index if d.month == rank_month]
    px = close.reindex(mcap.index)
    for rd in rank_dates:
        caps = mcap.loc[rd].where(px.loc[rd] >= min_price).dropna()
        top = caps.sort_values(ascending=False).index[:size]
        eff_start = pd.Timestamp(year=rd.year, month=7, day=1)
        eff_end = pd.Timestamp(year=rd.year + 1, month=6, day=30)
        rows = (members.index >= eff_start) & (members.index <= eff_end)
        members.loc[rows, :] = False
        members.loc[rows, top] = True
    return members


def coverage_report(
    floats: pd.DataFrame,
    priced_ciks: set[int],
    names: pd.Series,
    rank_dates: list[pd.Timestamp],
    *,
    size: int = 1000,
    max_age_days: int = 456,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per rank date: share of the float-ranked true top-N that we can price.

    Returns (summary per year, list of missing names per year)."""
    rows, missing = [], []
    rec = floats.dropna(subset=["val"])
    for rd in rank_dates:
        live = rec[(rec["avail"] < rd) & (rec["end"] >= rd - pd.Timedelta(days=max_age_days))]
        latest = live.sort_values("end").groupby("cik").tail(1)
        top = latest.sort_values("val", ascending=False).head(size)
        covered = top["cik"].isin(priced_ciks)
        rows.append(
            {
                "rank_date": rd.date(),
                "true_top_n": len(top),
                "covered_n": int(covered.sum()),
                "coverage_count": float(covered.mean()) if len(top) else np.nan,
                "coverage_float_weight": float(top.loc[covered, "val"].sum() / top["val"].sum())
                if len(top)
                else np.nan,
            }
        )
        miss = top.loc[~covered].head(25)
        for _, r in miss.iterrows():
            missing.append(
                {
                    "rank_date": rd.date(),
                    "cik": int(r["cik"]),
                    "name": names.get(int(r["cik"]), ""),
                    "public_float_busd": r["val"] / 1e9,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(missing)
