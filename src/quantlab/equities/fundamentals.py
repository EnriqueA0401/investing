"""Point-in-time fundamentals from the EDGAR facts table.

Rules (each one exists to prevent look-ahead):
* A fact is usable ``lag_days`` after its ``filed`` date, never before.
* For each reporting period the FIRST filing wins (as originally reported);
  restated comparatives in later filings are ignored.
* Trailing-twelve-month flows use the vendor-standard identity
  TTM = FY(prev) + YTD(cur) - YTD(prior-year same period), built only from
  figures that were public when the current YTD was filed.
* Data older than ``max_age_days`` (by period end) is treated as missing.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
COST_TAGS = ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold")
EQUITY_TAGS = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
SIMPLE_FLOWS = {
    "net_income": ("NetIncomeLoss",),
    "op_income": ("OperatingIncomeLoss",),
    "cfo": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
}


def first_filed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep the earliest filing of each period (as originally reported)."""
    return df.sort_values("filed").drop_duplicates(keys, keep="first")


def merge_tags(facts: pd.DataFrame, tags: Iterable[str], keys: list[str]) -> pd.DataFrame:
    """One value per period from a precedence list of alternative tags."""
    tags = list(tags)
    sub = facts[facts["tag"].isin(tags) & (facts["unit"] == "USD")].copy()
    sub["prio"] = sub["tag"].map({t: i for i, t in enumerate(tags)})
    sub = first_filed(sub, [*keys, "tag"])
    return sub.sort_values("prio").drop_duplicates(keys, keep="first").drop(columns="prio")


def _durations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["start"]).copy()
    df["days"] = (df["end"] - df["start"]).dt.days
    return df


def ttm_records(flow: pd.DataFrame) -> pd.DataFrame:
    """TTM observations (cik, end, val, avail) from one flow's duration facts."""
    flow = _durations(flow)
    day = np.timedelta64(1, "D")
    out: list[pd.DataFrame] = []
    for cik, g in flow.groupby("cik", sort=False):
        start = g["start"].to_numpy()
        end = g["end"].to_numpy()
        val = g["val"].to_numpy(dtype=float)
        filed = g["filed"].to_numpy()
        days = g["days"].to_numpy()
        is_fy = (days >= 350) & (days <= 380)
        is_ytd = (days >= 80) & (days <= 290)
        ends, vals, avails = list(end[is_fy]), list(val[is_fy]), list(filed[is_fy])
        fy_idx = np.flatnonzero(is_fy)
        ytd_idx = np.flatnonzero(is_ytd)
        for j in ytd_idx if len(fy_idx) else []:
            # Previous fiscal year must end the day before this YTD starts (+-7 days).
            gap = np.abs((end[fy_idx] - (start[j] - day)) / day)
            k = fy_idx[int(gap.argmin())]
            if gap.min() > 7:
                continue
            prior = ytd_idx[
                (np.abs((start[ytd_idx] - start[k]) / day) <= 7)
                & (np.abs(days[ytd_idx] - days[j]) <= 10)
            ]
            if prior.size == 0:
                continue
            p = prior[0]
            ends.append(end[j])
            vals.append(val[k] + val[j] - val[p])
            avails.append(max(filed[j], filed[k], filed[p]))
        out.append(pd.DataFrame({"cik": cik, "end": ends, "val": vals, "avail": avails}))
    if not out:
        return pd.DataFrame(columns=["cik", "end", "val", "avail"])
    return pd.concat(out, ignore_index=True)


def instant_records(facts: pd.DataFrame, tags: Iterable[str]) -> pd.DataFrame:
    sub = merge_tags(facts, tags, ["cik", "end"])
    return sub[["cik", "end", "val", "filed"]].rename(columns={"filed": "avail"})


SHARE_SOURCES = (
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)


def shares_source_records(facts: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Per-filing share counts from one tag (cover-page dei shares are summed across
    the classes of a filing; other tags keep the first-filed value per period)."""
    sub = facts[(facts["tag"] == tag) & (facts["unit"] == "shares")]
    if tag != "EntityCommonStockSharesOutstanding":
        sub = first_filed(sub, ["cik", "end"])
    agg = sub.groupby(["cik", "accn"], as_index=False).agg(
        val=("val", "sum"), end=("end", "max"), avail=("filed", "min")
    )
    return agg[["cik", "end", "val", "avail"]]


def drop_scale_outliers(records: pd.DataFrame, max_ratio: float = 10.0) -> pd.DataFrame:
    """Drop values more than ``max_ratio`` x away from the company's median.

    Apply AFTER converting to today's split units, so genuine split-driven jumps
    are not mistaken for XBRL scaling errors (e.g. shares tagged x1000)."""
    if records.empty:
        return records
    logv = np.log10(records["val"].where(records["val"] > 0))
    med = logv.groupby(records["cik"]).transform("median")
    keep = ((logv - med).abs() <= np.log10(max_ratio)).fillna(False)
    out: pd.DataFrame = records[keep.to_numpy()]
    return out


def shares_records(facts: pd.DataFrame) -> pd.DataFrame:
    """All share-count records with a ``source`` rank (0 = preferred)."""
    parts = [
        shares_source_records(facts, tag).assign(source=i) for i, tag in enumerate(SHARE_SOURCES)
    ]
    return pd.concat(parts, ignore_index=True)


def float_records(facts: pd.DataFrame) -> pd.DataFrame:
    sub = facts[(facts["tag"] == "EntityPublicFloat") & (facts["unit"] == "USD")]
    per_filing = sub.groupby(["cik", "accn"], as_index=False).agg(
        val=("val", "sum"), end=("end", "max"), avail=("filed", "min")
    )
    return per_filing[["cik", "end", "val", "avail"]]


def asof_panel(
    records: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lag_days: int = 1,
    max_age_days: int = 456,
    lookback_years: int = 0,
) -> pd.DataFrame:
    """dates x cik panel: value of the latest-ending record public by each date.

    ``lookback_years=1`` instead returns the record whose period ended about one
    year before that latest record (for growth rates), also public by the date.
    """
    if records.empty:
        return pd.DataFrame(index=dates, dtype=float)
    rec = records.dropna(subset=["val"]).copy()
    # Normalise to datetime64[ns]: object-dtype columns (from empty groups) would make
    # numpy compare Timestamps with nanosecond integers.
    rec["end"] = pd.to_datetime(rec["end"]).astype("datetime64[ns]")
    rec["avail"] = pd.to_datetime(rec["avail"]).astype("datetime64[ns]")
    rec["val"] = rec["val"].astype(float)
    dates = pd.DatetimeIndex(pd.DatetimeIndex(dates).astype("datetime64[ns]"))
    rec["usable"] = rec["avail"] + pd.Timedelta(days=lag_days)
    cols: dict[int, np.ndarray] = {}
    for cik, grp in rec.groupby("cik", sort=False):
        g = grp.sort_values(["end", "usable"])
        usable = g["usable"].to_numpy()
        ends = g["end"].to_numpy()
        vals = g["val"].to_numpy(dtype=float)
        out = np.full(len(dates), np.nan)
        for i, t in enumerate(dates.to_numpy()):
            ok = usable <= t
            if not ok.any():
                continue
            e_ok, v_ok = ends[ok], vals[ok]
            latest_end = e_ok.max()
            if (t - latest_end) > np.timedelta64(max_age_days, "D"):
                continue
            if lookback_years == 0:
                out[i] = v_ok[np.flatnonzero(e_ok == latest_end)[-1]]
            else:
                target = latest_end - np.timedelta64(365 * lookback_years, "D")
                diff = np.abs((e_ok - target).astype("timedelta64[D]").astype(int))
                j = int(diff.argmin())
                if diff[j] <= 45:
                    out[i] = v_ok[j]
        cols[int(str(cik))] = out
    return pd.DataFrame(cols, index=dates)


def build_fundamental_panels(
    facts: pd.DataFrame, dates: pd.DatetimeIndex, ciks: Iterable[int], lag_days: int = 1
) -> dict[str, pd.DataFrame]:
    """Monthly PIT panels (dates x cik) for the fields used by the factors."""
    ciks = set(int(c) for c in ciks)
    f = facts[facts["cik"].isin(ciks)]
    keys = ["cik", "start", "end"]
    revenue = merge_tags(f, REVENUE_TAGS, keys)
    cost = merge_tags(f, COST_TAGS, keys)
    gp_direct = merge_tags(f, ("GrossProfit",), keys)
    gp_derived = revenue.merge(cost[[*keys, "val", "filed"]], on=keys, suffixes=("", "_c"))
    gp_derived["val"] = gp_derived["val"] - gp_derived["val_c"]
    gp_derived["filed"] = gp_derived[["filed", "filed_c"]].max(axis=1)
    gross = (
        pd.concat([gp_direct.assign(p=0), gp_derived[[*keys, "val", "filed"]].assign(p=1)])
        .sort_values("p")
        .drop_duplicates(keys)
    )
    flows = {"revenue": revenue, "gross_profit": gross}
    flows |= {k: merge_tags(f, tags, keys) for k, tags in SIMPLE_FLOWS.items()}

    panels: dict[str, pd.DataFrame] = {}
    for name, flow in flows.items():
        panels[f"{name}_ttm"] = asof_panel(ttm_records(flow), dates, lag_days)
    assets = instant_records(f, ("Assets",))
    panels["assets"] = asof_panel(assets, dates, lag_days)
    panels["assets_1y_ago"] = asof_panel(assets, dates, lag_days, lookback_years=1)
    panels["equity"] = asof_panel(instant_records(f, EQUITY_TAGS), dates, lag_days)
    panels["liabilities"] = asof_panel(instant_records(f, ("Liabilities",)), dates, lag_days)
    panels["shares"] = asof_panel(shares_records(f), dates, lag_days, max_age_days=200)
    return panels
