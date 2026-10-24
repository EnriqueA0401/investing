"""Point-in-time accounting: first-reported values, TTM identity, availability by filing date."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.equities.fundamentals import asof_panel, instant_records, merge_tags, ttm_records

T = pd.Timestamp


def _facts(
    rows: list[tuple[str, str | None, str, float, str]], tag: str = "NetIncomeLoss"
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cik": 1,
                "tag": tag,
                "unit": "USD",
                "start": T(s) if s else pd.NaT,
                "end": T(e),
                "val": v,
                "filed": T(f),
                "form": "10-Q",
                "accn": f"a{i}",
            }
            for i, (_, s, e, v, f) in enumerate(rows)
        ]
    )


FLOW = _facts(
    [
        ("FY19", "2019-01-01", "2019-12-31", 100.0, "2020-02-20"),
        ("Q1-19", "2019-01-01", "2019-03-31", 20.0, "2019-05-01"),
        ("FY20", "2020-01-01", "2020-12-31", 120.0, "2021-02-20"),
        ("Q1-20", "2020-01-01", "2020-03-31", 30.0, "2020-05-01"),
        ("H1-20", "2020-01-01", "2020-06-30", 55.0, "2020-08-01"),
        ("H1-19", "2019-01-01", "2019-06-30", 45.0, "2019-08-01"),
    ]
)


def test_ttm_identity_and_availability() -> None:
    rec = ttm_records(FLOW)
    q1 = rec[rec["end"] == T("2020-03-31")].iloc[0]
    assert q1["val"] == pytest.approx(100 + 30 - 20)
    assert q1["avail"] == T("2020-05-01")
    h1 = rec[rec["end"] == T("2020-06-30")].iloc[0]
    assert h1["val"] == pytest.approx(100 + 55 - 45)


def test_asof_uses_only_published_values() -> None:
    dates = pd.DatetimeIndex(["2020-04-30", "2020-05-01", "2020-05-29", "2020-08-31", "2021-03-31"])
    p = asof_panel(ttm_records(FLOW), dates, lag_days=1)[1]
    assert p.iloc[0] == pytest.approx(100)  # Q1 not yet filed -> FY19
    assert p.iloc[1] == pytest.approx(100)  # filed that day, usable next day
    assert p.iloc[2] == pytest.approx(110)
    assert p.iloc[3] == pytest.approx(110)  # H1 filed 2020-08-01 -> 100+55-45
    assert p.iloc[4] == pytest.approx(120)


def test_restatements_are_ignored() -> None:
    restated = pd.concat(
        [FLOW, _facts([("FY19r", "2019-01-01", "2019-12-31", 999.0, "2021-02-20")])],
        ignore_index=True,
    )
    merged = merge_tags(restated, ("NetIncomeLoss",), ["cik", "start", "end"])
    fy19 = merged[(merged["start"] == T("2019-01-01")) & (merged["end"] == T("2019-12-31"))]
    assert fy19["val"].tolist() == [100.0]


def test_stale_data_is_dropped_and_lookback() -> None:
    inst = _facts(
        [
            ("A18", None, "2018-12-31", 10.0, "2019-02-01"),
            ("A19", None, "2019-12-31", 12.0, "2020-02-01"),
        ],
        tag="Assets",
    )
    rec = instant_records(inst, ("Assets",))
    dates = pd.DatetimeIndex(["2020-03-31", "2021-12-31"])
    now = asof_panel(rec, dates)[1]
    ago = asof_panel(rec, dates, lookback_years=1)[1]
    assert now.iloc[0] == 12.0
    assert ago.iloc[0] == 10.0
    assert np.isnan(now.iloc[1])  # older than max_age_days


def test_asof_handles_object_dtype_records() -> None:
    rec = pd.DataFrame(
        {
            "cik": [1, 1],
            "end": [T("2019-12-31"), T("2020-12-31")],
            "val": [1.0, 2.0],
            "avail": [T("2020-02-01"), T("2021-02-01")],
        },
        dtype=object,
    )
    p = asof_panel(rec, pd.DatetimeIndex(["2020-06-30", "2021-06-30"]))[1]
    assert p.tolist() == [1.0, 2.0]
