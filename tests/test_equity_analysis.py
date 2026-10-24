from __future__ import annotations

import pandas as pd
import pytest

from quantlab.equities.analysis import survivorship_drag


def test_drag_matches_preregistered_formula() -> None:
    cov = pd.DataFrame({"rank_date": ["2012-05-31", "2020-05-29"], "coverage_count": [0.65, 0.90]})
    idx = pd.bdate_range("2012-07-02", "2021-06-30")
    d = survivorship_drag(cov, idx)
    assert d.loc["2013-01-02"] * 252 == pytest.approx(2 * 0.30 * 0.35 / (2026.75 - 2012))
    assert d.loc["2020-12-01"] * 252 == pytest.approx(2 * 0.30 * 0.10 / (2026.75 - 2020))
    assert d.loc["2016-01-04"] == 0.0  # no coverage row for that year -> no drag assumed
