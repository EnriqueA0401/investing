from __future__ import annotations

import pandas as pd

from quantlab.equities.linking import link_predecessors, name_similarity, relabel

T = pd.Timestamp


def test_name_similarity_ignores_legal_suffixes() -> None:
    assert name_similarity("Exxon Mobil Corporation", "ExxonMobil Holdings Corp") == 1.0
    assert name_similarity("WALT DISNEY CO/", "Walt Disney Co") == 1.0
    assert name_similarity("GOOGLE INC.", "Alphabet Inc.") < 0.5


def _facts(cik: int, first: str, last: str) -> list[dict[str, object]]:
    return [
        {
            "cik": cik,
            "tag": "Assets",
            "end": T(d),
            "filed": T(d) + pd.Timedelta(days=40),
            "val": 1.0,
        }
        for d in pd.date_range(first, last, freq="QE")
    ]


def test_links_reorganized_company_and_relabels_history() -> None:
    facts = pd.DataFrame(
        _facts(1, "2010-03-31", "2019-03-31")
        + _facts(2, "2019-03-31", "2024-12-31")
        + _facts(3, "2010-03-31", "2015-12-31")
    )
    names = pd.Series({1: "WALT DISNEY CO/", 2: "Walt Disney Co", 3: "Unrelated Corp"})
    links = link_predecessors(facts, names, current_ciks={2})
    assert links == {1: 2}
    out = relabel(facts, links)
    assert out.loc[out["cik"] == 2, "end"].min() == T("2010-03-31")
    assert (out["cik"] == 1).sum() == 1  # only the overlapping period stays with the predecessor
