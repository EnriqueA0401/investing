"""Link predecessor CIKs to the CIK a company files under today.

Reorganizations (new holding company, redomicile) give a listed company a new
CIK whose EDGAR history starts at the reorganization (e.g. Medtronic, Disney,
Exxon). Without linking, the company drops out of the historical universe and
the survivorship coverage measure counts it as 'dead'. A predecessor is linked
when it stopped filing close to when the successor started and the names are
identical after removing legal suffixes. Fuzzy thresholds (0.85-0.95) were tried
and rejected: generic words ('Pharmaceuticals', 'Acquisition Corp II/III') produced
false links such as Atheros -> Charter Communications.
Name-changing reorganizations (Google -> Alphabet) are not caught; this is a
documented limitation, not silently patched.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

STOP = {
    "inc", "corp", "corporation", "co", "company", "ltd", "plc", "holdings", "holding",
    "the", "group", "new", "llc", "lp", "sa", "nv", "ag", "de", "del",
}  # fmt: skip


def _norm(name: str) -> str:
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t and t not in STOP]
    return "".join(tokens)


def name_similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def link_predecessors(
    facts: pd.DataFrame,
    names: pd.Series,
    current_ciks: set[int],
    *,
    min_similarity: float = 0.999,
    gap_days: tuple[int, int] = (-200, 450),
    since: str = "2009-06-01",
) -> dict[int, int]:
    """{predecessor_cik: successor_cik} for successors listed today."""
    span = facts.groupby("cik")["filed"].agg(["min", "max"])
    new = span[(span.index.isin(current_ciks)) & (span["min"] >= pd.Timestamp(since))]
    dead = span[~span.index.isin(current_ciks)]
    links: dict[int, int] = {}
    for succ, row in new.iterrows():
        gap = (row["min"] - dead["max"]).dt.days
        near = dead[(gap >= gap_days[0]) & (gap <= gap_days[1])]
        best, best_sim = None, min_similarity
        succ_name = str(names.get(int(str(succ)), ""))
        for pred in near.index:
            sim = name_similarity(str(names.get(int(str(pred)), "")), succ_name)
            if sim >= best_sim:
                best, best_sim = int(str(pred)), sim
        if best is not None:
            links[best] = int(str(succ))
    return links


def relabel(facts: pd.DataFrame, links: dict[int, int]) -> pd.DataFrame:
    """Assign predecessor facts to the successor for periods before the successor's
    own first reported period end."""
    if not links:
        return facts
    first_end = facts.groupby("cik")["end"].min()
    out = facts.copy()
    for pred, succ in links.items():
        cutoff = first_end.get(succ, pd.Timestamp.max)
        mask = (out["cik"] == pred) & (out["end"] < cutoff)
        out.loc[mask, "cik"] = succ
    return out
