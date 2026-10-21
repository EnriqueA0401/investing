"""Purged cross-validation for time-indexed samples with overlapping labels.

López de Prado (2018), AFML ch. 7 & 12. Each sample ``i`` has an observation
time ``t0[i]`` (features known) and a label end ``t1[i]`` (label known). A
training sample is *purged* if its label interval [t0, t1] overlaps any test
interval, and *embargoed* if it starts within ``embargo`` of the end of a test
block. Never use ordinary k-fold on financial time series: overlapping labels
leak test information into training.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[int, ...]


def _groups(t0: pd.Series, n_groups: int) -> np.ndarray:
    """Contiguous groups over *unique* observation times (all assets of a date stay together)."""
    times = np.sort(t0.unique())
    bounds = np.array_split(times, n_groups)
    group_of_time = {t: g for g, chunk in enumerate(bounds) for t in chunk}
    return t0.map(group_of_time).to_numpy()


def _purge(
    t0: pd.Series, t1: pd.Series, test_mask: np.ndarray, embargo: pd.Timedelta
) -> np.ndarray:
    train = ~test_mask
    test_t0, test_t1 = t0[test_mask], t1[test_mask]
    # Contiguous test blocks: purge against each block's [min t0, max t1 + embargo].
    groups = pd.Series(test_mask).ne(pd.Series(test_mask).shift()).cumsum()[test_mask]
    for _, idx in groups.groupby(groups).groups.items():
        lo = test_t0.loc[idx].min()
        hi = test_t1.loc[idx].max()
        overlap = (t0 <= hi + embargo) & (t1 >= lo)
        train &= ~overlap.to_numpy()
    return train


def combinatorial_purged_cv(
    t0: pd.Series, t1: pd.Series, n_groups: int = 6, k_test: int = 2, embargo_days: int = 21
) -> Iterator[Split]:
    """CPCV: every combination of ``k_test`` of ``n_groups`` groups is a test set.

    ``t0``/``t1`` must be sorted by ``t0`` and share a RangeIndex.
    """
    if not t0.is_monotonic_increasing:
        raise ValueError("samples must be sorted by observation time")
    if (t1 < t0).any():
        raise ValueError("label end before observation time")
    g = _groups(t0, n_groups)
    embargo = pd.Timedelta(days=embargo_days)
    for test_groups in combinations(range(n_groups), k_test):
        test_mask = np.isin(g, test_groups)
        train_mask = _purge(t0, t1, test_mask, embargo)
        yield Split(np.flatnonzero(train_mask), np.flatnonzero(test_mask), test_groups)


def cpcv_paths(n_groups: int, k_test: int) -> list[list[tuple[int, ...]]]:
    """Assign each (split, group) to a backtest path: path j uses, for every group,
    the j-th split in which that group is tested."""
    splits = list(combinations(range(n_groups), k_test))
    per_group = {g: [s for s in splits if g in s] for g in range(n_groups)}
    n_paths = len(per_group[0])
    return [[per_group[g][j] for g in range(n_groups)] for j in range(n_paths)]
