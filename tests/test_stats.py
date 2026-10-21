from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import stats
from quantlab.cv import combinatorial_purged_cv, cpcv_paths


def _noise(n: int, k: int, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-01", periods=n)
    return pd.DataFrame(rng.normal(drift, 0.01, (n, k)), index=idx)


def test_psr_calibration() -> None:
    zero = _noise(5000, 1, drift=0.0)[0]
    strong = _noise(5000, 1, drift=0.002)[0]
    assert 0.05 < stats.probabilistic_sharpe_ratio(zero) < 0.95
    assert stats.probabilistic_sharpe_ratio(strong) > 0.999


def test_expected_max_sharpe_grows_with_trials() -> None:
    srs = np.random.default_rng(1).normal(0, 0.02, 50)
    assert stats.expected_max_sharpe(srs, 1) == 0.0
    assert 0 < stats.expected_max_sharpe(srs, 10) < stats.expected_max_sharpe(srs, 100)


def test_dsr_penalises_many_trials() -> None:
    m = _noise(2500, 30, seed=2)
    srs = m.apply(stats.per_period_sharpe).to_numpy()
    best = m[m.apply(stats.per_period_sharpe).idxmax()]
    dsr, hurdle = stats.deflated_sharpe_ratio(best, srs, n_trials=30)
    assert hurdle > 0
    assert dsr < stats.probabilistic_sharpe_ratio(best)
    assert dsr < 0.95  # best-of-30 pure noise must not pass


def test_pbo_noise_is_near_half_and_real_edge_is_low() -> None:
    noise = stats.probability_of_backtest_overfitting(_noise(2000, 10, seed=3), n_blocks=10)
    assert 0.25 < noise.pbo < 0.75
    m = _noise(2000, 10, seed=4)
    m[0] += 0.0015  # one genuinely better strategy
    edge = stats.probability_of_backtest_overfitting(m, n_blocks=10)
    assert edge.pbo < 0.1
    assert edge.n_combinations == 252


def test_sharpe_difference_test() -> None:
    m = stats.to_monthly(_noise(5000, 2, seed=5))
    same = stats.sharpe_difference_test(m[0], m[0], n_boot=500)
    assert same.diff_annual == 0.0
    assert same.p_value == 1.0
    better = m[0] + 0.01
    res = stats.sharpe_difference_test(better, m[1], n_boot=2000)
    assert res.diff_annual > 0
    assert res.p_value < 0.01
    assert res.ci_low < res.diff_annual < res.ci_high


def test_block_bootstrap_indices_are_circular_blocks() -> None:
    idx = stats.circular_block_bootstrap_indices(10, 3, 5, np.random.default_rng(0))
    assert idx.shape == (5, 10)
    assert ((np.diff(idx[:, :3], axis=1) % 10) == 1).all()


def test_walk_forward_selection() -> None:
    m = _noise(252 * 8, 4, seed=6)
    m[2] += 0.001
    wf = stats.walk_forward_selection(m, min_years=3)
    assert len(wf) == 5
    assert (wf["picked"] == "2").mean() >= 0.8


def _panel(n_dates: int = 60, n_assets: int = 5) -> tuple[pd.Series, pd.Series]:
    dates = pd.date_range("2010-01-31", periods=n_dates, freq="ME")
    t0 = pd.Series(np.repeat(dates, n_assets))
    t1 = pd.Series(np.repeat(dates + pd.offsets.MonthEnd(1), n_assets))
    return t0, t1


def test_cpcv_purges_overlapping_labels() -> None:
    t0, t1 = _panel()
    splits = list(combinatorial_purged_cv(t0, t1, n_groups=6, k_test=2, embargo_days=21))
    assert len(splits) == 15
    for s in splits:
        assert not set(s.train) & set(s.test)
        for lo_t0, hi_t1 in zip(t0[s.test], t1[s.test], strict=True):
            overlap = (t0[s.train] <= hi_t1) & (t1[s.train] >= lo_t0)
            assert not overlap.any()


def test_cpcv_rejects_unsorted() -> None:
    t0, t1 = _panel()
    with pytest.raises(ValueError, match="sorted"):
        list(
            combinatorial_purged_cv(
                t0[::-1].reset_index(drop=True), t1[::-1].reset_index(drop=True)
            )
        )


def test_cpcv_paths_cover_every_group_once() -> None:
    paths = cpcv_paths(6, 2)
    assert len(paths) == 5
    for path in paths:
        assert all(g in split for g, split in enumerate(path))
