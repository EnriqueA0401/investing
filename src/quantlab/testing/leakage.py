"""Adversarial look-ahead detection for any transformation of ``MarketData``.

``assert_causal(fn, data)`` checks, at several cutoffs ``t``, that the output
of ``fn`` on rows <= t is identical whether ``fn`` sees:

1. the full dataset,
2. the dataset truncated at ``t`` (``data.asof(t)``), and
3. the dataset with every row after ``t`` replaced by random garbage.

Any dependence on the future (negative shifts, centred windows, full-sample
normalisation, fitting on the whole sample, ``bfill``...) makes (1)/(3)
differ from (2). Every Factor/feature must have a test calling this.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from quantlab.data.market_data import MarketData


class LookAheadError(AssertionError):
    pass


def _scramble_after(data: MarketData, t: pd.Timestamp, rng: np.random.Generator) -> MarketData:
    future = data.dates > t

    def noisy(df: pd.DataFrame, scale: float) -> pd.DataFrame:
        out = df.copy()
        n = int(future.sum())
        out.loc[future] = scale * np.exp(rng.normal(0.0, 1.0, size=(n, df.shape[1])))
        return out

    return MarketData(
        close=noisy(data.close, 50.0),
        total_return_index=noisy(data.total_return_index, 50.0),
        volume=noisy(data.volume, 1e6),
        asset_class=data.asset_class,
    )


def _as_frame(x: pd.DataFrame | pd.Series) -> pd.DataFrame:
    return x.to_frame() if isinstance(x, pd.Series) else x


def assert_causal(
    fn: Callable[[MarketData], pd.DataFrame | pd.Series],
    data: MarketData,
    *,
    cutoffs: Sequence[pd.Timestamp] | None = None,
    n_cutoffs: int = 8,
    seed: int = 0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    rng = np.random.default_rng(seed)
    if cutoffs is None:
        # Span the whole sample: some leaks (e.g. bfill) only bite near the start
        # of a series or around an asset's inception.
        positions = np.linspace(5, len(data.dates) - 2, n_cutoffs).astype(int)
        cutoffs = [pd.Timestamp(data.dates[p]) for p in positions]

    full = _as_frame(fn(data))
    for t in cutoffs:
        truncated = _as_frame(fn(data.asof(t)))
        scrambled = _as_frame(fn(_scramble_after(data, t, rng)))
        if truncated.index.max() > t:
            raise LookAheadError(f"output on data.asof({t.date()}) has rows after the cutoff")
        ref = truncated.loc[:t]
        for label, other in (("full data", full), ("future-scrambled data", scrambled)):
            cand = other.loc[:t].reindex(index=ref.index, columns=ref.columns)
            a, b = ref.to_numpy(dtype=float), cand.to_numpy(dtype=float)
            same_nan = np.isnan(a) == np.isnan(b)
            close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
            if not (same_nan.all() and close.all()):
                bad = np.argwhere(~(same_nan & close))[0]
                raise LookAheadError(
                    f"look-ahead detected at cutoff {t.date()}: output on {label} differs from "
                    f"truncated data at row {ref.index[bad[0]]}, column {ref.columns[bad[1]]}"
                )
