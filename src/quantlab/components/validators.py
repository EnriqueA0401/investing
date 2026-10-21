"""Data-quality validators. Errors abort a run; warnings are logged and recorded."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.contracts import ComponentParams, DataValidator, Severity, ValidationIssue
from quantlab.data.market_data import MarketData
from quantlab.registry import register


def _fmt(dates: pd.Index, limit: int = 5) -> tuple[str, ...]:
    return tuple(str(pd.Timestamp(d).date()) for d in dates[:limit])


@register("validator", "basic_integrity")
class BasicIntegrity(DataValidator):
    """Structural checks: empty series, non-positive prices, internal gaps,
    implausible jumps, stale prices, negative volume."""

    class Params(ComponentParams):
        max_abs_daily_return: float = 0.25
        max_internal_gap_days: int = 3
        max_stale_days: int = 5
        max_abs_daily_cash_return: float = 0.001

    params: Params

    def validate(self, data: MarketData) -> list[ValidationIssue]:
        p = self.params
        issues: list[ValidationIssue] = []
        tri, rets = data.total_return_index, data.returns()
        for t in data.tickers:
            s = tri[t]
            if s.notna().sum() == 0:
                issues.append(ValidationIssue("empty", Severity.ERROR, "no observations", t))
                continue
            live = s.loc[s.first_valid_index() :]
            bad = live[live <= 0].index
            if len(bad):
                issues.append(
                    ValidationIssue("non_positive", Severity.ERROR, "price <= 0", t, _fmt(bad))
                )
            gaps = live[live.isna()].index
            if len(gaps):
                run = int(live.isna().astype(int).groupby(live.notna().cumsum()).sum().max())
                sev = Severity.ERROR if run > p.max_internal_gap_days else Severity.WARNING
                issues.append(
                    ValidationIssue(
                        "internal_gap",
                        sev,
                        f"{len(gaps)} missing (longest run {run})",
                        t,
                        _fmt(gaps),
                    )
                )
            jumps = rets[t][rets[t].abs() > p.max_abs_daily_return].index
            if len(jumps):
                issues.append(
                    ValidationIssue(
                        "extreme_return",
                        Severity.WARNING,
                        f"|r|>{p.max_abs_daily_return}",
                        t,
                        _fmt(jumps),
                    )
                )
            unchanged = live.diff().eq(0).astype(int)
            stale_run = int(unchanged.groupby((unchanged == 0).cumsum()).sum().max() or 0)
            if stale_run >= p.max_stale_days:
                issues.append(
                    ValidationIssue("stale", Severity.WARNING, f"{stale_run} unchanged days", t)
                )
            zero_vol = data.volume[t].loc[live.index].eq(0)
            if zero_vol.any():
                issues.append(
                    ValidationIssue(
                        "zero_volume",
                        Severity.WARNING,
                        f"{int(zero_vol.sum())} sessions with zero volume",
                        t,
                        _fmt(zero_vol[zero_vol].index),
                    )
                )
            vol = data.volume[t].to_numpy(dtype=float)
            if np.nanmin(vol, initial=0.0) < 0:
                issues.append(ValidationIssue("negative_volume", Severity.ERROR, "volume < 0", t))
        if data.cash_return is not None:
            cash = data.cash_return
            if cash.isna().any():
                issues.append(ValidationIssue("cash_gap", Severity.ERROR, "NaN cash returns"))
            extreme = cash[cash.abs() > p.max_abs_daily_cash_return].index
            if len(extreme):
                issues.append(
                    ValidationIssue(
                        "cash_extreme",
                        Severity.ERROR,
                        "implausible daily cash return",
                        None,
                        _fmt(extreme),
                    )
                )
        return issues
