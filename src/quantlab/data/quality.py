"""Data-quality report that respects the holdout seal.

Return-based statistics are computed on the research window only. For the
holdout window the report shows *structural* checks (row counts, missing
values, non-positive prices) and nothing that reveals performance.
"""

from __future__ import annotations

import pandas as pd

from quantlab import metrics as m
from quantlab.config import ExperimentConfig
from quantlab.contracts import DataValidator
from quantlab.data.market_data import MarketData


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or ""), *cols]) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)


def research_stats(data: MarketData, config: ExperimentConfig) -> pd.DataFrame:
    research = data.asof(pd.Timestamp(config.split.research_end))
    rets = research.returns()
    adv = (research.close * research.volume).rolling(20).mean()
    rows = {}
    for t in research.tickers:
        r = rets[t].dropna()
        series = research.total_return_index[t]
        first = series.first_valid_index()
        live = series.loc[first:]
        hist = series.notna().astype(int).cumsum()
        investable_from = hist.index[(hist >= config.universe.min_history_days).to_numpy()]
        rows[t] = {
            "class": research.asset_class.get(t, "?"),
            "first_obs": str(live.index[0].date()),
            "investable_from": str(investable_from[0].date()) if len(investable_from) else "-",
            "n_obs": int(live.notna().sum()),
            "internal_nans": int(live.isna().sum()),
            "zero_vol_days": int(research.volume[t].loc[first:].eq(0).sum()),
            "cagr": m.annualized_return(r),
            "vol": m.annualized_vol(r),
            "max_dd": m.max_drawdown(r),
            "median_adv_musd": float(adv[t].median() / 1e6),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "ticker"
    return out


def holdout_structural(data: MarketData, config: ExperimentConfig) -> pd.DataFrame:
    hold = data.between(pd.Timestamp(config.split.holdout_start), None)
    tri = hold.total_return_index
    out = pd.DataFrame(
        {
            "n_rows": tri.notna().sum(),
            "missing": tri.isna().sum(),
            "non_positive": (tri <= 0).sum(),
        }
    )
    out.index.name = "ticker"
    return out


def data_quality_report(
    data: MarketData, config: ExperimentConfig, snapshot_id: str, validators: list[DataValidator]
) -> str:
    research = data.asof(pd.Timestamp(config.split.research_end))
    issues = [i for v in validators for i in v.validate(research)]
    stats = research_stats(data, config)
    hold = holdout_structural(data, config)
    cash = research.cash()
    issue_lines = [
        f"- `{i.severity.value}` **{i.check}** {i.ticker or ''}: {i.message}"
        + (f" (e.g. {', '.join(i.dates)})" if i.dates else "")
        for i in issues
    ] or ["- none"]
    return "\n".join(
        [
            "# Data quality report",
            "",
            f"- Snapshot: `{snapshot_id}`  ",
            f"- Sample: {data.dates[0].date()} → {data.end.date()} ({len(data.dates)} sessions)  ",
            f"- Research window used for statistics: up to {config.split.research_end}  ",
            f"- Holdout (from {config.split.holdout_start}): **structural checks only**, "
            "no return statistics, to keep it sealed.",
            "",
            "Total-return index reconstructed from split-adjusted close + ex-date "
            "distributions; cross-checked against vendor Adj Close at fetch time "
            "(fetch fails above 0.5% daily divergence).",
            "",
            "## Research window — per asset",
            "",
            _md_table(stats),
            "",
            f"Cash (3M T-bill, FRED DTB3, 1-session lag): CAGR {m.annualized_return(cash):.4f} "
            "over the research window.",
            "",
            "## Validator findings (research window)",
            "",
            *issue_lines,
            "",
            "## Holdout window — structural only",
            "",
            _md_table(hold),
            "",
        ]
    )
