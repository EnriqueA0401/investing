"""Out-of-sample (holdout) comparison of the study portfolios against a single-asset
benchmark (default SPY, buy-and-hold total return, no costs).

Descriptive only: computed after the single holdout evaluation, it cannot feed
back into any research decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from quantlab import metrics as m
from quantlab import stats
from quantlab.data.market_data import MarketData
from quantlab.report_figures import GRID, SERIES, SURFACE, TEXT, TEXT_2, _style, _x, _year_ticks

BENCH_COLOR = "#3b3a36"  # recessive neutral reference line
LABELS = {
    "H4-IV-TS12": "H4-IV-TS12 (seleccionada)",
    "B-EW": "Equal-weight (propuesta)",
    "60-40": "60/40",
}


def holdout_streams(
    holdout_dir: Path, data: MarketData, bench: str
) -> tuple[pd.DataFrame, pd.Series]:
    h = pd.read_parquet(holdout_dir / "returns_net.parquet")
    df = h[list(LABELS)].copy()
    df[bench] = data.returns()[bench].reindex(h.index)
    if df[bench].isna().any():
        raise ValueError(f"{bench} has gaps in the holdout window")
    return df, h["cash"]


def summary(df: pd.DataFrame, cash: pd.Series, bench: str) -> pd.DataFrame:
    mo = cast(pd.DataFrame, stats.to_monthly(df))
    b = mo[bench].to_numpy()
    up, down = b > 0, b < 0
    rows = {}
    for c in df:
        r = df[c]
        mc = mo[c].to_numpy()
        dd_series = m.drawdown(r)
        rows[c] = {
            "retorno_total": float(np.prod(1 + r.to_numpy()) - 1),
            "cagr": m.annualized_return(r),
            "volatilidad": m.annualized_vol(r),
            "sharpe": m.sharpe_ratio(r, cash),
            "sortino": m.sortino_ratio(r, cash),
            "max_drawdown": float(dd_series.min()),
            "calmar": m.calmar_ratio(r),
            "corr_diaria": float(r.corr(df[bench])),
            "corr_mensual": float(mo[c].corr(mo[bench])),
            "beta": m.beta(r, df[bench]),
            "up_capture": float(mc[up].mean() / b[up].mean()),
            "down_capture": float(mc[down].mean() / b[down].mean()),
            "meses_positivos": float((mo[c] > 0).mean()),
        }
    return pd.DataFrame(rows).T


def benchmark_drawdowns(df: pd.DataFrame, bench: str, threshold: float = -0.05) -> pd.DataFrame:
    """Every benchmark peak-to-trough fall deeper than ``threshold`` and each series' return
    over the same window (peak close to trough close)."""
    dd = m.drawdown(df[bench])
    in_dd = dd < 0
    episode = (~in_dd).cumsum()[in_dd]
    rows = []
    for _, idx in episode.groupby(episode).groups.items():
        seg = dd.loc[idx]
        if seg.min() > threshold:
            continue
        start = pd.Timestamp(seg.index[0])
        trough = pd.Timestamp(seg.index[int(np.argmin(seg.to_numpy()))])
        window = df.loc[start:trough]
        pos = int(np.flatnonzero(df.index == start)[0])
        peak = pd.Timestamp(df.index[max(pos - 1, 0)])
        row: dict[str, object] = {"pico": peak.date(), "suelo": trough.date()}
        row |= {str(c): float(np.prod(1 + window[c].to_numpy()) - 1) for c in df}
        rows.append(row)
    return pd.DataFrame(rows)


def worst_benchmark_months(df: pd.DataFrame, bench: str, n: int = 6) -> pd.DataFrame:
    mo = cast(pd.DataFrame, stats.to_monthly(df))
    out = mo.sort_values(bench).head(n)
    out.index = pd.DatetimeIndex(out.index).strftime("%Y-%m")
    return out


def pnl_figure(df: pd.DataFrame, bench: str, aum: float, out: Path) -> None:
    _style()
    order = [*LABELS, bench]
    colors = {**SERIES, bench: BENCH_COLOR}
    labels = {**LABELS, bench: f"{bench} (benchmark)"}
    pnl = ((1 + df[order]).cumprod() - 1) * aum
    dd = (1 + df[order]).cumprod()
    dd = dd / dd.cummax() - 1
    x = [_x(d) for d in df.index]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    for c in order:
        lw, ls = (1.4, "--") if c == bench else (1.7, "-")
        ax1.plot(x, pnl[c] / 1e3, color=colors[c], lw=lw, ls=ls, label=labels[c])
        ax2.plot(x, dd[c] * 100, color=colors[c], lw=1.2, ls=ls)
    ax1.axhline(0, color=GRID, lw=1)
    ax1.set_title(f"PnL acumulado fuera de muestra, neto de costes (AUM ${aum / 1e6:.0f}M)")
    ax1.set_ylabel("Miles de USD")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}k"))
    ax1.legend(loc="upper left")
    span = x[-1] - x[0]
    for c in order:
        ax1.annotate(
            f"${pnl[c].iloc[-1] / 1e3:,.0f}k",
            xy=(x[-1], pnl[c].iloc[-1] / 1e3),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=TEXT,
            fontsize=8.5,
        )
    ax1.set_xlim(x[0], x[-1] + 0.09 * span)
    ax2.set_title("Drawdown desde el máximo (%)", fontsize=10)
    ax2.set_ylabel("%", color=TEXT_2)
    ax2.set_facecolor(SURFACE)
    _year_ticks(ax2, 2023, 2026, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run(
    holdout_dir: Path, data: MarketData, aum: float, figure: Path, bench: str = "SPY"
) -> dict[str, pd.DataFrame]:
    df, cash = holdout_streams(holdout_dir, data, bench)
    out = holdout_dir / f"vs_{bench.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "resumen": summary(df, cash, bench),
        "caidas_benchmark": benchmark_drawdowns(df, bench),
        "peores_meses_benchmark": worst_benchmark_months(df, bench),
    }
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", float_format="%.6g")
    pnl_figure(df, bench, aum, figure)
    return tables
