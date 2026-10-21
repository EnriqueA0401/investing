"""Static figures for the committee report (light mode, validated palette slots 1-3).

Every figure has a table counterpart in the report (identity is never color-alone).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b4b2a9"
SERIES = {"H4-IV-TS12": "#2a78d6", "B-EW": "#eb6834", "60-40": "#1baf7a"}
LABELS = {"H4-IV-TS12": "H4-IV-TS12 (selected)", "B-EW": "Equal-weight", "60-40": "60/40"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _x(ts: pd.Timestamp) -> float:
    """Matplotlib date number (days since its default 1970-01-01 epoch)."""
    return float((pd.Timestamp(ts) - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1))


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": TEXT,
            "axes.titlelocation": "left",
            "legend.frameon": False,
        }
    )


def _streams(results: Path) -> pd.DataFrame:
    rs = pd.read_parquet(results / "returns_net.parquet")["H4-IV-TS12"]
    ref = pd.read_parquet(results / "reference.parquet")
    research = pd.DataFrame(
        {"H4-IV-TS12": rs, "B-EW": ref["baseline_ew"], "60-40": ref["benchmark_60_40"]}
    )
    hold = pd.read_parquet(results / "holdout" / "returns_net.parquet")[list(SERIES)]
    return pd.concat([research.dropna(), hold.dropna()])


def _end_labels(ax: Axes, df: pd.DataFrame, fmt: str) -> None:
    last = df.iloc[-1].sort_values()
    lo, hi = np.log(ax.get_ylim())
    ys = list(np.log(last.to_numpy(dtype=float)))
    for i in range(1, len(ys)):  # nudge apart (log space) to avoid collisions
        ys[i] = max(ys[i], ys[i - 1] + 0.05 * (hi - lo))
    for (name, value), y in zip(last.items(), ys, strict=True):
        ax.annotate(
            f"{LABELS[str(name)]}  {fmt.format(float(value))}",
            xy=(_x(df.index[-1]), float(np.exp(y))),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=TEXT,
            fontsize=8.5,
        )


def _year_ticks(ax: Axes, first: int, last: int, step: int) -> None:
    years = range(first, last + 1, step)
    ax.set_xticks([_x(pd.Timestamp(f"{y}-01-01")) for y in years], [str(y) for y in years])


def wealth_figure(results: Path, holdout_start: str, out: Path) -> None:
    df = _streams(results)
    wealth = (1 + df).cumprod()
    x = [_x(d) for d in wealth.index]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for name, color in SERIES.items():
        ax.plot(x, wealth[name], color=color, lw=1.6, label=LABELS[name])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 5]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    x_hold = _x(pd.Timestamp(holdout_start))
    ax.axvline(x_hold, color=TEXT_2, lw=1, ls="--")
    ax.text(x_hold, ax.get_ylim()[1], "  holdout →", va="top", color=TEXT_2, fontsize=8.5)
    ax.set_title("Growth of $1, net of costs (log scale)")
    ax.set_ylabel("Wealth")
    ax.legend(loc="upper left")
    _end_labels(ax, wealth, "${:.2f}")
    ax.set_xlim(x[0], _x(wealth.index[-1] + pd.Timedelta(days=1500)))
    ax.spines["bottom"].set_bounds(x[0], x[-1])
    _year_ticks(ax, 2006, 2026, 4)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def drawdown_figure(results: Path, out: Path) -> None:
    df = _streams(results)
    wealth = (1 + df).cumprod()
    dd = wealth / wealth.cummax() - 1
    x = [_x(d) for d in dd.index]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name, color in SERIES.items():
        ax.plot(x, dd[name] * 100, color=color, lw=1.4, label=LABELS[name])
    ax.set_title("Drawdown from peak (%)")
    ax.set_ylabel("%")
    ax.legend(loc="lower right")
    _year_ticks(ax, 2006, 2026, 4)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def candidates_figure(results: Path, out: Path) -> None:
    v = json.loads((results / "validation.json").read_text())
    c = pd.DataFrame(v["candidates"]).T
    c["sharpe"] = c["sharpe"].astype(float)
    c = c.sort_values("sharpe")
    fig, ax = plt.subplots(figsize=(8, 5))
    # Color follows the entity: EW and the selected config keep their identity colors.
    colors = [SERIES.get(str(k), NEUTRAL) for k in c.index]
    ax.set_axisbelow(True)
    ax.barh(c.index, c["sharpe"], color=colors, height=0.62)
    m = pd.read_csv(results / "metrics.csv", index_col=0)
    bench = float(m["bench_sharpe"].iloc[0])
    ax.axvline(bench, color=SERIES["60-40"], lw=1.6)
    ax.text(bench, len(c) - 0.35, f"60/40 = {bench:.2f}", color=TEXT, fontsize=8.5, ha="center")
    x_col = max(float(c["sharpe"].max()), bench) * 1.08
    ax.text(x_col, len(c) - 0.35, "Sharpe · DSR", color=TEXT_2, fontsize=8, fontweight="bold")
    for i, (_, row) in enumerate(c.iterrows()):
        ax.text(
            x_col,
            i,
            f"{row['sharpe']:.2f} · {float(row['dsr']):.2f}",
            va="center",
            fontsize=8,
            color=TEXT_2,
        )
    ax.set_title("Research-window net Sharpe of the 15 pre-registered configs")
    ax.set_xlabel("Annualised Sharpe (excess of T-bill), 2005-2022")
    ax.set_xlim(0, max(float(c["sharpe"].max()), bench) * 1.32)
    ax.set_ylim(-0.6, len(c) + 0.4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def robustness_heatmap(results: Path, out: Path) -> None:
    m = pd.read_csv(results / "robustness" / "variants_H4-IV-TS12.csv", index_col=0)
    grid = m[m.index.str.startswith("param_")]["sharpe"]
    parsed = grid.index.str.extract(r"trend(\d+)_vol(\d+)").astype(int)
    tab = pd.DataFrame(
        {"trend": parsed[0].to_numpy(), "vol": parsed[1].to_numpy(), "s": grid.to_numpy()}
    )
    piv = tab.pivot_table(index="trend", columns="vol", values="s")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(piv.to_numpy(), cmap=cmap, vmin=0.3, vmax=0.75, aspect="auto")
    for (i, j), val in np.ndenumerate(piv.to_numpy()):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(range(len(piv.columns)), [f"{c}d" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{r}d" for r in piv.index])
    ax.set_xlabel("Inverse-vol window")
    ax.set_ylabel("Trend lookback")
    ax.set_title("H4 Sharpe across parameters (EW = 0.43)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def make_all(results: Path, holdout_start: str, out_dir: Path) -> list[Path]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Callable[[Path], None]] = {
        "wealth.png": lambda p: wealth_figure(results, holdout_start, p),
        "drawdowns.png": lambda p: drawdown_figure(results, p),
        "candidates.png": lambda p: candidates_figure(results, p),
        "robustness_heatmap.png": lambda p: robustness_heatmap(results, p),
    }
    for name, fn in paths.items():
        fn(out_dir / name)
    return [out_dir / n for n in paths]
