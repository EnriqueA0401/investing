"""H-006 diagnostic: CPCV out-of-sample IC of ridge vs. an untrained z-score combination.

Research window only. Samples are (month-end s, asset); label = next-month
return minus the cross-sectional mean; t0 = s, t1 = next month-end.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as st

from quantlab import registry
from quantlab.calendar import rebalance_dates
from quantlab.config import ComponentSpec
from quantlab.cv import combinatorial_purged_cv, cpcv_paths
from quantlab.data.market_data import MarketData


def build_panel(
    data: MarketData,
    factor_specs: list[ComponentSpec],
    signs: dict[str, float],
    tradable: list[str],
    start: pd.Timestamp,
) -> pd.DataFrame:
    panels = {}
    for spec in factor_specs:
        f = registry.build("factor", spec)
        panels[f.label] = f.compute(data)  # causal factors: row s uses data <= s
    tri = data.total_return_index
    me = [d for d in rebalance_dates(data.dates, "ME") if d >= start]
    rows = []
    for s, s_next in pairwise(me):
        feats = pd.concat({k: panels[k].loc[s] for k in signs}, axis=1).reindex(tradable).dropna()
        fwd = (tri.loc[s_next] / tri.loc[s] - 1).reindex(feats.index).dropna()
        feats = feats.loc[fwd.index]
        if len(feats) < 5:
            continue
        z = (feats - feats.mean()) / feats.std(ddof=0)
        z["label"] = fwd - fwd.mean()
        z["t0"], z["t1"] = s, s_next
        rows.append(z.reset_index(names="ticker"))
    out: pd.DataFrame = pd.concat(rows, ignore_index=True)
    return out


def _ic_by_date(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    tmp = df[["t0", "label"]].assign(score=score)
    ics = {t0: float(st.spearmanr(g["score"], g["label"]).statistic) for t0, g in tmp.groupby("t0")}
    return pd.Series(ics)


def cpcv_ic(
    panel: pd.DataFrame,
    signs: dict[str, float],
    *,
    alpha: float = 1.0,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_days: int = 21,
) -> dict[str, Any]:
    feats = list(signs)
    x = panel[feats].to_numpy()
    y = panel["label"].to_numpy()
    combo = (panel[feats] * pd.Series(signs)).mean(axis=1).to_numpy()
    results: dict[tuple[int, ...], dict[int, tuple[float, float]]] = {}
    groups_of_t0 = {
        t: g
        for g, chunk in enumerate(np.array_split(np.sort(panel["t0"].unique()), n_groups))
        for t in chunk
    }
    for split in combinatorial_purged_cv(panel["t0"], panel["t1"], n_groups, k_test, embargo_days):
        xt, yt = x[split.train], y[split.train]
        beta = np.linalg.solve(xt.T @ xt + alpha * np.eye(len(feats)), xt.T @ yt)
        test = panel.iloc[split.test]
        ic_ridge = _ic_by_date(test, x[split.test] @ beta)
        ic_combo = _ic_by_date(test, combo[split.test])
        g_of = test.groupby("t0").size().index.map(groups_of_t0)
        per_group = {}
        for g in split.test_groups:
            mask = np.asarray(g_of == g)
            per_group[g] = (float(ic_ridge[mask].mean()), float(ic_combo[mask].mean()))
        results[split.test_groups] = per_group
    paths = cpcv_paths(n_groups, k_test)
    path_rows = []
    for j, path in enumerate(paths):
        r = np.mean([results[split][g][0] for g, split in enumerate(path)])
        c = np.mean([results[split][g][1] for g, split in enumerate(path)])
        path_rows.append({"path": j, "ic_ridge": float(r), "ic_combo": float(c)})
    df = pd.DataFrame(path_rows)
    ridge_wins = int((df["ic_ridge"] > df["ic_combo"]).sum())
    return {
        "paths": path_rows,
        "median_ic_ridge": float(df["ic_ridge"].median()),
        "median_ic_combo": float(df["ic_combo"].median()),
        "ridge_wins_paths": ridge_wins,
        "skeptical_thesis_refuted": bool(
            df["ic_ridge"].median() > df["ic_combo"].median() and ridge_wins >= 4
        ),
        "n_samples": len(panel),
        "n_months": int(panel["t0"].nunique()),
    }


def walk_forward_ic(
    panel: pd.DataFrame, signs: dict[str, float], *, alpha: float = 1.0, min_train_months: int = 36
) -> dict[str, Any]:
    """Expanding-window IC: the model at month s is trained only on labels ending <= s.

    Reconciles CPCV (which may train on data *after* the test block) with the
    deployable, past-only setting used in the H6-RIDGE backtest."""
    feats = list(signs)
    months = np.sort(panel["t0"].unique())
    combo_all = (panel[feats] * pd.Series(signs)).mean(axis=1)
    rows = []
    for s in months[min_train_months:]:
        train = panel[panel["t1"] <= s]
        test = panel[panel["t0"] == s]
        xt, yt = train[feats].to_numpy(), train["label"].to_numpy()
        beta = np.linalg.solve(xt.T @ xt + alpha * np.eye(len(feats)), xt.T @ yt)
        ic_r = st.spearmanr(test[feats].to_numpy() @ beta, test["label"]).statistic
        ic_c = st.spearmanr(combo_all.loc[test.index], test["label"]).statistic
        rows.append(
            {
                "t0": s,
                "ic_ridge": float(ic_r),
                "ic_combo": float(ic_c),
                **dict(zip(feats, beta, strict=True)),
            }
        )
    df = pd.DataFrame(rows).set_index("t0")
    return {
        "mean_ic_ridge": float(df["ic_ridge"].mean()),
        "mean_ic_combo": float(df["ic_combo"].mean()),
        "t_stat_ic_diff": float(
            (df["ic_ridge"] - df["ic_combo"]).mean()
            / (df["ic_ridge"] - df["ic_combo"]).std(ddof=1)
            * np.sqrt(len(df))
        ),
        "n_months": len(df),
        "coef_first": {f: float(df[f].iloc[0]) for f in feats},
        "coef_last": {f: float(df[f].iloc[-1]) for f in feats},
        "coef_sign_flips": {f: int((np.diff(np.sign(df[f])) != 0).sum()) for f in feats},
    }
