"""Risk analytics for reporting: tail risk, drawdowns, stress tests, factor exposures, regimes.

Nothing in this module feeds back into portfolio decisions; it describes
realised or hypothetical risk of a return stream / weight vector.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats as st

from quantlab.data.market_data import MarketData

# Historical stress windows (peak-to-trough of the S&P 500 or the defining move).
STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "GFC 2007-09": ("2007-10-09", "2009-03-09"),
    "Euro crisis 2011": ("2011-04-29", "2011-10-03"),
    "Taper tantrum 2013": ("2013-05-21", "2013-06-24"),
    "China/oil 2015-16": ("2015-08-10", "2016-02-11"),
    "Q4 2018": ("2018-09-20", "2018-12-24"),
    "COVID 2020": ("2020-02-19", "2020-03-23"),
    "Rates shock 2022": ("2022-01-03", "2022-10-14"),
    "Regional banks 2023": ("2023-03-08", "2023-03-17"),
    "Tariff shock 2025": ("2025-02-19", "2025-04-08"),
}

# Hypothetical instantaneous shocks (simple returns) per ticker. Assumptions:
# bond losses ≈ -duration × Δy (durations: SHY 1.9, IEF 7.5, TLT 17, AGG 6,
# TIP 6.8 real, LQD 8.5, HYG 3.7, EMB 7); equity/credit moves calibrated to
# GFC/2022-type episodes. They are scenarios, not forecasts.
HYPOTHETICAL_SHOCKS: dict[str, dict[str, float]] = {
    "Equity crash (-30%), flight to quality": {
        "SPY": -0.30, "IWM": -0.35, "EFA": -0.30, "EWJ": -0.28, "VGK": -0.32, "EEM": -0.35,
        "SHY": 0.01, "IEF": 0.05, "TLT": 0.10, "AGG": 0.03, "TIP": 0.01,
        "LQD": -0.03, "HYG": -0.15, "EMB": -0.15, "VNQ": -0.35, "GLD": 0.05, "DBC": -0.20,
    },
    "Rates +200bp, equities -15% (2022-type)": {
        "SPY": -0.15, "IWM": -0.18, "EFA": -0.15, "EWJ": -0.12, "VGK": -0.16, "EEM": -0.15,
        "SHY": -0.038, "IEF": -0.15, "TLT": -0.34, "AGG": -0.12, "TIP": -0.136,
        "LQD": -0.17, "HYG": -0.10, "EMB": -0.16, "VNQ": -0.22, "GLD": -0.05, "DBC": 0.10,
    },
    "Stagflation (rates +150bp, commodities +25%)": {
        "SPY": -0.20, "IWM": -0.22, "EFA": -0.20, "EWJ": -0.18, "VGK": -0.22, "EEM": -0.20,
        "SHY": -0.028, "IEF": -0.11, "TLT": -0.25, "AGG": -0.09, "TIP": -0.04,
        "LQD": -0.14, "HYG": -0.12, "EMB": -0.15, "VNQ": -0.20, "GLD": 0.15, "DBC": 0.25,
    },
    "Credit crisis (HY spreads +500bp)": {
        "SPY": -0.20, "IWM": -0.25, "EFA": -0.20, "EWJ": -0.18, "VGK": -0.22, "EEM": -0.25,
        "SHY": 0.01, "IEF": 0.04, "TLT": 0.08, "AGG": 0.01, "TIP": 0.0,
        "LQD": -0.08, "HYG": -0.20, "EMB": -0.20, "VNQ": -0.25, "GLD": 0.03, "DBC": -0.15,
    },
}  # fmt: skip


def var_cvar(returns: pd.Series, levels: tuple[float, ...] = (0.95, 0.99)) -> pd.DataFrame:
    """Historical and Cornish-Fisher VaR / historical CVaR, as positive loss fractions."""
    x = returns.dropna().to_numpy(dtype=float)
    mu, sd = x.mean(), x.std(ddof=1)
    s, k = float(st.skew(x)), float(st.kurtosis(x))  # excess kurtosis
    rows = {}
    for lvl in levels:
        q = np.quantile(x, 1 - lvl)
        z = st.norm.ppf(1 - lvl)
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        rows[f"{lvl:.0%}"] = {
            "VaR_hist": -q,
            "CVaR_hist": -x[x <= q].mean(),
            "VaR_cornish_fisher": -(mu + z_cf * sd),
            "VaR_normal": -(mu + z * sd),
        }
    return pd.DataFrame(rows).T


def drawdown_table(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    wealth = (1 + returns.fillna(0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    rows = []
    in_dd = dd < 0
    episode = (~in_dd).cumsum()[in_dd]
    for _, idx in episode.groupby(episode).groups.items():
        seg = dd.loc[idx]
        trough = pd.Timestamp(seg.index[int(np.argmin(seg.to_numpy()))])
        start = pd.Timestamp(seg.index[0])
        end = pd.Timestamp(seg.index[-1])
        recovered = end != dd.index[-1] or dd.iloc[-1] == 0
        rows.append(
            {
                "start": start.date(),
                "trough": trough.date(),
                "recovery": end.date() if recovered else None,
                "depth": float(seg.min()),
                "days_to_trough": int(np.busday_count(start.date(), trough.date())),
                "days_underwater": len(seg),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("depth").head(top).reset_index(drop=True) if len(out) else out


def historical_stress(returns: Mapping[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, (a, b) in STRESS_WINDOWS.items():
        rows[name] = {
            k: float(np.prod(1 + r.loc[a:b].to_numpy(dtype=float)) - 1)
            if len(r.loc[a:b])
            else np.nan
            for k, r in returns.items()
        }
    return pd.DataFrame(rows).T


def hypothetical_stress(weights: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Instant P&L of weight vectors under each scenario (cash = 0)."""
    rows = {}
    for scen, shocks in HYPOTHETICAL_SHOCKS.items():
        s = pd.Series(shocks)
        rows[scen] = {
            k: float((w.reindex(s.index).fillna(0) * s).sum()) for k, w in weights.items()
        }
    return pd.DataFrame(rows).T


def factor_returns(data: MarketData) -> pd.DataFrame:
    """Monthly excess-return factors from the universe itself (observable, investable)."""
    tri = data.total_return_index
    m = tri.resample("ME").last().pct_change(fill_method=None)
    cash = ((1 + data.cash()).resample("ME").prod() - 1).reindex(m.index)
    return pd.DataFrame(
        {
            "equity": m["SPY"] - cash,
            "duration": m["IEF"] - cash,
            "credit": m["HYG"] - m["IEF"],
            "commodity": m["DBC"] - cash,
            "gold": m["GLD"] - cash,
        }
    ).dropna()


def factor_exposures(
    excess_monthly: pd.Series, factors: pd.DataFrame, lags: int = 3
) -> pd.DataFrame:
    """OLS of monthly excess returns on factors with Newey-West (HAC) t-stats."""
    df = pd.concat([excess_monthly.rename("y"), factors], axis=1, join="inner").dropna()
    y = df["y"].to_numpy()
    x = np.column_stack([np.ones(len(df)), df[factors.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    xtx_inv = np.linalg.inv(x.T @ x)
    s = (x * resid[:, None]).T @ (x * resid[:, None])
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = (x[lag:] * resid[lag:, None]).T @ (x[:-lag] * resid[:-lag, None])
        s += w * (g + g.T)
    se = np.sqrt(np.diag(xtx_inv @ s @ xtx_inv))
    names = ["alpha_monthly", *factors.columns]
    out = pd.DataFrame({"coef": beta, "t_stat": beta / se}, index=names)
    r2 = 1 - resid.var() / y.var()
    out.loc["alpha_annual", ["coef", "t_stat"]] = [beta[0] * 12, beta[0] / se[0]]
    out.loc["R2", "coef"] = r2
    out.loc["n_months", "coef"] = len(df)
    return out


def regime_labels(data: MarketData) -> pd.DataFrame:
    """Causal monthly regimes known at each month-end ``t`` (applied to month t+1).

    trend: SPY 12-month excess return > 0; vol: SPY 63d realised vol above its
    expanding median (only past values)."""
    tri = data.total_return_index["SPY"]
    cash_idx = (1 + data.cash()).cumprod()
    trend = (tri / tri.shift(252)) - (cash_idx / cash_idx.shift(252))
    vol = tri.pct_change(fill_method=None).rolling(63).std()
    vol_med = vol.expanding(252).median()
    month_end = tri.resample("ME").last().index
    daily_to_me = (
        pd.DataFrame(
            {
                "trend_up": trend > 0,
                "high_vol": vol > vol_med,
                "valid": trend.notna() & vol_med.notna(),
            }
        )
        .resample("ME")
        .last()
    )
    labels = daily_to_me.loc[daily_to_me["valid"].astype(bool)].reindex(month_end).dropna()
    return labels[["trend_up", "high_vol"]].astype(bool).shift(1).dropna().astype(bool)


def performance_by_regime(monthly: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Annualised mean monthly return by regime (labels already lagged one month)."""
    df = monthly.join(labels, how="inner")
    rows = {}
    for col, (yes, no) in {"trend_up": ("equity uptrend", "equity downtrend"),
                           "high_vol": ("high vol", "low vol")}.items():  # fmt: skip
        for flag, name in ((True, yes), (False, no)):
            sub = df[df[col] == flag][monthly.columns]
            rows[f"{name} (n={len(sub)})"] = sub.mean() * 12
    return pd.DataFrame(rows).T
