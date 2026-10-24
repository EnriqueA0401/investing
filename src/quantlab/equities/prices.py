"""Daily prices for the candidate stocks (Yahoo Finance via yfinance).

Survivorship caveat: Yahoo only serves tickers that still exist, so companies
that were acquired or went bankrupt are absent. ``universe.coverage_report``
measures how much of the true point-in-time universe this removes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.components.vendors import reconstruct_total_return
from quantlab.logging_setup import get_logger

log = get_logger(__name__)

FIELDS = ("Close", "Volume", "Dividends", "Stock Splits", "Capital Gains")


def download_prices(
    tickers: list[str], start: str, end: str, raw_dir: Path, chunk: int = 150
) -> dict[str, pd.DataFrame]:
    """Download in chunks (cached per chunk) and return wide panels per field."""
    import yfinance as yf  # noqa: PLC0415 - heavy optional import

    raw_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, list[pd.DataFrame]] = {f: [] for f in FIELDS}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        cache = raw_dir / f"chunk_{i // chunk:04d}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            kwargs: dict[str, Any] = {
                "start": start,
                "end": end,
                "auto_adjust": False,
                "actions": True,
                "progress": False,
                "group_by": "column",
                "threads": True,
            }
            df = yf.download(batch, **kwargs)
            if df is None or df.empty:
                continue
            df.to_parquet(cache)
            log.info("price chunk", extra={"chunk": i // chunk, "tickers": len(batch)})
        for f in FIELDS:
            if f in df.columns.get_level_values(0):
                parts[f].append(pd.DataFrame(df[f]))
    out = {}
    for f, frames in parts.items():
        if frames:
            panel = pd.concat(frames, axis=1)
            panel.index = pd.DatetimeIndex(
                pd.to_datetime(panel.index).normalize(), name="date"
            ).as_unit("ns")
            out[f] = panel.loc[:, ~panel.columns.duplicated()].astype("float64")
    return out


def build_price_panels(raw: dict[str, pd.DataFrame], min_obs: int = 252) -> dict[str, pd.DataFrame]:
    """close (split-adjusted), total-return index, volume, split factor after each date."""
    close = raw["Close"]
    keep = close.notna().sum() >= min_obs
    close = close.loc[:, keep]
    rows = close.notna().any(axis=1).to_numpy()
    close = close.loc[rows]
    cols = close.columns
    dist = raw["Dividends"].reindex(index=close.index, columns=cols).fillna(0.0)
    if "Capital Gains" in raw:
        dist = dist + raw["Capital Gains"].reindex(index=close.index, columns=cols).fillna(0.0)
    # Fill short gaps (<= 5 sessions) inside a listing: flat price, zero return.
    live = close.ffill().notna() & close.bfill().notna()
    filled = close.ffill(limit=5).where(live)
    tri = reconstruct_total_return(filled, dist)
    splits = raw["Stock Splits"].reindex(index=close.index, columns=cols).fillna(0.0)
    ratio = splits.where(splits > 0, 1.0)
    # factor_after[t] = product of split ratios strictly after t (today's units / units at t).
    rev_cum = ratio.iloc[::-1].cumprod().iloc[::-1]
    factor_after = (rev_cum / ratio).astype(float)
    volume = raw["Volume"].reindex(index=close.index, columns=cols)
    return {
        "close": filled,
        "total_return_index": tri,
        "volume": volume.where(filled.notna()),
        "split_factor_after": factor_after.where(filled.notna()).ffill(),
    }


def split_factor_at(factor_after: pd.DataFrame, ticker: str, when: pd.Timestamp) -> float:
    s = factor_after[ticker]
    s = s.loc[:when].dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def to_today_units(shares: pd.DataFrame, factor_after: pd.DataFrame) -> pd.DataFrame:
    """Shares records (ticker, end, val) -> val expressed in today's split units."""
    out = shares.copy()
    f = []
    for t, end in zip(out["ticker"], out["end"], strict=True):
        f.append(split_factor_at(factor_after, t, end) if t in factor_after else np.nan)
    out["val"] = out["val"] * np.asarray(f)
    return out
