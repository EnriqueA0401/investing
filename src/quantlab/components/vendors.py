"""Free-vendor adapters: Yahoo Finance (prices, dividends, volume) and FRED (T-bill rate).

Point-in-time notes
-------------------
* The total-return index is *reconstructed* from split-adjusted close plus
  cash distributions on their ex-dates: ``TR_t = TR_{t-1} (C_t + D_t + CG_t) / C_{t-1}``.
  Ex-dates are announced in advance, so this is observable at the close of ``t``.
  The vendor's back-adjusted "Adj Close" is used only as an independent cross-check.
* Split adjustment of the close is retroactive but multiplicative, so ratios
  (returns) are unaffected; volume is split-adjusted consistently.
* The cash return on day ``d`` accrues the 3-month T-bill rate observed on the
  previous trading day (``cash_rate_lag_days``), so it never uses a rate fixed
  after the accrual period started.
"""

from __future__ import annotations

import io
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any, cast

import pandas as pd
from pydantic import Field

from quantlab.config import Asset
from quantlab.contracts import ComponentParams, DataSource
from quantlab.data.market_data import MarketData
from quantlab.logging_setup import get_logger
from quantlab.registry import register

log = get_logger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class VendorDataError(RuntimeError):
    pass


def _retry[T](fn: Callable[[], T], attempts: int, what: str) -> T:
    delay = 2.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1:
                raise VendorDataError(f"{what} failed after {attempts} attempts") from exc
            log.warning("vendor call failed, retrying", extra={"what": what, "error": repr(exc)})
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def fetch_fred_series(
    series_id: str, start: date, end: date | None, attempts: int = 4
) -> pd.Series:
    """Download a FRED series as a float Series (missing values -> NaN)."""
    query = {"id": series_id, "cosd": start.isoformat()}
    if end is not None:
        query["coed"] = end.isoformat()
    url = f"{FRED_CSV_URL}?{urllib.parse.urlencode(query)}"

    def _get() -> bytes:
        with urllib.request.urlopen(url, timeout=60) as resp:
            body: bytes = resp.read()
            return body

    raw = pd.read_csv(io.BytesIO(_retry(_get, attempts, f"FRED {series_id}")))
    date_col = raw.columns[0]
    s = pd.to_numeric(raw[series_id], errors="coerce")
    s.index = pd.DatetimeIndex(pd.to_datetime(raw[date_col]), name="date")
    return s.rename(series_id)


def reconstruct_total_return(close: pd.DataFrame, distributions: pd.DataFrame) -> pd.DataFrame:
    """Total-return index from split-adjusted close and ex-date cash distributions."""
    gross = (close + distributions.reindex_like(close).fillna(0.0)) / close.shift(1)
    tri = gross.copy()
    for col in tri.columns:
        if close[col].notna().any():
            tri.loc[close[col].first_valid_index(), col] = 1.0
    tri = tri.where(close.notna())
    out: pd.DataFrame = 100.0 * tri.cumprod(skipna=True).where(close.notna())
    return out


def cash_return_from_rate(
    rate_pct: pd.Series, trading_dates: pd.DatetimeIndex, lag_days: int
) -> pd.Series:
    """Daily cash return on each trading date from an annualised percentage yield.

    The rate applied on ``d`` is the last observation available ``lag_days``
    trading sessions earlier.
    """
    observed = rate_pct.dropna().sort_index()
    on_dates = observed.reindex(observed.index.union(trading_dates)).ffill().reindex(trading_dates)
    lagged = on_dates.shift(lag_days)
    daily = (1.0 + lagged / 100.0) ** (1.0 / 252.0) - 1.0
    return daily.rename("cash")


@register("data_source", "yfinance")
class YahooFred(DataSource):
    """Yahoo Finance daily prices/distributions + FRED 3-month T-bill as cash."""

    class Params(ComponentParams):
        cash_series: str | None = "DTB3"
        cash_rate_lag_days: int = Field(default=1, ge=1)
        attempts: int = Field(default=4, ge=1)
        max_vendor_divergence: float = Field(
            default=0.005,
            description="Max |daily return diff| vs vendor Adj Close before the fetch fails.",
        )

    params: Params

    def _download(self, tickers: list[str], start: date, end: date | None) -> pd.DataFrame:
        import yfinance as yf  # noqa: PLC0415 - optional heavy import only when fetching

        kwargs: dict[str, Any] = {
            "start": start.isoformat(),
            "auto_adjust": False,
            "actions": True,
            "progress": False,
            "group_by": "column",
            "threads": False,
        }
        if end is not None:
            kwargs["end"] = (end + timedelta(days=1)).isoformat()  # yfinance end is exclusive
        df = _retry(lambda: yf.download(tickers, **kwargs), self.params.attempts, "yfinance")
        if df is None or df.empty:
            raise VendorDataError("yfinance returned no data")
        missing = set(tickers) - set(df["Close"].columns)
        if missing:
            raise VendorDataError(f"yfinance returned no data for {sorted(missing)}")
        return cast(pd.DataFrame, df)

    def fetch(self, assets: Sequence[Asset], start: date, end: date | None) -> MarketData:
        tickers = [a.ticker for a in assets]
        raw = self._download(tickers, start, end)
        idx = pd.DatetimeIndex(pd.to_datetime(raw.index).normalize(), name="date").as_unit("ns")

        def panel(field: str) -> pd.DataFrame:
            if field not in raw.columns.get_level_values(0):
                return pd.DataFrame(0.0, index=idx, columns=tickers)
            out = cast(pd.DataFrame, raw[field]).reindex(columns=tickers).astype(float)
            out.index = idx
            out.columns.name = None
            return out

        close = panel("Close")
        # Drop sessions where no asset traded (vendor artefacts).
        keep = close.notna().any(axis=1).to_numpy()
        close = close.loc[keep]
        dist = (panel("Dividends").fillna(0.0) + panel("Capital Gains").fillna(0.0)).loc[keep]
        volume = panel("Volume").loc[keep].where(close.notna())
        vendor_adj = panel("Adj Close").loc[keep]

        tri = reconstruct_total_return(close, dist)
        diff = (tri.pct_change(fill_method=None) - vendor_adj.pct_change(fill_method=None)).abs()
        worst = diff.max()
        if (worst > self.params.max_vendor_divergence).any():
            bad = worst[worst > self.params.max_vendor_divergence]
            raise VendorDataError(
                f"reconstructed TR diverges from vendor Adj Close: {bad.to_dict()}"
            )
        log.info(
            "vendor cross-check passed",
            extra={"max_daily_divergence": {k: round(float(v), 6) for k, v in worst.items()}},
        )

        cash: pd.Series | None = None
        if self.params.cash_series is not None:
            rate = fetch_fred_series(
                self.params.cash_series, start - timedelta(days=30), end, self.params.attempts
            )
            dates = cast(pd.DatetimeIndex, close.index)
            cash = cash_return_from_rate(rate, dates, self.params.cash_rate_lag_days)
            first_ok = cash.first_valid_index()
            if first_ok is None or first_ok > close.index[self.params.cash_rate_lag_days]:
                raise VendorDataError("cash rate series does not cover the requested window")
            cash = cash.fillna(0.0)  # only the first `lag` rows, before any decision

        return MarketData(
            close=close,
            total_return_index=tri,
            volume=volume,
            asset_class={a.ticker: a.asset_class for a in assets},
            cash_return=cash,
        )
