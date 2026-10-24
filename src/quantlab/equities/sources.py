"""Auxiliary free sources for the equity study.

* SEC ticker/exchange map and per-company SIC codes (EDGAR submissions API).
* Fama-French 12-industry classification (Ken French's Siccodes12 definition).
* Ken French daily factor returns (FF5 + momentum), used ONLY for ex-post
  attribution (they are revised over time, so never as decision inputs).
* FRED market series for the macro overlay (Moody's Baa-10y spread, 10y-3m
  slope). These are market prints that are not revised; a one-session
  publication lag is still applied when they are used.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from quantlab.components.vendors import fetch_fred_series
from quantlab.equities.edgar import SUBMISSIONS_URL, TICKERS_URL, USER_AGENT
from quantlab.logging_setup import get_logger

log = get_logger(__name__)

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# Ken French Siccodes12 (verified against the published Siccodes12.txt).
FF12_RANGES: dict[str, list[tuple[int, int]]] = {
    "NoDur": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    "Durbl": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
              (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    "Manuf": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
              (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
              (3830, 3839), (3860, 3899)],
    "Enrgy": [(1200, 1399), (2900, 2999)],
    "Chems": [(2800, 2829), (2840, 2899)],
    "BusEq": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    "Telcm": [(4800, 4899)],
    "Utils": [(4900, 4949)],
    "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
    "Hlth": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    "Money": [(6000, 6999)],
}  # fmt: skip


def ff12(sic: int | None) -> str:
    if sic is None:
        return "Other"
    for name, ranges in FF12_RANGES.items():
        if any(lo <= sic <= hi for lo, hi in ranges):
            return name
    return "Other"


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def ticker_map() -> pd.DataFrame:
    """Current SEC map: cik, name, ticker, exchange (one row per listed ticker)."""
    raw = _get_json(TICKERS_URL)
    assert isinstance(raw, dict)
    df = pd.DataFrame(raw["data"], columns=raw["fields"])
    return df.rename(
        columns={"cik": "cik", "name": "name", "ticker": "ticker", "exchange": "exchange"}
    )


def sic_codes(ciks: list[int], cache: Path, pause: float = 0.12) -> pd.DataFrame:
    """SIC and former names per CIK from the submissions API (cached; ~8 req/s)."""
    rows: dict[int, dict[str, object]] = {}
    if cache.exists():
        rows = {int(k): v for k, v in json.loads(cache.read_text()).items()}
    todo = [c for c in ciks if c not in rows]
    for i, cik in enumerate(todo):
        try:
            doc = _get_json(SUBMISSIONS_URL.format(cik=cik))
            assert isinstance(doc, dict)
            sic = int(doc["sic"]) if str(doc.get("sic") or "").isdigit() else None
            rows[cik] = {
                "sic": sic,
                "sic_desc": doc.get("sicDescription"),
                "category": doc.get("category"),
            }
        except Exception as exc:
            log.warning("submissions fetch failed", extra={"cik": cik, "error": repr(exc)})
        time.sleep(pause)
        if i % 200 == 0:
            cache.write_text(json.dumps(rows))
    cache.write_text(json.dumps(rows))
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "cik"
    out["ff12"] = [ff12(None if pd.isna(s) else int(s)) for s in out["sic"]]
    return out


def _french_csv(name: str) -> pd.DataFrame:
    req = urllib.request.Request(FRENCH + name, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
    text = z.read(z.namelist()[0]).decode("latin1")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip()[:8].isdigit())
    header = lines[start - 1]
    body = []
    for ln in lines[start:]:
        if not ln.strip()[:8].isdigit():
            break
        body.append(ln)
    df = pd.read_csv(io.StringIO("\n".join([header, *body])), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.columns = [c.strip() for c in df.columns]
    return df / 100.0


def french_daily_factors() -> pd.DataFrame:
    """Daily Mkt-RF, SMB, HML, RMW, CMA, RF and UMD (decimal returns)."""
    ff5 = _french_csv("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    mom = _french_csv("F-F_Momentum_Factor_daily_CSV.zip").rename(columns={"Mom": "UMD"})
    return ff5.join(mom, how="inner")


def macro_series(start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    """Daily FRED: BAA10Y (credit spread, %) and T10Y3M (curve slope, %)."""
    s = start.date()
    e = end.date() if end is not None else None
    return pd.concat(
        [fetch_fred_series("BAA10Y", s, e), fetch_fred_series("T10Y3M", s, e)], axis=1
    ).ffill()
