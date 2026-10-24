"""SEC EDGAR XBRL 'companyfacts' -> compact long table of point-in-time facts.

Every fact keeps its ``filed`` date, so a value is usable only after the filing
became public. Values are kept *as originally reported* downstream (first filing
of each period wins); later restatements are ignored, as an investor at the time
could not have known them.
"""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from quantlab.logging_setup import get_logger

log = get_logger(__name__)

COMPANYFACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
# SEC asks for a descriptive User-Agent; no personal data is sent by default.
USER_AGENT = os.environ.get("SEC_USER_AGENT", "quantlab-research admin@quantlab.invalid")

FLOW_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
)
INSTANT_TAGS = (
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "CommonStockSharesOutstanding",
)
DEI_TAGS = ("EntityCommonStockSharesOutstanding", "EntityPublicFloat")
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-QT"}


def _get(url: str, dest: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def download_companyfacts(raw_dir: Path) -> Path:
    dest = raw_dir / "companyfacts.zip"
    if not dest.exists():
        log.info("downloading EDGAR companyfacts bulk file (~1.4 GB)")
        _get(COMPANYFACTS_URL, dest.with_suffix(".part")).rename(dest)
    return dest


CATEGORICAL = ("tag", "unit", "form", "fp")
MIN_END = "2007-01-01"  # nothing earlier is needed (prices start mid-2009)


def _frame(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=["cik", "tag", "unit", "val", "start", "end", "filed", "form", "fy", "fp", "accn"],
    )
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("float32")
    df["cik"] = df["cik"].astype("int32")
    for c in CATEGORICAL:
        df[c] = df[c].astype("string")
    df = df.dropna(subset=["val", "end", "filed"])
    return df[df["end"] >= MIN_END]


def extract_facts(zip_path: Path, parts_dir: Path, flush_every: int = 1500) -> dict[int, str]:
    """Stream the zip, writing compact parquet parts (bounded memory). Returns entity names."""
    wanted = {("us-gaap", t) for t in FLOW_TAGS + INSTANT_TAGS} | {("dei", t) for t in DEI_TAGS}
    parts_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[object, ...]] = []
    names: dict[int, str] = {}
    part = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".json")]
        for i, member in enumerate(members):
            try:
                doc = json.loads(zf.read(member))
            except (json.JSONDecodeError, KeyError):
                continue
            cik = int(doc.get("cik", 0) or 0)
            names[cik] = str(doc.get("entityName", ""))
            facts = doc.get("facts", {})
            for ns, tag in wanted:
                units = facts.get(ns, {}).get(tag, {}).get("units", {})
                for unit, obs in units.items():
                    for o in obs:
                        if o.get("form") not in FORMS:
                            continue
                        rows.append(
                            (
                                cik,
                                tag,
                                unit,
                                o.get("val"),
                                o.get("start"),
                                o.get("end"),
                                o.get("filed"),
                                o.get("form"),
                                o.get("fy"),
                                o.get("fp"),
                                o.get("accn"),
                            )
                        )
            if (i + 1) % flush_every == 0 or i == len(members) - 1:
                _frame(rows).to_parquet(parts_dir / f"part_{part:03d}.parquet", compression="zstd")
                log.info("edgar extract progress", extra={"done": i + 1, "of": len(members)})
                rows, part = [], part + 1
    return names


def build_facts_table(raw_dir: Path) -> Path:
    out = raw_dir / "facts"
    if (raw_dir / "entity_names.parquet").exists():
        return out
    names = extract_facts(download_companyfacts(raw_dir), out)
    ser = pd.Series(names, name="name")
    ser.index.name = "cik"
    ser.to_frame().to_parquet(raw_dir / "entity_names.parquet")
    return out


def load_facts(raw_dir: Path, ciks: set[int] | None = None) -> pd.DataFrame:
    """Read the parquet parts, optionally only for some CIKs (filter pushed down)."""
    filters = [("cik", "in", sorted(ciks))] if ciks is not None else None
    return pd.read_parquet(raw_dir / "facts", filters=filters)


if __name__ == "__main__":
    from quantlab.logging_setup import configure_logging

    configure_logging()
    print(build_facts_table(Path("data/raw/edgar")))
