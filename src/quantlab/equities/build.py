"""Assemble the point-in-time equity dataset and freeze it as a snapshot.

    uv run python -m quantlab.equities.build

Steps: EDGAR facts -> candidates (listed today, public float >= threshold at
some point) -> Yahoo prices -> market caps (PIT shares x price) -> annual
top-N reconstitution -> SIC/FF12 sectors -> monthly PIT fundamentals -> FRED
macro and T-bill cash -> survivorship coverage report -> snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.components.vendors import cash_return_from_rate, fetch_fred_series
from quantlab.data.market_data import MarketData
from quantlab.data.store import SnapshotStore
from quantlab.equities import edgar, fundamentals, linking, prices, sources, universe
from quantlab.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class BuildConfig:
    raw_dir: Path = Path("data/raw")
    snapshot_dir: Path = Path("data/snapshots")
    report_dir: Path = Path("research/results/equity_data")
    price_start: str = "2009-06-01"
    price_end: str = "2026-10-01"  # exclusive; last session 2026-09-30
    min_float_usd: float = 5e8
    universe_size: int = 1000
    benchmark: str = "SPY"
    exchanges: tuple[str, ...] = ("NYSE", "Nasdaq", "CBOE")


def listed_today(cfg: BuildConfig) -> pd.DataFrame:
    tmap = sources.ticker_map()
    return tmap[tmap["exchange"].isin(cfg.exchanges)].drop_duplicates("cik", keep="first")


def candidate_tickers(facts: pd.DataFrame, tmap: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    floats = fundamentals.float_records(facts)
    floats = floats[floats["end"] >= pd.Timestamp("2008-06-30")]
    max_float = floats.groupby("cik")["val"].max()
    big = max_float[max_float >= cfg.min_float_usd].index
    cand = tmap[tmap["cik"].isin(big)].copy()
    cand["ticker"] = cand["ticker"].str.replace(".", "-", regex=False)
    return cand[["cik", "ticker", "name", "exchange"]].reset_index(drop=True)


def build_shares_panel(
    facts: pd.DataFrame,
    cik_to_ticker: dict[int, str],
    factor_after: pd.DataFrame,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Month-end shares in today's split units: best available source on each date
    (cover page > balance sheet > weighted average > float-implied), scale outliers
    removed."""
    rec = fundamentals.shares_records(facts[facts["cik"].isin(cik_to_ticker)])
    rec["ticker"] = rec["cik"].map(cik_to_ticker)
    rec = prices.to_today_units(rec, factor_after)
    combined: pd.DataFrame | None = None
    for src in sorted(rec["source"].unique()):
        part = fundamentals.drop_scale_outliers(rec[rec["source"] == src])
        panel = fundamentals.asof_panel(
            part[["cik", "end", "val", "avail"]], dates, max_age_days=400
        )
        combined = panel if combined is None else combined.combine_first(panel)
    assert combined is not None
    combined.columns = [cik_to_ticker[int(c)] for c in combined.columns]
    # Last resort (issuers reporting shares only per class, e.g. Visa): shares implied
    # by EntityPublicFloat / split-adjusted close at the float date. Already in today's
    # units because the close is split-adjusted. Understates by insider holdings.
    flt = fundamentals.float_records(facts[facts["cik"].isin(cik_to_ticker)]).copy()
    flt["ticker"] = flt["cik"].map(cik_to_ticker)
    px = close.reindex(close.index.union(pd.DatetimeIndex(flt["end"].unique()))).ffill()
    px_arr = px.to_numpy(dtype=float)
    row_of = {d: i for i, d in enumerate(px.index)}
    col_of = {t: j for j, t in enumerate(px.columns)}
    prices_at = np.array(
        [
            px_arr[row_of[e], col_of[t]] if t in col_of else np.nan
            for e, t in zip(flt["end"], flt["ticker"], strict=True)
        ]
    )
    flt["val"] = flt["val"].to_numpy(dtype=float) / np.where(prices_at > 0, prices_at, np.nan)
    implied = fundamentals.asof_panel(
        fundamentals.drop_scale_outliers(
            flt.dropna(subset=["val"])[["cik", "end", "val", "avail"]]
        ),
        dates,
        max_age_days=550,
    )
    implied.columns = [cik_to_ticker[int(c)] for c in implied.columns]
    return combined.combine_first(implied)


def build(cfg: BuildConfig) -> str:  # noqa: PLR0915 - linear, documented build recipe
    edgar_dir = cfg.raw_dir / "edgar"
    edgar.build_facts_table(edgar_dir)
    facts = edgar.load_facts(edgar_dir)
    names = pd.read_parquet(edgar_dir / "entity_names.parquet")["name"]
    log.info("facts loaded", extra={"rows": len(facts)})

    tmap = listed_today(cfg)
    links = linking.link_predecessors(facts, names, set(tmap["cik"].astype(int)))
    facts = linking.relabel(facts, links)
    log.info("predecessor CIKs linked", extra={"n": len(links)})
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "predecessor_cik": p,
                "predecessor": names.get(p, ""),
                "successor_cik": s,
                "successor": names.get(s, ""),
            }
            for p, s in links.items()
        ]
    ).to_csv(cfg.report_dir / "cik_links.csv", index=False)

    cand = candidate_tickers(facts, tmap, cfg)
    log.info("candidates", extra={"n": len(cand)})
    tickers = sorted({*cand["ticker"], cfg.benchmark})
    raw = prices.download_prices(tickers, cfg.price_start, cfg.price_end, cfg.raw_dir / "prices")
    panels = prices.build_price_panels(raw)
    priced = set(panels["close"].columns)
    cand = cand[cand["ticker"].isin(priced)]
    cik_to_ticker = dict(zip(cand["cik"], cand["ticker"], strict=True))
    log.info("priced candidates", extra={"n": len(cand)})

    # PIT shares in today's split units -> monthly market caps -> membership.
    trading = pd.DatetimeIndex(panels["close"].index)
    dates = universe.month_end_dates(trading)
    shares_panel = build_shares_panel(
        facts, cik_to_ticker, panels["split_factor_after"], panels["close"], dates
    )
    mcap = universe.monthly_market_cap(shares_panel, panels["close"])
    members = universe.reconstitute(mcap, panels["close"], size=cfg.universe_size)
    ever = members.columns[members.any()]
    log.info("universe", extra={"ever_members": len(ever)})

    member_ciks = [c for c, t in cik_to_ticker.items() if t in set(ever)]
    sic = sources.sic_codes(member_ciks, cfg.raw_dir / "edgar" / "sic_cache.json")
    sector = {cik_to_ticker[int(str(c))]: str(s) for c, s in sic["ff12"].items()}

    fund = fundamentals.build_fundamental_panels(facts, dates, member_ciks)
    fund = {k: v.rename(columns=cik_to_ticker).reindex(columns=ever) for k, v in fund.items()}
    fund["shares"] = shares_panel.reindex(columns=ever)
    fund["market_cap"] = mcap.reindex(columns=ever)

    keep = [*ever, cfg.benchmark]
    idx = trading
    cash = cash_return_from_rate(
        fetch_fred_series("DTB3", date(2009, 1, 1), date(2026, 10, 1)), idx, lag_days=1
    ).fillna(0.0)
    macro = sources.macro_series(pd.Timestamp("2009-01-01"), pd.Timestamp("2026-10-01"))
    macro = macro.reindex(macro.index.union(idx)).ffill().reindex(idx)
    data = MarketData(
        close=panels["close"][keep],
        total_return_index=panels["total_return_index"][keep],
        volume=panels["volume"][keep],
        asset_class={**sector, cfg.benchmark: "benchmark"},
        cash_return=cash,
        fundamentals=fund,
        universe=members.reindex(columns=keep, fill_value=False),
        extras={c: macro[c] for c in macro.columns},
    )

    floats_all = fundamentals.float_records(facts)
    rank_dates = [d for d in dates if d.month == 5]
    summary, missing = universe.coverage_report(
        floats_all, set(cik_to_ticker), names, rank_dates, size=cfg.universe_size
    )
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cfg.report_dir / "survivorship_coverage.csv", index=False, float_format="%.4f")
    missing.to_csv(
        cfg.report_dir / "survivorship_missing_top.csv", index=False, float_format="%.3f"
    )
    pd.Series(sector).rename("ff12").to_csv(cfg.report_dir / "sectors.csv")
    members.sum(axis=1).rename("n_members").to_csv(cfg.report_dir / "universe_size.csv")

    snap_id = SnapshotStore(cfg.snapshot_dir).save(
        data,
        {
            "source": "yfinance prices + SEC EDGAR companyfacts + FRED (DTB3, BAA10Y, T10Y3M)",
            "build": {k: str(v) for k, v in cfg.__dict__.items()},
            "survivorship": "only companies listed at build time have prices; see coverage report",
        },
    )
    (cfg.report_dir / "snapshot_id.json").write_text(json.dumps({"snapshot_id": snap_id}))
    log.info("equity snapshot", extra={"snapshot_id": snap_id})
    return snap_id


if __name__ == "__main__":
    configure_logging()
    print(build(BuildConfig()))
