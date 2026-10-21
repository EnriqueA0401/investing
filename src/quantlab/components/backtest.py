"""Daily drift backtester with lagged execution and explicit costs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd
from pydantic import Field

from quantlab.calendar import shift_trading_days
from quantlab.config import ExperimentConfig
from quantlab.contracts import Backtester, BacktestResult, ComponentParams, CostContext, CostModel
from quantlab.data.market_data import MarketData
from quantlab.logging_setup import get_logger
from quantlab.registry import register

log = get_logger(__name__)


@register("backtester", "daily_drift")
class DailyDrift(Backtester):
    """Simulates a portfolio at daily resolution.

    Timing: a target decided at the close of ``t`` is filled at the close of
    ``t + execution_lag_days``. On the fill day the portfolio earns the return
    of the *pre-trade* holdings, then trades; costs are charged on that day.
    Between fills, weights drift with prices. Un-invested weight is cash
    earning ``data.cash_return`` (0% if no cash series is loaded).
    """

    class Params(ComponentParams):
        adv_window: int = Field(default=20, ge=1)
        vol_window: int = Field(default=60, ge=2)

    params: Params

    def market_state(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ADV (USD) and daily vol usable on day d: measured through d-1 (shift(1))."""
        p = self.params
        adv = (data.close * data.volume).rolling(p.adv_window, min_periods=p.adv_window).mean()
        sig = data.returns().rolling(p.vol_window, min_periods=p.vol_window).std()
        return adv.shift(1), sig.shift(1)

    @staticmethod
    def fill_schedule(
        targets: Mapping[pd.Timestamp, pd.Series], data: MarketData, lag: int
    ) -> dict[pd.Timestamp, pd.Series]:
        """Map each decision to its fill date; decisions filling beyond the data are dropped."""
        fills: dict[pd.Timestamp, pd.Series] = {}
        for t, target in sorted(targets.items()):
            fill = shift_trading_days(data.dates, pd.Timestamp(t), lag)
            if fill is None:
                log.info("target dropped: fill date beyond data", extra={"decision": str(t)})
                continue
            fills[fill] = target.reindex(data.tickers).fillna(0.0)
        return fills

    def run(
        self,
        targets: Mapping[pd.Timestamp, pd.Series],
        data: MarketData,
        cost_model: CostModel,
        config: ExperimentConfig,
        start: pd.Timestamp | None = None,
    ) -> BacktestResult:
        dates, tickers = data.dates, data.tickers
        fills = self.fill_schedule(targets, data, config.execution.execution_lag_days)
        if not fills:
            raise ValueError("no executable targets")

        first = min(fills)
        if start is not None and first < start:
            raise ValueError("a target fills before the evaluation start")

        rets = data.returns()
        adv, sig = self.market_state(data)

        sim = dates[dates >= first]
        r_mat = rets.loc[sim].to_numpy(dtype=float)
        cash = data.cash().loc[sim]
        if cash.isna().any():
            raise ValueError("cash return series has gaps inside the simulation window")
        rc = cash.to_numpy(dtype=float)
        w = np.zeros(len(tickers))
        nav = 1.0
        gross_out = np.empty(len(sim))
        net_out = np.empty(len(sim))
        cost_out = np.zeros(len(sim))
        w_out = np.empty((len(sim), len(tickers)))
        turnover: dict[pd.Timestamp, float] = {}
        aum = config.fund.aum_usd

        for i, d in enumerate(sim):
            r = r_mat[i]
            held = np.abs(w) > 0
            if np.isnan(r[held]).any():
                bad = [tickers[j] for j in np.flatnonzero(held & np.isnan(r))]
                raise ValueError(f"{d.date()}: holding assets without a return: {bad}")
            r = np.nan_to_num(r, nan=0.0)
            gross = float(w @ r) + (1.0 - float(w.sum())) * rc[i]
            w = w * (1 + r) / (1 + gross)
            cost = 0.0
            if d in fills:
                tgt = fills[d].to_numpy(dtype=float)
                trades = pd.Series(tgt - w, index=tickers)
                ctx = CostContext(
                    trade_date=d,
                    nav_usd=aum * nav * (1 + gross),
                    adv_usd=cast(pd.Series, adv.loc[d]),
                    daily_vol=cast(pd.Series, sig.loc[d]),
                    asset_class=data.asset_class,
                    config=config.costs,
                )
                cost = float(cost_model.cost(trades[trades != 0], ctx).sum())
                turnover[d] = float(trades.abs().sum() / 2)
                w = tgt
            net = (1 + gross) * (1 - cost) - 1
            nav *= 1 + net
            gross_out[i], net_out[i], cost_out[i] = gross, net, cost
            w_out[i] = w

        target_frame = pd.DataFrame(fills).T.reindex(columns=tickers)
        target_frame.index.name = "fill_date"
        return BacktestResult(
            gross_returns=pd.Series(gross_out, index=sim, name="gross"),
            net_returns=pd.Series(net_out, index=sim, name="net"),
            costs=pd.Series(cost_out, index=sim, name="cost"),
            weights=pd.DataFrame(w_out, index=sim, columns=tickers),
            targets=target_frame,
            turnover=pd.Series(turnover, name="turnover", dtype=float),
            cash_returns=cash.rename("cash"),
        )
