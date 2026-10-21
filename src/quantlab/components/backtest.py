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
    (0% return until a cash-rate series is wired in Phase 2).
    """

    class Params(ComponentParams):
        adv_window: int = Field(default=20, ge=1)
        vol_window: int = Field(default=60, ge=2)

    params: Params

    def run(
        self,
        targets: Mapping[pd.Timestamp, pd.Series],
        data: MarketData,
        cost_model: CostModel,
        config: ExperimentConfig,
        start: pd.Timestamp | None = None,
    ) -> BacktestResult:
        dates, tickers = data.dates, data.tickers
        lag = config.execution.execution_lag_days
        fills: dict[pd.Timestamp, pd.Series] = {}
        for t, target in sorted(targets.items()):
            fill = shift_trading_days(dates, pd.Timestamp(t), lag)
            if fill is None:
                log.info("target dropped: fill date beyond data", extra={"decision": str(t)})
                continue
            fills[fill] = target.reindex(tickers).fillna(0.0)
        if not fills:
            raise ValueError("no executable targets")

        first = min(fills)
        if start is not None and first < start:
            raise ValueError("a target fills before the evaluation start")

        rets = data.returns()
        # Market state for costs on day d is measured through d-1 (shift(1)).
        dollar_vol = data.close * data.volume
        adv = (
            dollar_vol.rolling(self.params.adv_window, min_periods=self.params.adv_window)
            .mean()
            .shift(1)
        )
        sig = (
            rets.rolling(self.params.vol_window, min_periods=self.params.vol_window).std().shift(1)
        )

        sim = dates[dates >= first]
        r_mat = rets.loc[sim].to_numpy(dtype=float)
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
            gross = float(w @ r)
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
        )
