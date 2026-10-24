"""Abstract contracts for every pipeline stage.

DataSource -> DataValidator -> Factor -> (Model) -> SignalGenerator ->
PortfolioConstructor -> RiskManager -> Backtester (+ CostModel) -> Evaluator -> Reporter

Temporal-integrity rule enforced by the orchestrator: any component invoked at
decision date ``t`` receives ``MarketData`` already truncated with
``data.asof(t)``. Components must not obtain data by any other route.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from quantlab.config import Asset, CostConfig, ExperimentConfig, PortfolioConstraints
from quantlab.data.market_data import MarketData


class ComponentParams(BaseModel):
    """Base for component parameter models: strict and immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Component:
    """Base class: components are built from validated ``Params``."""

    Params: ClassVar[type[ComponentParams]] = ComponentParams

    def __init__(self, params: ComponentParams) -> None:
        self.params = params

    @classmethod
    def from_params(cls, raw: Mapping[str, Any]) -> Self:
        return cls(cls.Params.model_validate(dict(raw)))


# --------------------------------------------------------------------------- data


class DataSource(Component, ABC):
    @abstractmethod
    def fetch(self, assets: Sequence[Asset], start: date, end: date | None) -> MarketData: ...


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    severity: Severity
    message: str
    ticker: str | None = None
    dates: tuple[str, ...] = ()


class DataValidator(Component, ABC):
    @abstractmethod
    def validate(self, data: MarketData) -> list[ValidationIssue]: ...


# --------------------------------------------------------------------- features


class Factor(Component, ABC):
    """A feature grounded in a pre-registered hypothesis.

    Subclasses MUST document in their docstring: the economic/behavioural
    thesis, the expected sign of the relation with forward returns, and the
    risk or bias it is supposed to be compensating/exploiting.
    """

    hypothesis_id: ClassVar[str]
    precomputable: ClassVar[bool] = False
    """True only for factors whose ``assert_causal`` test exists: the orchestrator may
    then compute the panel once on the stage view and slice it at each decision
    (identical values by construction, verified by that test)."""

    @property
    def label(self) -> str:
        """Key under which the panel is exposed in ``DecisionContext.features``."""
        return str(getattr(self.params, "label", type(self).__name__))

    @abstractmethod
    def compute(self, history: MarketData) -> pd.DataFrame:
        """Dates x tickers panel. Row ``d`` may use only rows <= ``d`` of ``history``."""


class Model(Component, ABC):
    """Optional learned mapping features -> expected-return scores.

    ``fit`` receives only samples whose label horizon ends on or before the
    decision date (purged); the orchestrator is responsible for the purge.
    """

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None: ...

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series: ...


# ---------------------------------------------------------------- decisions


@dataclass(frozen=True)
class DecisionContext:
    """Everything a decision-stage component may see at decision date ``as_of``."""

    as_of: pd.Timestamp
    history: MarketData
    investable: list[str]
    previous_weights: pd.Series
    constraints: PortfolioConstraints
    rng: np.random.Generator
    features: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    current_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    """Holdings at ``as_of`` close: previous target drifted with prices (pre-trade)."""

    def __post_init__(self) -> None:
        if self.history.end > self.as_of:
            end, as_of = self.history.end.date(), self.as_of.date()
            raise ValueError(f"look-ahead: history ends {end} after as_of {as_of}")


class SignalGenerator(Component, ABC):
    @abstractmethod
    def generate(self, ctx: DecisionContext) -> pd.Series:
        """Scores indexed by (a subset of) ``ctx.investable``; higher = more attractive."""


class PortfolioConstructor(Component, ABC):
    @abstractmethod
    def construct(self, scores: pd.Series, ctx: DecisionContext) -> pd.Series:
        """Target weights (fraction of NAV) indexed by ticker; remainder is cash."""


class RiskManager(Component, ABC):
    @abstractmethod
    def adjust(self, weights: pd.Series, ctx: DecisionContext) -> pd.Series: ...


# ------------------------------------------------------------- simulation


@dataclass(frozen=True)
class CostContext:
    """Market state known *before* trading on ``trade_date``."""

    trade_date: pd.Timestamp
    nav_usd: float
    adv_usd: pd.Series
    daily_vol: pd.Series
    asset_class: Mapping[str, str]
    config: CostConfig


class CostModel(Component, ABC):
    @abstractmethod
    def cost(self, trades: pd.Series, ctx: CostContext) -> pd.Series:
        """Cost per ticker as a fraction of NAV for weight changes ``trades``."""


@dataclass(frozen=True)
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    weights: pd.DataFrame
    targets: pd.DataFrame
    turnover: pd.Series
    cash_returns: pd.Series

    def __post_init__(self) -> None:
        if not self.net_returns.index.equals(self.gross_returns.index):
            raise ValueError("gross/net return indices differ")
        if not self.cash_returns.index.equals(self.net_returns.index):
            raise ValueError("cash/net return indices differ")

    def excess_returns(self) -> pd.Series:
        return self.net_returns - self.cash_returns


class Backtester(Component, ABC):
    @abstractmethod
    def run(
        self,
        targets: Mapping[pd.Timestamp, pd.Series],
        data: MarketData,
        cost_model: CostModel,
        config: ExperimentConfig,
        start: pd.Timestamp | None = None,
    ) -> BacktestResult: ...


class Evaluator(Component, ABC):
    @abstractmethod
    def evaluate(
        self, result: BacktestResult, benchmark: BacktestResult | None = None
    ) -> dict[str, float]: ...


class Reporter(Component, ABC):
    @abstractmethod
    def report(self, payload: Mapping[str, Any], out_dir: str) -> str:
        """Write a report and return its path."""
