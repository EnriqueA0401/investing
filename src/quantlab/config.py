"""Declarative, validated experiment configuration.

Every tunable of an experiment lives here and nowhere else. Configs are
frozen, reject unknown keys, and have a stable content hash that is used to
identify a configuration in the trial ledger (multiple-testing accounting).

YAML files may inherit from another file via ``extends: <relative path>``;
the child is deep-merged on top of the parent.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentSpec(_Strict):
    """Reference to a registered pipeline component and its parameters."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Asset(_Strict):
    ticker: str
    asset_class: Literal[
        "equity",
        "government_bond",
        "credit",
        "inflation_linked",
        "real_estate",
        "commodity",
        "cash",
    ]
    region: str = "US"
    tradable: bool = Field(
        default=True, description="False = data only (e.g. a cash proxy), never held."
    )


class UniverseConfig(_Strict):
    """Universe fixed ex-ante. Assets enter only once they have traded (see data layer)."""

    assets: tuple[Asset, ...]
    min_history_days: int = Field(
        default=252,
        ge=0,
        description="Trading days of history required before an asset is investable.",
    )
    source: Literal["static", "data"] = Field(
        default="static",
        description="'data': tradable names come from the dataset (single stocks); "
        "listed assets are then reference/non-tradable entries such as the benchmark.",
    )
    use_membership: bool = Field(
        default=False, description="Restrict to the dataset's point-in-time universe membership."
    )
    min_price: float = Field(default=0.0, ge=0)
    min_adv_usd: float = Field(default=0.0, ge=0, description="63-day average dollar volume floor.")
    exclude_asset_classes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _unique_tickers(self) -> UniverseConfig:
        tickers = [a.ticker for a in self.assets]
        if len(tickers) != len(set(tickers)):
            raise ValueError("duplicate tickers in universe")
        return self

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(a.ticker for a in self.assets)

    @property
    def tradable_tickers(self) -> tuple[str, ...]:
        return tuple(a.ticker for a in self.assets if a.tradable)


class SplitConfig(_Strict):
    """Temporal partition. Everything from ``holdout_start`` on is sealed until final evaluation.

    ``data_start`` .. ``research_start`` is warm-up history for look-back windows only.
    """

    data_start: date
    research_start: date
    research_end: date
    holdout_start: date
    holdout_end: date | None = None
    embargo_days: int = Field(
        default=21, ge=0, description="Trading-day embargo for purged CV / walk-forward (Phase 5)."
    )

    @model_validator(mode="after")
    def _ordered(self) -> SplitConfig:
        if not self.data_start <= self.research_start < self.research_end < self.holdout_start:
            raise ValueError("require data_start <= research_start < research_end < holdout_start")
        if self.holdout_end is not None and self.holdout_end <= self.holdout_start:
            raise ValueError("holdout_end must be after holdout_start")
        return self


class ExecutionConfig(_Strict):
    """Decision/execution timing convention.

    Signals are formed with information up to and including the close of the
    rebalance date ``t``; trades fill at the close ``execution_lag_days``
    trading days later. A lag of 0 is rejected: it implies trading at the same
    close used to form the signal, which is not achievable in practice.
    """

    rebalance_frequency: Literal["W", "ME", "QE"] = "ME"
    execution_lag_days: int = Field(default=1, ge=1)


class CostConfig(_Strict):
    """Transaction-cost assumptions, in basis points of traded notional."""

    half_spread_bps: dict[str, float] = Field(
        default_factory=lambda: {"default": 2.0},
        description="Half bid-ask spread per asset class; key 'default' is the fallback.",
    )
    commission_bps: float = Field(default=0.5, ge=0)
    slippage_bps: float = Field(default=1.0, ge=0)
    impact_coefficient: float = Field(
        default=0.1, ge=0, description="Square-root impact: k * sigma_daily * sqrt(trade / ADV)."
    )
    borrow_fee_bps_annual: float = Field(
        default=0.0, ge=0, description="Stock-loan fee on short notional, charged daily."
    )

    @model_validator(mode="after")
    def _has_default(self) -> CostConfig:
        if "default" not in self.half_spread_bps:
            raise ValueError("half_spread_bps must contain a 'default' entry")
        if any(v < 0 for v in self.half_spread_bps.values()):
            raise ValueError("spreads must be non-negative")
        return self


class PortfolioConstraints(_Strict):
    long_only: bool = True
    max_gross_leverage: float = Field(default=1.0, gt=0)
    max_weight: float = Field(default=0.20, gt=0, le=1)
    min_weight: float = Field(default=0.0, ge=0)
    max_turnover: float | None = Field(
        default=None, gt=0, description="One-way turnover cap per rebalance (sum |dw| / 2)."
    )
    max_asset_class_weight: dict[str, float] = Field(default_factory=dict)
    max_net_exposure: float | None = Field(
        default=None, ge=0, description="|sum of weights| cap (market-neutral books)."
    )

    @model_validator(mode="after")
    def _consistent(self) -> PortfolioConstraints:
        if self.long_only and self.max_gross_leverage > 1.0:
            raise ValueError("long-only mandate is unlevered: max_gross_leverage must be <= 1")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight > max_weight")
        return self


class FundConfig(_Strict):
    aum_usd: float = Field(default=1_000_000.0, gt=0)
    base_currency: Literal["USD"] = "USD"
    target_volatility: float | None = Field(default=0.10, gt=0)


class BenchmarkConfig(_Strict):
    weights: dict[str, float]
    rebalance_frequency: Literal["ME", "QE"] = "ME"

    @model_validator(mode="after")
    def _sums_to_one(self) -> BenchmarkConfig:
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("benchmark weights must sum to 1")
        return self


class PipelineConfig(_Strict):
    """Which concrete implementation fills each contract slot."""

    data_source: ComponentSpec
    validators: tuple[ComponentSpec, ...] = ()
    factors: tuple[ComponentSpec, ...] = ()
    model: ComponentSpec | None = None
    signal: ComponentSpec
    portfolio: ComponentSpec
    risk: tuple[ComponentSpec, ...] = ()
    cost_model: ComponentSpec
    backtester: ComponentSpec = ComponentSpec(name="daily_drift")
    evaluator: ComponentSpec
    reporter: ComponentSpec | None = None


class TrackingConfig(_Strict):
    runs_dir: Path = Path("runs")
    ledger_path: Path = Path("research/trials.jsonl")
    snapshot_dir: Path = Path("data/snapshots")


class ExperimentConfig(_Strict):
    name: str
    hypothesis_ids: tuple[str, ...] = Field(
        default=(), description="Pre-registered hypotheses (docs/hypotheses) this run tests."
    )
    seed: int = 20260101
    data_snapshot: str | None = Field(
        default=None, description="Content hash of a frozen data snapshot; None = fetch live."
    )
    universe: UniverseConfig
    split: SplitConfig
    execution: ExecutionConfig = ExecutionConfig()
    costs: CostConfig = CostConfig()
    constraints: PortfolioConstraints = PortfolioConstraints()
    fund: FundConfig = FundConfig()
    benchmark: BenchmarkConfig
    baseline: ComponentSpec = Field(
        default=ComponentSpec(name="equal_weight"),
        description="'Dumb' baseline portfolio computed in every run under the same mandate.",
    )
    pipeline: PipelineConfig
    tracking: TrackingConfig = TrackingConfig()

    @model_validator(mode="after")
    def _benchmark_in_universe(self) -> ExperimentConfig:
        missing = set(self.benchmark.weights) - set(self.universe.tickers)
        if missing:
            raise ValueError(f"benchmark tickers not in universe: {sorted(missing)}")
        return self

    def config_hash(self) -> str:
        """Stable hash of the full config, excluding the free-text name.

        Fields added after configs were first ledgered are omitted while they hold
        their default value, so earlier configurations keep their hashes."""
        payload = self.model_dump(mode="json", exclude={"name"})
        for section, key, default in _LATER_FIELDS:
            if payload.get(section, {}).get(key) == default:
                payload[section].pop(key)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# (section, field, default) added after the ETF study was ledgered.
_LATER_FIELDS: tuple[tuple[str, str, object], ...] = (
    ("universe", "source", "static"),
    ("universe", "use_membership", False),
    ("universe", "min_price", 0.0),
    ("universe", "min_adv_usd", 0.0),
    ("universe", "exclude_asset_classes", []),
    ("constraints", "max_net_exposure", None),
    ("costs", "borrow_fee_bps_annual", 0.0),
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular 'extends' chain at {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(_load_raw(path.parent / parent, _seen | {path}), raw)


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_load_raw(Path(path)))
