"""Name -> implementation registry so configs can select components declaratively."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from quantlab.config import ComponentSpec
from quantlab.contracts import Component

Kind = Literal[
    "data_source",
    "validator",
    "factor",
    "model",
    "signal",
    "portfolio",
    "risk",
    "cost_model",
    "backtester",
    "evaluator",
    "reporter",
]

C = TypeVar("C", bound=type[Component])

_REGISTRY: dict[str, dict[str, type[Component]]] = {}
_BUILTINS_LOADED = False


class UnknownComponentError(KeyError):
    pass


def register(kind: Kind, name: str) -> Callable[[C], C]:
    def deco(cls: C) -> C:
        slot = _REGISTRY.setdefault(kind, {})
        if name in slot and slot[name] is not cls:
            raise ValueError(f"{kind} '{name}' already registered by {slot[name].__qualname__}")
        slot[name] = cls
        return cls

    return deco


def _load_builtins() -> None:
    global _BUILTINS_LOADED  # noqa: PLW0603 - one-time lazy import guard
    if not _BUILTINS_LOADED:
        importlib.import_module("quantlab.components")
        _BUILTINS_LOADED = True


def available(kind: Kind) -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY.get(kind, {}))


def build(kind: Kind, spec: ComponentSpec) -> Any:
    _load_builtins()
    try:
        cls = _REGISTRY[kind][spec.name]
    except KeyError:
        raise UnknownComponentError(
            f"no {kind} named '{spec.name}'; available: {available(kind)}"
        ) from None
    return cls.from_params(spec.params)
