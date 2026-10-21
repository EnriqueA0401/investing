"""Command-line entry point: ``quantlab <command> ...``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from quantlab import registry
from quantlab.config import load_config
from quantlab.logging_setup import configure_logging
from quantlab.pipeline import load_data, run_experiment
from quantlab.tracking import TrialLedger


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    p = cfg.pipeline
    slots: list[tuple[registry.Kind, str]] = [
        ("data_source", p.data_source.name),
        ("signal", p.signal.name),
        ("portfolio", p.portfolio.name),
        ("cost_model", p.cost_model.name),
        ("backtester", p.backtester.name),
        ("evaluator", p.evaluator.name),
        *[("validator", s.name) for s in p.validators],
        *[("factor", s.name) for s in p.factors],
        *[("risk", s.name) for s in p.risk],
    ]
    missing = [f"{k}:{n}" for k, n in slots if n not in registry.available(k)]
    print(json.dumps({"config_hash": cfg.config_hash(), "unresolved_components": missing}))
    return 1 if missing else 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    _, snap_id = load_data(load_config(args.config))
    print(snap_id)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.stage == "final" and not args.unlock_holdout:
        raise SystemExit("final stage requires --unlock-holdout '<reason>' (this is ledgered)")
    result = run_experiment(cfg, stage=args.stage, unlock_reason=args.unlock_holdout)
    print(json.dumps({"run_id": result.run_id, **result.metrics}, indent=2))
    return 0


def _cmd_trials(args: argparse.Namespace) -> int:
    ledger = TrialLedger(Path(args.ledger))
    print(
        json.dumps(
            {
                "n_trials_research": ledger.n_trials(args.hypothesis, "research"),
                "n_trials_final": ledger.n_trials(args.hypothesis, "final"),
                "holdout_unlocks": len(ledger.holdout_unlocks()),
            }
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantlab")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a config and resolve its components")
    v.add_argument("config")
    v.set_defaults(func=_cmd_validate)

    s = sub.add_parser("snapshot", help="fetch data and freeze it as a content-addressed snapshot")
    s.add_argument("config")
    s.set_defaults(func=_cmd_snapshot)

    r = sub.add_parser("run", help="run an experiment")
    r.add_argument("config")
    r.add_argument("--stage", choices=["research", "final"], default="research")
    r.add_argument("--unlock-holdout", metavar="REASON", default=None)
    r.set_defaults(func=_cmd_run)

    t = sub.add_parser("trials", help="multiple-testing trial counts from the ledger")
    t.add_argument("--ledger", default="research/trials.jsonl")
    t.add_argument("--hypothesis", default=None)
    t.set_defaults(func=_cmd_trials)

    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return int(args.func(args))
