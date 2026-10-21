"""Command-line entry point: ``quantlab <command> ...``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from quantlab import registry
from quantlab.analysis import evaluate_study
from quantlab.config import load_config
from quantlab.data.quality import data_quality_report
from quantlab.logging_setup import configure_logging
from quantlab.pipeline import load_data, run_experiment
from quantlab.research_run import run_holdout, run_robustness
from quantlab.study import load_study, run_study
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


def _cmd_data_report(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data, snap_id = load_data(cfg)
    validators = [registry.build("validator", s) for s in cfg.pipeline.validators]
    report = data_quality_report(data, cfg, snap_id, validators)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report)
    print(args.out)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.stage == "final" and not args.unlock_holdout:
        raise SystemExit("final stage requires --unlock-holdout '<reason>' (this is ledgered)")
    result = run_experiment(cfg, stage=args.stage, unlock_reason=args.unlock_holdout)
    print(json.dumps({"run_id": result.run_id, **result.metrics}, indent=2))
    return 0


def _cmd_study(args: argparse.Namespace) -> int:
    study = load_study(args.study)
    results = run_study(study, only=args.only or None)
    cols = [
        "sharpe",
        "base_sharpe",
        "bench_sharpe",
        "cagr",
        "vol",
        "max_drawdown",
        "turnover_annual",
    ]
    table = {k: {c: round(r.metrics[c], 3) for c in cols} for k, r in results.items()}
    print(json.dumps(table, indent=1))
    return 0


def _cmd_validate_study(args: argparse.Namespace) -> int:
    out = Path(args.results)
    verdict = evaluate_study(out, TrialLedger(Path(args.ledger)), n_boot=args.n_boot)
    (out / "validation.json").write_text(verdict.to_json())
    print(verdict.to_json())
    return 0


def _selected(results: Path) -> str:
    return str(json.loads((results / "validation.json").read_text())["selected"])


def _cmd_robustness(args: argparse.Namespace) -> int:
    study = load_study(args.study)
    out = run_robustness(study, _selected(study.output_dir), study.output_dir)
    print(json.dumps(out, indent=2, default=float))
    return 0


def _cmd_holdout(args: argparse.Namespace) -> int:
    study = load_study(args.study)
    selected = _selected(study.output_dir)
    if args.experiment != selected:
        raise SystemExit(f"pre-registered rule selected {selected}, not {args.experiment}")
    res = run_holdout(study, selected, args.unlock_holdout, study.output_dir)
    print(json.dumps(res.metrics, indent=2))
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

    q = sub.add_parser("data-report", help="data-quality report (holdout: structural only)")
    q.add_argument("config")
    q.add_argument("--out", default="docs/data_quality.md")
    q.set_defaults(func=_cmd_data_report)

    r = sub.add_parser("run", help="run an experiment")
    r.add_argument("config")
    r.add_argument("--stage", choices=["research", "final"], default="research")
    r.add_argument("--unlock-holdout", metavar="REASON", default=None)
    r.set_defaults(func=_cmd_run)

    st = sub.add_parser("study", help="run the pre-registered experiment set (research stage)")
    st.add_argument("study")
    st.add_argument("--only", nargs="*", default=None)
    st.set_defaults(func=_cmd_study)

    vs = sub.add_parser("validate-study", help="apply pre-registered acceptance criteria")
    vs.add_argument("--results", default="research/results/study")
    vs.add_argument("--ledger", default="research/trials.jsonl")
    vs.add_argument("--n-boot", type=int, default=10_000)
    vs.set_defaults(func=_cmd_validate_study)

    rb = sub.add_parser("robustness", help="post-selection robustness on the research window")
    rb.add_argument("study")
    rb.set_defaults(func=_cmd_robustness)

    ho = sub.add_parser("holdout", help="THE single holdout evaluation of the selected config")
    ho.add_argument("study")
    ho.add_argument("--experiment", required=True)
    ho.add_argument("--unlock-holdout", required=True, metavar="REASON")
    ho.set_defaults(func=_cmd_holdout)

    t = sub.add_parser("trials", help="multiple-testing trial counts from the ledger")
    t.add_argument("--ledger", default="research/trials.jsonl")
    t.add_argument("--hypothesis", default=None)
    t.set_defaults(func=_cmd_trials)

    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return int(args.func(args))
