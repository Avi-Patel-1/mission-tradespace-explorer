from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_study
from .campaign import compare_campaigns, expand_campaign, replay_run, run_campaign, summarize_campaign_study
from .config import ConfigError, apply_overrides, load_config, starter_config, strip_internal, validate_config
from .config_tools import diff_configs, freeze_config, generate_template
from .deep_sensitivity import write_deep_sensitivity
from .mission_models import available_models
from .optimization import optimize_config
from .pareto import write_pareto
from .reliability import analyze_reliability, run_stress_test, write_margin_report
from .reports import rewrite_report_from_study
from .runner import run_monte_carlo, run_resume
from .storage import export_sqlite
from .surrogate import fit_surrogate, predict_surrogate
from .sweep import run_sweep
from .uq import write_uq_artifacts


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _load_with_overrides(args: argparse.Namespace) -> dict:
    return apply_overrides(
        load_config(args.config),
        seed=getattr(args, "seed", None),
        sampler=getattr(args, "sampler", None),
        write_sqlite=getattr(args, "write_sqlite", None),
    )


def _parse_objectives(raw_values: list[str] | None) -> list[dict[str, str]] | None:
    if not raw_values:
        return None
    objectives = []
    for raw in raw_values:
        if ":" not in raw:
            raise ValueError(f"invalid objective '{raw}', expected metric:min or metric:max")
        metric, sense = raw.split(":", 1)
        if sense not in {"min", "max"}:
            raise ValueError(f"invalid objective sense '{sense}', expected min or max")
        objectives.append({"metric": metric, "sense": sense})
    return objectives


def _parse_constraints(raw_values: list[str] | None, *, allow_failures: bool) -> dict:
    constraints: dict[str, object] = {"require_success": not allow_failures}
    for raw in raw_values or []:
        if "<=" in raw:
            metric, value = raw.split("<=", 1)
            constraints.setdefault(metric, {})["max"] = float(value)
        elif ">=" in raw:
            metric, value = raw.split(">=", 1)
            constraints.setdefault(metric, {})["min"] = float(value)
        elif "=" in raw:
            metric, value = raw.split("=", 1)
            constraints.setdefault(metric, {})["equals"] = value
        else:
            raise ValueError(f"invalid constraint '{raw}', expected metric<=value, metric>=value, or metric=value")
    return constraints


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tradespace", description="Mission Monte Carlo and trade-space analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate a study configuration.")
    validate.add_argument("--config", required=True)

    models = sub.add_parser("list-models", help="List available mission model names.")

    inspect = sub.add_parser("inspect-config", help="Print the resolved configuration.")
    inspect.add_argument("--config", required=True)

    init = sub.add_parser("init-config", help="Write a starter configuration file.")
    init.add_argument("--out", required=True)
    init.add_argument("--kind", choices=["monte_carlo", "sweep"], default="monte_carlo")
    init.add_argument("--name", default="new_study")
    init.add_argument("--force", action="store_true")

    config_diff = sub.add_parser("config-diff", help="Diff two resolved configurations.")
    config_diff.add_argument("--left", required=True)
    config_diff.add_argument("--right", required=True)

    freeze = sub.add_parser("freeze-config", help="Write a resolved config plus hash.")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--out", required=True)

    template = sub.add_parser("generate-template", help="Write a config template.")
    template.add_argument("--out", required=True)
    template.add_argument("--kind", choices=["monte_carlo", "sweep"], default="monte_carlo")
    template.add_argument("--name", default="template")
    template.add_argument("--model", default="point_mass")

    run = sub.add_parser("run", help="Run a Monte Carlo campaign.")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--runs", type=int, default=None)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    run.add_argument("--sampler", default=None, help="Override sampler.method.")
    run.add_argument("--no-sqlite", dest="write_sqlite", action="store_false", default=None, help="Skip SQLite run storage.")

    sweep = sub.add_parser("sweep", help="Run a trade-space sweep.")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--out", required=True)
    sweep.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    sweep.add_argument("--sampler", default=None, help="Override sampler.method for uncertainty runs.")
    sweep.add_argument("--uncertainty-runs", type=int, default=None, help="Override sweep uncertainty runs per design.")

    robustness = sub.add_parser("robustness", help="Run a robustness Monte Carlo campaign.")
    robustness.add_argument("--config", required=True)
    robustness.add_argument("--out", required=True)
    robustness.add_argument("--runs", type=int, default=None)
    robustness.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    robustness.add_argument("--sampler", default=None, help="Override sampler.method.")
    robustness.add_argument("--no-sqlite", dest="write_sqlite", action="store_false", default=None, help="Skip SQLite run storage.")

    pareto = sub.add_parser("pareto", help="Compute Pareto and dominance rankings from a CSV.")
    pareto.add_argument("--input", required=True)
    pareto.add_argument("--out", required=True)
    pareto.add_argument("--objective", action="append", help="Objective as metric:min or metric:max. Can be repeated.")
    pareto.add_argument("--constraint", action="append", help="Constraint as metric<=value, metric>=value, or metric=value. Can be repeated.")
    pareto.add_argument("--allow-failures", action="store_true", help="Do not require success=true during feasibility filtering.")

    report = sub.add_parser("report", help="Regenerate a static report for a study directory.")
    report.add_argument("--study", required=True)

    audit = sub.add_parser("audit", help="Check a study directory for missing or inconsistent artifacts.")
    audit.add_argument("--study", required=True)

    reliability = sub.add_parser("reliability", help="Compute reliability and robustness diagnostics for a study.")
    reliability.add_argument("--study", required=True)
    reliability.add_argument("--margin-metric", default="robustness_margin")

    stress = sub.add_parser("stress-test", help="Generate and run uncertainty corner stress cases.")
    stress.add_argument("--config", required=True)
    stress.add_argument("--out", required=True)
    stress.add_argument("--max-cases", type=int, default=64)

    margin = sub.add_parser("margin-report", help="Write a margin decomposition report for a study.")
    margin.add_argument("--study", required=True)

    uq = sub.add_parser("uq-report", help="Write bootstrap, jackknife, ECDF, and convergence artifacts.")
    uq.add_argument("--study", required=True)
    uq.add_argument("--metric", default="miss_distance_m")

    sensitivity_deep = sub.add_parser("sensitivity-deep", help="Write expanded sensitivity diagnostics for a study.")
    sensitivity_deep.add_argument("--study", required=True)
    sensitivity_deep.add_argument("--metric", default="miss_distance_m")

    fit_surr = sub.add_parser("fit-surrogate", help="Fit a dependency-light surrogate model from a study.")
    fit_surr.add_argument("--study", required=True)
    fit_surr.add_argument("--out", required=True)
    fit_surr.add_argument("--metric", default="miss_distance_m")
    fit_surr.add_argument("--model-type", choices=["polynomial", "rbf"], default="polynomial")
    fit_surr.add_argument("--degree", type=int, default=2)
    fit_surr.add_argument("--centers", type=int, default=24)

    pred_surr = sub.add_parser("surrogate-predict", help="Predict with a fitted surrogate.")
    pred_surr.add_argument("--model", required=True)
    pred_surr.add_argument("--params", default=None, help="JSON object of feature values.")
    pred_surr.add_argument("--input-csv", default=None)
    pred_surr.add_argument("--out-csv", default=None)

    optimize = sub.add_parser("optimize", help="Run a dependency-light design optimizer.")
    optimize.add_argument("--config", required=True)
    optimize.add_argument("--out", required=True)
    optimize.add_argument("--method", choices=["genetic", "random", "anneal", "simulated_annealing", "coordinate", "pattern", "pattern_search"], default="genetic")
    optimize.add_argument("--iterations", type=int, default=20)
    optimize.add_argument("--population", type=int, default=16)

    rank_designs = sub.add_parser("rank-designs", help="Rank design rows from a CSV using Pareto scoring.")
    rank_designs.add_argument("--input", required=True)
    rank_designs.add_argument("--out", required=True)
    rank_designs.add_argument("--objective", action="append")
    rank_designs.add_argument("--constraint", action="append")
    rank_designs.add_argument("--allow-failures", action="store_true")

    robust_pareto = sub.add_parser("robust-pareto", help="Compute a robust Pareto ranking from aggregate design rows.")
    robust_pareto.add_argument("--input", required=True)
    robust_pareto.add_argument("--out", required=True)

    campaign = sub.add_parser("campaign", help="Run or inspect a scenario campaign manifest.")
    campaign.add_argument("--manifest", required=True)
    campaign.add_argument("--out", default=None)
    campaign.add_argument("--jobs", type=int, default=1)
    campaign.add_argument("--plan-only", action="store_true")
    campaign.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    campaign.add_argument("--retry-failed", action="store_true")

    replay = sub.add_parser("replay-run", help="Replay one deterministic Monte Carlo run from a campaign scenario.")
    replay.add_argument("--campaign", required=True)
    replay.add_argument("--scenario", required=True)
    replay.add_argument("--run-index", type=int, default=0)

    compare = sub.add_parser("compare-campaigns", help="Compare two campaign summary directories.")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--out", required=True)

    resume = sub.add_parser("resume", help="Continue a previously started campaign.")
    resume.add_argument("--study", required=True)
    resume.add_argument("--config", default=None)

    export = sub.add_parser("export", help="Export a study SQLite database to CSV and JSON.")
    export.add_argument("--db", required=True)
    export.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            config = load_config(args.config)
            errors = validate_config(config)
            if errors:
                _print({"valid": False, "errors": errors})
                raise SystemExit(1)
            _print({"valid": True, "config": str(Path(args.config))})
        elif args.command == "list-models":
            _print({"models": available_models()})
        elif args.command == "inspect-config":
            _print(strip_internal(load_config(args.config)))
        elif args.command == "init-config":
            out = Path(args.out)
            if out.exists() and not args.force:
                raise FileExistsError(f"{out} already exists; pass --force to overwrite")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(starter_config(args.kind, args.name), indent=2) + "\n")
            _print({"written": str(out), "kind": args.kind})
        elif args.command == "config-diff":
            _print(diff_configs(args.left, args.right))
        elif args.command == "freeze-config":
            _print(freeze_config(args.config, args.out))
        elif args.command == "generate-template":
            _print(generate_template(args.out, kind=args.kind, name=args.name, model=args.model))
        elif args.command == "run":
            _print(run_monte_carlo(_load_with_overrides(args), args.out, args.runs, resume=args.resume))
        elif args.command == "sweep":
            config = _load_with_overrides(args)
            if args.uncertainty_runs is not None:
                config.setdefault("sweep", {})
                config["sweep"]["uncertainty_runs"] = args.uncertainty_runs
            _print(run_sweep(config, args.out))
        elif args.command == "robustness":
            _print(run_monte_carlo(_load_with_overrides(args), args.out, args.runs))
        elif args.command == "pareto":
            _print(
                write_pareto(
                    args.input,
                    args.out,
                    objectives=_parse_objectives(args.objective),
                    constraints=_parse_constraints(args.constraint, allow_failures=args.allow_failures),
                )
            )
        elif args.command == "report":
            _print(rewrite_report_from_study(args.study))
        elif args.command == "audit":
            result = audit_study(args.study)
            _print(result)
            if not result["ok"]:
                raise SystemExit(1)
        elif args.command == "reliability":
            _print(analyze_reliability(args.study, margin_metric=args.margin_metric))
        elif args.command == "stress-test":
            _print(run_stress_test(args.config, args.out, max_cases=args.max_cases))
        elif args.command == "margin-report":
            _print(write_margin_report(args.study))
        elif args.command == "uq-report":
            _print(write_uq_artifacts(args.study, metric=args.metric))
        elif args.command == "sensitivity-deep":
            _print(write_deep_sensitivity(args.study, metric=args.metric))
        elif args.command == "fit-surrogate":
            _print(
                fit_surrogate(
                    args.study,
                    args.out,
                    metric=args.metric,
                    model_type=args.model_type,
                    degree=args.degree,
                    centers=args.centers,
                )
            )
        elif args.command == "surrogate-predict":
            params = json.loads(args.params) if args.params else None
            _print(predict_surrogate(args.model, params=params, input_csv=args.input_csv, out_csv=args.out_csv))
        elif args.command == "optimize":
            _print(optimize_config(args.config, args.out, method=args.method, iterations=args.iterations, population=args.population))
        elif args.command == "rank-designs":
            _print(
                write_pareto(
                    args.input,
                    args.out,
                    objectives=_parse_objectives(args.objective),
                    constraints=_parse_constraints(args.constraint, allow_failures=args.allow_failures),
                )
            )
        elif args.command == "robust-pareto":
            _print(
                write_pareto(
                    args.input,
                    args.out,
                    objectives=[
                        {"metric": "miss_p95_m", "sense": "min"},
                        {"metric": "robustness_margin", "sense": "max"},
                        {"metric": "success_rate", "sense": "max"},
                    ],
                    constraints={"require_success": False},
                )
            )
        elif args.command == "campaign":
            if args.plan_only:
                _print(expand_campaign(args.manifest, args.out))
            else:
                summary = run_campaign(args.manifest, args.out, resume=args.resume, retry_failed=args.retry_failed, jobs=args.jobs)
                rollup = summarize_campaign_study(summary["out_dir"])
                _print({**summary, "rollup_studies": rollup["studies"]})
        elif args.command == "replay-run":
            _print(replay_run(args.campaign, args.scenario, args.run_index))
        elif args.command == "compare-campaigns":
            _print(compare_campaigns(args.left, args.right, args.out))
        elif args.command == "resume":
            _print(run_resume(args.study, args.config))
        elif args.command == "export":
            _print(export_sqlite(args.db, args.out))
    except (ConfigError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
