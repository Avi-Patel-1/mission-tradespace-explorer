from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_config, validate_or_raise
from .distributions import _normal_ppf
from .mission_models import simulate_model
from .reports import read_csv, svg_hist, svg_scatter, write_csv, write_json
from .samplers import run_seed
from .uq import rare_event_probability, wilson_interval


def _as_bool(value: Any) -> bool:
    return value in {True, "True", "true", 1, "1", "yes"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _numeric_sample_names(samples: list[dict[str, Any]]) -> list[str]:
    excluded = {"run", "run_seed", "case", "replicate"}
    return sorted({key for row in samples for key, value in row.items() if key not in excluded and isinstance(value, (int, float))})


def _series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([_as_float(row.get(key), math.nan) for row in rows], dtype=float)


def probability_of_violation(results: list[dict[str, Any]], constraints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    constraints = constraints or {
        "miss_violation_m": {"threshold": 0.0},
        "qbar_violation_pa": {"threshold": 0.0},
        "load_violation_g": {"threshold": 0.0},
    }
    rows = []
    n = max(len(results), 1)
    for metric, rule in constraints.items():
        threshold = float(rule.get("threshold", 0.0)) if isinstance(rule, dict) else 0.0
        hits = sum(1 for row in results if _as_float(row.get(metric), 0.0) > threshold)
        low, high = wilson_interval(hits, n)
        rows.append({"constraint": metric, "violations": hits, "runs": len(results), "probability": hits / n, "low": low, "high": high})
    return rows


def reliability_index(success_probability: float) -> float:
    p = min(max(float(success_probability), 1e-12), 1.0 - 1e-12)
    return float(_normal_ppf(p))


def failure_mode_ranking(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("failure_reason", "none")) for row in results if not _as_bool(row.get("success")))
    total_failures = max(sum(counts.values()), 1)
    return [
        {"failure_reason": reason, "count": count, "fraction_of_failures": count / total_failures, "fraction_of_runs": count / max(len(results), 1)}
        for reason, count in counts.most_common()
    ]


def margin_decomposition(samples: list[dict[str, Any]], results: list[dict[str, Any]], margin_metric: str = "robustness_margin") -> list[dict[str, Any]]:
    names = _numeric_sample_names(samples)
    margin = _series(results, margin_metric)
    rows = []
    for name in names:
        x = _series(samples, name)
        mask = np.isfinite(x) & np.isfinite(margin)
        if int(mask.sum()) < 4 or float(np.std(x[mask])) <= 1e-12:
            continue
        corr = float(np.corrcoef(x[mask], margin[mask])[0, 1])
        low, high = np.percentile(x[mask], [10, 90])
        low_margin = float(np.median(margin[mask][x[mask] <= low])) if np.any(x[mask] <= low) else float(np.median(margin[mask]))
        high_margin = float(np.median(margin[mask][x[mask] >= high])) if np.any(x[mask] >= high) else float(np.median(margin[mask]))
        rows.append(
            {
                "parameter": name,
                "margin_metric": margin_metric,
                "correlation": corr,
                "low_decile_margin": low_margin,
                "high_decile_margin": high_margin,
                "margin_shift": high_margin - low_margin,
                "score": abs(corr) + 0.01 * abs(high_margin - low_margin),
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows


def form_like_index(samples: list[dict[str, Any]], results: list[dict[str, Any]], margin_metric: str = "robustness_margin") -> dict[str, Any]:
    names = _numeric_sample_names(samples)
    if not names:
        margins = _series(results, margin_metric)
        mean = float(np.nanmean(margins)) if len(margins) else math.nan
        std = float(np.nanstd(margins)) if len(margins) else math.nan
        return {"margin_metric": margin_metric, "beta": mean / max(std, 1e-12), "design_point": {}}

    x = np.column_stack([_series(samples, name) for name in names])
    y = _series(results, margin_metric)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if int(mask.sum()) < len(names) + 3:
        mean = float(np.nanmean(y))
        std = float(np.nanstd(y))
        return {"margin_metric": margin_metric, "beta": mean / max(std, 1e-12), "design_point": {}, "note": "insufficient samples for linearized FORM"}

    x = x[mask]
    y = y[mask]
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma <= 1e-12] = 1.0
    z = (x - mu) / sigma
    design = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(beta[0])
    gradient = beta[1:]
    grad_norm = float(np.linalg.norm(gradient))
    reliability = intercept / max(grad_norm, 1e-12)
    design_point_z = -reliability * gradient / max(grad_norm, 1e-12)
    return {
        "margin_metric": margin_metric,
        "beta": float(reliability),
        "intercept_margin": intercept,
        "gradient_norm": grad_norm,
        "design_point": {name: float(mu[i] + design_point_z[i] * sigma[i]) for i, name in enumerate(names)},
        "importance_factors": {name: float((gradient[i] / max(grad_norm, 1e-12)) ** 2) for i, name in enumerate(names)},
    }


def sorm_like_adjustment(form: dict[str, Any], results: list[dict[str, Any]], margin_metric: str = "robustness_margin") -> dict[str, Any]:
    margins = _series(results, margin_metric)
    margins = margins[np.isfinite(margins)]
    if len(margins) < 4:
        return {"beta_sorm": form.get("beta", math.nan), "curvature_proxy": math.nan}
    centered = margins - float(np.mean(margins))
    std = float(np.std(margins))
    curvature = float(np.mean(centered**3) / max(std**3, 1e-12))
    beta_form = float(form.get("beta", 0.0))
    return {"beta_sorm": float(beta_form - 0.1 * curvature), "curvature_proxy": curvature}


def failure_mode_clusters(samples: list[dict[str, Any]], results: list[dict[str, Any]], max_clusters: int = 4) -> list[dict[str, Any]]:
    names = _numeric_sample_names(samples)
    failures = [(sample, result) for sample, result in zip(samples, results) if not _as_bool(result.get("success"))]
    if not names or not failures:
        return []
    x = np.array([[_as_float(sample.get(name), 0.0) for name in names] for sample, _ in failures], dtype=float)
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma <= 1e-12] = 1.0
    xz = (x - mu) / sigma
    k = min(max_clusters, len(xz))
    centers = xz[np.linspace(0, len(xz) - 1, k).astype(int)].copy()
    labels = np.zeros(len(xz), dtype=int)
    for _ in range(8):
        distances = np.linalg.norm(xz[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        for cluster in range(k):
            if np.any(labels == cluster):
                centers[cluster] = np.mean(xz[labels == cluster], axis=0)
    rows = []
    for cluster in range(k):
        idx = np.where(labels == cluster)[0]
        if len(idx) == 0:
            continue
        reasons = Counter(str(failures[i][1].get("failure_reason", "failed")) for i in idx)
        center_real = mu + centers[cluster] * sigma
        rows.append(
            {
                "cluster": cluster,
                "count": int(len(idx)),
                "dominant_failure_reason": reasons.most_common(1)[0][0],
                **{f"center_{name}": float(center_real[i]) for i, name in enumerate(names)},
            }
        )
    return rows


def analyze_reliability(study_dir: str | Path, *, margin_metric: str = "robustness_margin") -> dict[str, Any]:
    study = Path(study_dir)
    results = read_csv(study / "results.csv")
    samples = read_csv(study / "samples.csv")
    successes = sum(1 for row in results if _as_bool(row.get("success")))
    p_success = successes / max(len(results), 1)
    success_low, success_high = wilson_interval(successes, len(results))
    form = form_like_index(samples, results, margin_metric)
    sorm = sorm_like_adjustment(form, results, margin_metric)
    margins = _series(results, margin_metric)
    reliability = {
        "runs": len(results),
        "successes": successes,
        "success_probability": p_success,
        "success_probability_low": success_low,
        "success_probability_high": success_high,
        "reliability_index": reliability_index(p_success),
        "rare_event_failed": rare_event_probability(results, event="failed"),
        "constraint_violations": probability_of_violation(results),
        "failure_modes": failure_mode_ranking(results),
        "form": form,
        "sorm": sorm,
        "margin_metric": margin_metric,
        "margin_mean": float(np.nanmean(margins)) if len(margins) else math.nan,
        "margin_p05": float(np.nanpercentile(margins, 5)) if len(margins) else math.nan,
    }
    write_json(study / "reliability_summary.json", reliability)
    write_csv(study / "constraint_violations.csv", reliability["constraint_violations"])
    write_csv(study / "failure_mode_ranking.csv", reliability["failure_modes"])
    write_csv(study / "margin_decomposition.csv", margin_decomposition(samples, results, margin_metric))
    write_csv(study / "failure_mode_clusters.csv", failure_mode_clusters(samples, results))
    if len(margins):
        (study / "robustness_margin_histogram.svg").write_text(svg_hist(margins, "Robustness Margin Distribution", "#0f766e"))
    return reliability


def _uncertainty_bounds(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for name, spec in config.get("uncertainties", {}).items():
        kind = str(spec.get("type", "constant"))
        if kind in {"uniform", "normal", "truncated_normal", "beta", "bounded_beta"} and "low" in spec and "high" in spec:
            bounds[name] = (float(spec["low"]), float(spec["high"]))
        elif kind == "triangular":
            bounds[name] = (float(spec["low"]), float(spec["high"]))
    return bounds


def generate_stress_cases(config: dict[str, Any], *, max_cases: int = 64) -> list[dict[str, Any]]:
    resolved = resolve_config(config, config.get("__config_dir"))
    bounds = _uncertainty_bounds(resolved)
    names = list(bounds.keys())
    cases: list[dict[str, Any]] = []
    for combo in itertools.product([0, 1], repeat=len(names)):
        case = dict(resolved.get("nominal", {}))
        for name, high_flag in zip(names, combo):
            low, high = bounds[name]
            case[name] = high if high_flag else low
        cases.append(case)
        if len(cases) >= max_cases:
            break
    return cases


def run_stress_test(config_path: str | Path, out_dir: str | Path, *, max_cases: int = 64) -> dict[str, Any]:
    config = load_config(config_path)
    validate_or_raise(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = generate_stress_cases(config, max_cases=max_cases)
    rows = []
    for index, params in enumerate(cases):
        params = dict(params)
        params["run_seed"] = run_seed(int(config.get("seed", 2026)), index)
        result = simulate_model(params, config)
        rows.append({"case": index, **{k: params[k] for k in params if k in _uncertainty_bounds(config)}, **result})
    write_csv(out / "stress_results.csv", rows)
    write_json(
        out / "stress_summary.json",
        {
            "cases": len(rows),
            "failures": sum(1 for row in rows if not _as_bool(row.get("success"))),
            "worst_case": min(rows, key=lambda row: _as_float(row.get("robustness_margin"), 0.0)) if rows else {},
        },
    )
    if rows:
        margin = _series(rows, "robustness_margin")
        miss = _series(rows, "miss_distance_m")
        (out / "stress_margin_histogram.svg").write_text(svg_hist(margin, "Stress-Case Robustness Margins", "#b91c1c"))
        (out / "stress_miss_vs_margin.svg").write_text(svg_scatter(margin, miss, "Stress Miss vs Margin", "robustness_margin", "miss_distance_m"))
    return {"cases": len(rows), "out": str(out)}


def write_margin_report(study_dir: str | Path) -> dict[str, Any]:
    study = Path(study_dir)
    results = read_csv(study / "results.csv")
    samples = read_csv(study / "samples.csv")
    rows = margin_decomposition(samples, results)
    write_csv(study / "margin_report.csv", rows)
    payload = {"rows": len(rows), "top_margin_driver": rows[0] if rows else {}}
    write_json(study / "margin_report.json", payload)
    return payload
