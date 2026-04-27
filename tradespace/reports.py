from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PREFERRED_COLUMNS = [
    "run",
    "case",
    "replicate",
    "success",
    "failure_reason",
    "miss_distance_m",
    "robustness_margin",
    "max_qbar_pa",
    "max_load_g",
    "time_of_flight_s",
    "impulse_n_s",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    all_keys = sorted({key for row in rows for key in row.keys()})
    keys = [key for key in PREFERRED_COLUMNS if key in all_keys] + [key for key in all_keys if key not in PREFERRED_COLUMNS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _parse_cell(value: str) -> Any:
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    try:
        if value.strip() == "":
            return ""
        return float(value)
    except ValueError:
        return value


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _parse_cell(value) for key, value in row.items()} for row in reader]


def _numeric_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return np.array(values, dtype=float)


def _stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def failure_reason_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("failure_reason", "none")) for row in results)
    total = max(len(results), 1)
    return [
        {
            "failure_reason": reason,
            "count": count,
            "fraction": count / total,
        }
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def percentile_rows(results: list[dict[str, Any]], metrics: list[str] | None = None) -> list[dict[str, Any]]:
    metrics = metrics or ["miss_distance_m", "max_qbar_pa", "max_load_g", "terminal_speed_mps", "time_of_flight_s", "robustness_margin"]
    rows = []
    for metric in metrics:
        stats = _stats(_numeric_array(results, metric))
        if stats:
            rows.append({"metric": metric, **stats})
    return rows


def summarize(results: list[dict[str, Any]], sensitivity: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    sensitivity = sensitivity or []
    success = np.array([1.0 if row.get("success") in {True, "True", "true", 1, "1"} else 0.0 for row in results], dtype=float)
    return {
        "runs": len(results),
        "success_rate": float(np.mean(success)) if len(success) else 0.0,
        "metrics": {row["metric"]: {k: v for k, v in row.items() if k != "metric"} for row in percentile_rows(results)},
        "miss_distance_m": _stats(_numeric_array(results, "miss_distance_m")),
        "max_qbar_pa": _stats(_numeric_array(results, "max_qbar_pa")),
        "failure_reasons": failure_reason_rows(results),
        "top_sensitivities": sensitivity[:10],
    }


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if abs(src_max - src_min) <= 1e-12:
        return (dst_min + dst_max) / 2.0
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def svg_hist(values: np.ndarray, title: str, fill: str = "#2563eb") -> str:
    values = values[np.isfinite(values)]
    width, height, pad = 760, 280, 44
    if len(values) == 0:
        values = np.array([0.0])
    counts, edges = np.histogram(values, bins=min(24, max(6, int(np.sqrt(len(values))))))
    max_count = max(int(counts.max()), 1)
    bars = []
    for i, count in enumerate(counts):
        x = pad + i * (width - 2 * pad) / len(counts)
        bar_w = (width - 2 * pad) / len(counts) - 2
        bar_h = (height - 2 * pad) * float(count) / max_count
        y = height - pad - bar_h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{fill}"/>')
    return "\n".join(
        [
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{pad}" y="28" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
            *bars,
            f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#1f2937"/>',
            f'<text x="{pad}" y="{height-12}" font-family="Arial" font-size="12">{edges[0]:.2f}</text>',
            f'<text x="{width-pad-92}" y="{height-12}" font-family="Arial" font-size="12">{edges[-1]:.2f}</text>',
            '</svg>',
        ]
    )


def svg_scatter(x: np.ndarray, y: np.ndarray, title: str, x_label: str, y_label: str) -> str:
    width, height, pad = 760, 320, 52
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        x, y = np.array([0.0]), np.array([0.0])
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    points = []
    for xi, yi in zip(x, y):
        sx = _scale(float(xi), x_min, x_max, pad, width - pad)
        sy = _scale(float(yi), y_min, y_max, height - pad, pad)
        points.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.6" fill="#0f766e" fill-opacity="0.72"/>')
    return "\n".join(
        [
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{pad}" y="28" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
            f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#1f2937"/>',
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#1f2937"/>',
            *points,
            f'<text x="{width/2-60:.1f}" y="{height-10}" font-family="Arial" font-size="12">{html.escape(x_label)}</text>',
            f'<text x="8" y="{pad-14}" font-family="Arial" font-size="12">{html.escape(y_label)}</text>',
            '</svg>',
        ]
    )


def svg_tornado(sensitivity: list[dict[str, Any]], title: str = "Sensitivity Tornado") -> str:
    width, height, pad = 760, 320, 56
    rows = sensitivity[:10]
    max_abs = max([abs(float(row.get("tornado_delta", 0.0))) for row in rows] + [1.0])
    center = width * 0.52
    row_h = 22
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad}" y="28" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{center:.1f}" y1="{pad-8}" x2="{center:.1f}" y2="{height-pad+10}" stroke="#6b7280"/>',
    ]
    for i, row in enumerate(rows):
        delta = float(row.get("tornado_delta", 0.0))
        length = abs(delta) / max_abs * (width * 0.32)
        y = pad + i * row_h
        x = center if delta >= 0 else center - length
        color = "#b91c1c" if delta >= 0 else "#1d4ed8"
        parts.append(f'<text x="{pad}" y="{y+12}" font-family="Arial" font-size="12">{html.escape(str(row.get("parameter", "")))}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{length:.1f}" height="14" fill="{color}" fill-opacity="0.82"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_report(
    out: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    sensitivity: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]] | None = None,
    pareto_rows: list[dict[str, Any]] | None = None,
    write_html: bool = True,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "failure_reasons.csv", failure_reason_rows(results))
    write_csv(out / "percentiles.csv", percentile_rows(results))

    miss = _numeric_array(results, "miss_distance_m")
    qbar = _numeric_array(results, "max_qbar_pa")
    load = _numeric_array(results, "max_load_g")
    impulse = _numeric_array(results, "impulse_n_s")
    (out / "miss_distance_histogram.svg").write_text(svg_hist(miss, "Miss Distance Distribution", "#2563eb"))
    (out / "qbar_histogram.svg").write_text(svg_hist(qbar, "Max Dynamic Pressure Distribution", "#7c3aed"))
    (out / "load_histogram.svg").write_text(svg_hist(load, "Load Proxy Distribution", "#b45309"))
    (out / "sensitivity_tornado.svg").write_text(svg_tornado(sensitivity))
    if len(miss) and len(qbar):
        (out / "miss_vs_qbar_scatter.svg").write_text(svg_scatter(qbar, miss, "Miss Distance vs Dynamic Pressure", "max_qbar_pa", "miss_distance_m"))
    if len(miss) and len(impulse):
        (out / "miss_vs_impulse_scatter.svg").write_text(svg_scatter(impulse, miss, "Miss Distance vs Impulse", "impulse_n_s", "miss_distance_m"))
    if pareto_rows:
        write_csv(out / "pareto.csv", pareto_rows)

    sensitivity_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('parameter', '')))}</td>"
        f"<td>{float(row.get('rank_correlation', 0.0)):+.3f}</td>"
        f"<td>{float(row.get('partial_rank_correlation', 0.0)):+.3f}</td>"
        f"<td>{float(row.get('success_correlation', 0.0)):+.3f}</td>"
        f"<td>{float(row.get('variance_contribution_proxy', 0.0)):.3f}</td>"
        "</tr>"
        for row in sensitivity[:12]
    )
    failure_rows = "".join(
        f"<tr><td>{html.escape(str(row['failure_reason']))}</td><td>{int(row['count'])}</td><td>{float(row['fraction']):.1%}</td></tr>"
        for row in summary.get("failure_reasons", [])
    )
    if not write_html:
        return

    report = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Mission Performance Report</title>
<style>
body{{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#17202a;max-width:1120px}}
table{{border-collapse:collapse;margin:12px 0 24px 0}}td,th{{border-bottom:1px solid #ddd;padding:7px 12px;text-align:left}}
.kpi{{display:inline-block;margin:0 24px 16px 0}}.kpi strong{{display:block;font-size:24px}}pre{{background:#f7f7f7;padding:12px;overflow:auto}}
svg{{max-width:100%;height:auto;border:1px solid #e5e7eb;margin:10px 0}}
</style>
</head>
<body>
<h1>Mission Performance Report</h1>
<div class="kpi"><span>Runs</span><strong>{summary.get('runs', 0)}</strong></div>
<div class="kpi"><span>Success rate</span><strong>{float(summary.get('success_rate', 0.0)):.1%}</strong></div>
<div class="kpi"><span>Median miss</span><strong>{summary.get('miss_distance_m', {}).get('p50', 0.0):.2f} m</strong></div>
<div class="kpi"><span>95th pct qbar</span><strong>{summary.get('max_qbar_pa', {}).get('p95', 0.0):.0f} Pa</strong></div>
{svg_hist(miss, 'Miss Distance Distribution', '#2563eb')}
{svg_hist(qbar, 'Max Dynamic Pressure Distribution', '#7c3aed')}
{svg_tornado(sensitivity)}
<h2>Failure Reasons</h2>
<table><tr><th>Reason</th><th>Count</th><th>Fraction</th></tr>{failure_rows}</table>
<h2>Top Sensitivities</h2>
<table><tr><th>Parameter</th><th>Rank</th><th>Partial rank</th><th>Success</th><th>Variance proxy</th></tr>{sensitivity_rows}</table>
<h2>Summary JSON</h2><pre>{html.escape(json.dumps(summary, indent=2, default=_json_default))}</pre>
</body>
</html>"""
    (out / "report.html").write_text(report)


def rewrite_report_from_study(study_dir: str | Path) -> dict[str, Any]:
    study = Path(study_dir)
    results = read_csv(study / "results.csv")
    samples = read_csv(study / "samples.csv")
    sensitivity = read_csv(study / "sensitivity.csv")
    existing_summary: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    if (study / "summary.json").exists():
        existing_summary = json.loads((study / "summary.json").read_text())
    if (study / "manifest.json").exists():
        manifest = json.loads((study / "manifest.json").read_text())
    summary = summarize(results, sensitivity)
    for key in ("config_hash", "sampler", "resumed_runs", "cases", "design_rows", "pareto_rows", "best_case"):
        if key in existing_summary:
            summary[key] = existing_summary[key]
    for key in ("config_hash", "seed", "sampler"):
        if key in manifest and key not in summary:
            summary[key] = manifest[key]
    write_json(study / "summary.json", summary)
    write_report(study, results, summary, sensitivity, samples=samples)
    return summary
