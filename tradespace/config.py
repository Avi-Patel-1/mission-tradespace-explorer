from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


class ConfigError(ValueError):
    """Raised when a study configuration is invalid."""


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 2026,
    "runs": 1000,
    "study": {
        "name": "baseline",
        "kind": "monte_carlo",
    },
    "sampler": {
        "method": "monte_carlo",
        "chunk_size": 100,
    },
    "mission": {
        "dt_s": 0.05,
        "max_time_s": 70.0,
        "target_x_m": 3000.0,
        "target_y_m": 0.0,
        "target_z_m": 0.0,
        "initial_x_m": 0.0,
        "initial_y_m": 0.0,
        "initial_z_m": 110.0,
        "initial_speed_mps": 145.0,
        "initial_flight_path_deg": 8.0,
        "initial_heading_deg": 0.0,
        "success_miss_m": 230.0,
        "ground_altitude_m": 0.0,
    },
    "vehicle": {
        "mass_kg": 42.0,
        "thrust_n": 1560.0,
        "burn_time_s": 8.0,
        "reference_area_m2": 0.032,
        "drag_cd": 0.22,
        "max_qbar_pa": 85000.0,
        "max_load_g": 18.0,
        "propellant_mass_kg": 6.5,
    },
    "environment": {
        "density_kg_m3": 1.1,
        "density_scale_height_m": 8500.0,
        "wind_x_mps": 0.0,
        "wind_y_mps": 0.0,
        "wind_z_mps": 0.0,
    },
    "guidance": {
        "guidance_gain": 2.25,
        "loft_angle_deg": 0.0,
        "actuator_time_constant_s": 0.12,
        "guidance_gain_qbar_slope": 0.0,
        "guidance_gain_speed_slope": 0.0,
    },
    "sensors": {
        "sensor_noise_m": 0.0,
        "sensor_bias_x_m": 0.0,
        "sensor_bias_y_m": 0.0,
        "sensor_bias_z_m": 0.0,
    },
    "uncertainty": {
        "distributions": {},
    },
    "outputs": {
        "write_sqlite": True,
        "write_html": True,
        "write_manifest": True,
    },
}


PARAMETER_BLOCKS = ("mission", "vehicle", "environment", "guidance", "sensors")

SUPPORTED_DISTRIBUTIONS = {
    "constant",
    "uniform",
    "normal",
    "triangular",
    "lognormal",
    "truncated_normal",
    "beta",
    "bounded_beta",
    "choice",
    "discrete",
    "bernoulli",
    "correlated_normal",
    "mixture",
    "empirical",
    "table",
    "sweep",
    "sequence",
}

SUPPORTED_SAMPLERS = {
    "monte_carlo",
    "crude",
    "crude_monte_carlo",
    "random",
    "latin",
    "latin_hypercube",
    "lhs",
    "stratified",
    "sobol",
    "sobol_like",
    "low_discrepancy",
    "halton",
    "latinized_sobol",
    "latin_sobol",
    "orthogonal",
    "orthogonal_array",
    "oa",
}

SUPPORTED_SWEEP_MODES = {"grid", "latin", "random", "sobol", "nested", "evolutionary"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_raw(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = seen or set()
    path = path.expanduser().resolve()
    if path in seen:
        raise ConfigError(f"{path}: config inheritance cycle")
    seen.add(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc

    parents = raw.get("extends") or raw.get("inherits")
    if not parents:
        raw["__config_dir"] = str(path.parent)
        return raw

    if isinstance(parents, (str, Path)):
        parent_list = [parents]
    else:
        parent_list = list(parents)

    merged: dict[str, Any] = {}
    for parent in parent_list:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = _deep_merge(merged, _load_raw(parent_path, seen))

    child = {k: v for k, v in raw.items() if k not in {"extends", "inherits"}}
    child["__config_dir"] = str(path.parent)
    return _deep_merge(merged, child)


def _flatten_parameter_blocks(config: dict[str, Any]) -> dict[str, Any]:
    nominal: dict[str, Any] = {}
    for block in PARAMETER_BLOCKS:
        values = config.get(block, {})
        if isinstance(values, dict):
            nominal.update(values)
    legacy_nominal = config.get("nominal", {})
    if isinstance(legacy_nominal, dict):
        nominal.update(legacy_nominal)
    return nominal


def resolve_config(raw: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Any]:
    config = _deep_merge(DEFAULT_CONFIG, raw)
    if base_dir is not None:
        config["__config_dir"] = str(Path(base_dir).expanduser().resolve())
    elif "__config_dir" not in config:
        config["__config_dir"] = str(Path.cwd())

    nominal = _flatten_parameter_blocks(config)
    config["nominal"] = nominal

    distributions: dict[str, Any] = {}
    uncertainty_block = config.get("uncertainty", {})
    if isinstance(uncertainty_block, dict):
        block_distributions = uncertainty_block.get("distributions", {})
        if isinstance(block_distributions, dict):
            distributions.update(block_distributions)
    legacy_uncertainties = config.get("uncertainties", {})
    if isinstance(legacy_uncertainties, dict):
        distributions.update(legacy_uncertainties)
    config["uncertainties"] = distributions
    config.setdefault("sweep", {})
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    raw = _load_raw(source)
    return resolve_config(raw, source.parent)


def strip_internal(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        key: value
        for key, value in config.items()
        if not str(key).startswith("__")
    }
    return copy.deepcopy(cleaned)


def config_hash(config: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(strip_internal(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_overrides(
    config: dict[str, Any],
    *,
    seed: int | None = None,
    sampler: str | None = None,
    write_sqlite: bool | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if seed is not None:
        updated["seed"] = int(seed)
    if sampler is not None:
        updated.setdefault("sampler", {})
        updated["sampler"]["method"] = sampler
    if write_sqlite is not None:
        updated.setdefault("outputs", {})
        updated["outputs"]["write_sqlite"] = bool(write_sqlite)
    return resolve_config(updated, updated.get("__config_dir"))


def starter_config(kind: str = "monte_carlo", name: str = "new_study") -> dict[str, Any]:
    base = {
        "seed": 2026,
        "runs": 250,
        "study": {"name": name, "kind": kind},
        "sampler": {"method": "latin_hypercube", "chunk_size": 100},
        "mission": {
            "target_x_m": 3000.0,
            "target_z_m": 0.0,
            "initial_z_m": 110.0,
            "initial_speed_mps": 145.0,
            "initial_flight_path_deg": 25.0,
            "success_miss_m": 230.0,
        },
        "vehicle": {
            "mass_kg": 42.0,
            "thrust_n": 2200.0,
            "burn_time_s": 12.0,
            "reference_area_m2": 0.032,
            "drag_cd": 0.22,
            "max_qbar_pa": 200000.0,
            "max_load_g": 18.0,
        },
        "environment": {
            "density_kg_m3": 1.1,
            "wind_x_mps": 0.0,
            "wind_y_mps": 0.0,
            "wind_z_mps": 0.0,
        },
        "guidance": {"guidance_gain": 2.25, "actuator_time_constant_s": 0.12},
        "sensors": {"sensor_noise_m": 8.0, "sensor_bias_x_m": 0.0, "sensor_bias_z_m": 0.0},
        "uncertainty": {
            "distributions": {
                "wind_x_mps": {"type": "normal", "mean": 0.0, "std": 6.0, "low": -18.0, "high": 18.0},
                "mass_kg": {"type": "triangular", "low": 39.5, "mode": 42.0, "high": 45.0},
                "thrust_n": {"type": "normal", "mean": 2200.0, "std": 120.0, "low": 1850.0, "high": 2550.0},
                "sensor_noise_m": {"type": "uniform", "low": 0.0, "high": 18.0},
            }
        },
    }
    if kind in {"sweep", "trade", "trade_space"}:
        base["study"]["kind"] = "sweep"
        base["sweep"] = {
            "mode": "grid",
            "uncertainty_runs": 4,
            "values": {
                "guidance_gain": [1.8, 2.2, 2.6],
                "thrust_n": [2000.0, 2200.0, 2400.0],
            },
            "constraints": {
                "require_success": False,
                "miss_distance_m": {"max": 260.0},
                "max_qbar_pa": {"max": 200000.0},
            },
        }
    return base


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_number(errors: list[str], path: str, value: Any, *, positive: bool = False, nonnegative: bool = False) -> None:
    if not _is_number(value):
        errors.append(f"{path}: expected a number")
        return
    if positive and float(value) <= 0.0:
        errors.append(f"{path}: expected a positive number")
    if nonnegative and float(value) < 0.0:
        errors.append(f"{path}: expected a nonnegative number")


def _check_bounds(errors: list[str], path: str, spec: dict[str, Any]) -> None:
    if "low" in spec and "high" in spec:
        _check_number(errors, f"{path}.low", spec["low"])
        _check_number(errors, f"{path}.high", spec["high"])
        if _is_number(spec.get("low")) and _is_number(spec.get("high")) and float(spec["low"]) > float(spec["high"]):
            errors.append(f"{path}: low must be <= high")


def _require_keys(errors: list[str], path: str, spec: dict[str, Any], keys: tuple[str, ...]) -> bool:
    ok = True
    for key in keys:
        if key not in spec:
            errors.append(f"{path}.{key}: required")
            ok = False
    return ok


def _check_range_pair(errors: list[str], path: str, value: Any) -> None:
    if not isinstance(value, list) or len(value) != 2:
        errors.append(f"{path}: expected [low, high]")
        return
    _check_number(errors, f"{path}[0]", value[0])
    _check_number(errors, f"{path}[1]", value[1])
    if _is_number(value[0]) and _is_number(value[1]) and float(value[0]) > float(value[1]):
        errors.append(f"{path}: low must be <= high")


def _check_value_list(errors: list[str], path: str, value: Any) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{path}: expected a nonempty list")


def _validate_distribution(errors: list[str], path: str, spec: Any, root: str | Path | None = None) -> None:
    if not isinstance(spec, dict):
        errors.append(f"{path}: expected distribution object")
        return

    kind = str(spec.get("type", "constant"))
    if kind not in SUPPORTED_DISTRIBUTIONS:
        errors.append(f"{path}.type: unsupported distribution type '{kind}'")
        return

    if kind == "constant":
        if "value" not in spec:
            errors.append(f"{path}.value: required for constant distribution")
    elif kind == "uniform":
        if _require_keys(errors, path, spec, ("low", "high")):
            _check_bounds(errors, path, spec)
    elif kind == "normal":
        for key in ("mean", "std"):
            _check_number(errors, f"{path}.{key}", spec.get(key), positive=(key == "std"))
        _check_bounds(errors, path, spec)
    elif kind == "triangular":
        for key in ("low", "mode", "high"):
            _check_number(errors, f"{path}.{key}", spec.get(key))
        if all(_is_number(spec.get(k)) for k in ("low", "mode", "high")):
            if not (float(spec["low"]) <= float(spec["mode"]) <= float(spec["high"])):
                errors.append(f"{path}: expected low <= mode <= high")
    elif kind == "lognormal":
        for key in ("mean", "sigma"):
            _check_number(errors, f"{path}.{key}", spec.get(key), positive=(key == "sigma"))
    elif kind == "truncated_normal":
        for key in ("mean", "std", "low", "high"):
            _check_number(errors, f"{path}.{key}", spec.get(key), positive=(key == "std"))
        _check_bounds(errors, path, spec)
    elif kind in {"beta", "bounded_beta"}:
        for key in ("alpha", "beta", "low", "high"):
            _check_number(errors, f"{path}.{key}", spec.get(key), positive=key in {"alpha", "beta"})
        _check_bounds(errors, path, spec)
    elif kind in {"choice", "discrete"}:
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            errors.append(f"{path}.values: expected a nonempty list")
        probabilities = spec.get("probabilities")
        if probabilities is not None:
            if not isinstance(probabilities, list) or len(probabilities) != len(values or []):
                errors.append(f"{path}.probabilities: expected one probability per value")
            elif any((not _is_number(p) or float(p) < 0.0) for p in probabilities):
                errors.append(f"{path}.probabilities: expected nonnegative numeric probabilities")
            elif sum(float(p) for p in probabilities) <= 0.0:
                errors.append(f"{path}.probabilities: expected at least one positive probability")
    elif kind == "bernoulli":
        _check_number(errors, f"{path}.p", spec.get("p"), nonnegative=True)
        if _is_number(spec.get("p")) and not (0.0 <= float(spec["p"]) <= 1.0):
            errors.append(f"{path}.p: expected 0 <= p <= 1")
    elif kind == "correlated_normal":
        names = spec.get("names")
        mean = spec.get("mean")
        cov = spec.get("cov")
        if not isinstance(names, list) or not names:
            errors.append(f"{path}.names: expected a nonempty list")
        if not isinstance(mean, list) or len(mean) != len(names or []):
            errors.append(f"{path}.mean: expected one mean per name")
        if not isinstance(cov, list) or len(cov) != len(names or []):
            errors.append(f"{path}.cov: expected square covariance matrix")
        elif any(not isinstance(row, list) or len(row) != len(names or []) for row in cov):
            errors.append(f"{path}.cov: expected square covariance matrix")
        else:
            try:
                cov_array = np.array(cov, dtype=float)
                if not np.allclose(cov_array, cov_array.T, atol=1e-9):
                    errors.append(f"{path}.cov: expected a symmetric matrix")
                elif float(np.min(np.linalg.eigvalsh(cov_array))) < -1e-9:
                    errors.append(f"{path}.cov: expected positive semidefinite covariance")
            except (TypeError, ValueError, np.linalg.LinAlgError):
                errors.append(f"{path}.cov: expected numeric covariance matrix")
    elif kind == "mixture":
        components = spec.get("components")
        if not isinstance(components, list) or not components:
            errors.append(f"{path}.components: expected a nonempty list")
        else:
            for i, component in enumerate(components):
                component_spec = component.get("distribution", component) if isinstance(component, dict) else component
                if isinstance(component, dict) and "weight" in component:
                    _check_number(errors, f"{path}.components[{i}].weight", component["weight"], nonnegative=True)
                _validate_distribution(errors, f"{path}.components[{i}]", component_spec, root)
    elif kind in {"empirical", "table"}:
        if not spec.get("path"):
            errors.append(f"{path}.path: required for table distribution")
        if not spec.get("column"):
            errors.append(f"{path}.column: required for table distribution")
        if spec.get("path") and spec.get("column"):
            table_path = Path(str(spec["path"])).expanduser()
            if not table_path.is_absolute() and root is not None:
                table_path = Path(root) / table_path
            if not table_path.exists():
                errors.append(f"{path}.path: table file does not exist: {table_path}")
            else:
                try:
                    with table_path.open(newline="") as handle:
                        reader = csv.DictReader(handle)
                        columns = set(reader.fieldnames or [])
                    if spec["column"] not in columns:
                        errors.append(f"{path}.column: '{spec['column']}' not found in {table_path}")
                    if spec.get("weight_column") and spec["weight_column"] not in columns:
                        errors.append(f"{path}.weight_column: '{spec['weight_column']}' not found in {table_path}")
                except OSError as exc:
                    errors.append(f"{path}.path: could not read table file: {exc}")
    elif kind in {"sweep", "sequence"}:
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            errors.append(f"{path}.values: expected a nonempty list")


def validate_config(config: dict[str, Any]) -> list[str]:
    resolved = resolve_config(config, config.get("__config_dir"))
    errors: list[str] = []

    if not isinstance(resolved.get("seed"), int):
        errors.append("seed: expected an integer")
    if not isinstance(resolved.get("runs"), int) or int(resolved.get("runs", 0)) <= 0:
        errors.append("runs: expected a positive integer")
    mission = resolved.get("mission", {})
    if isinstance(mission, dict):
        model = str(mission.get("model", resolved.get("model", "point_mass")))
        try:
            from .mission_models import available_models

            if model not in available_models():
                errors.append(f"mission.model: unsupported model '{model}'")
        except ImportError:
            pass

    sampler = resolved.get("sampler", {})
    if not isinstance(sampler, dict):
        errors.append("sampler: expected object")
    else:
        method = str(sampler.get("method", "monte_carlo"))
        if method not in SUPPORTED_SAMPLERS:
            errors.append(f"sampler.method: unsupported sampler '{method}'")
        if "chunk_size" in sampler and (not isinstance(sampler["chunk_size"], int) or sampler["chunk_size"] <= 0):
            errors.append("sampler.chunk_size: expected a positive integer")

    for key in ("dt_s", "max_time_s", "target_x_m", "initial_speed_mps", "mass_kg", "thrust_n", "burn_time_s"):
        if key in resolved.get("nominal", {}):
            _check_number(errors, f"nominal.{key}", resolved["nominal"][key], positive=key in {"dt_s", "max_time_s", "initial_speed_mps", "mass_kg"})

    for name, spec in resolved.get("uncertainties", {}).items():
        _validate_distribution(errors, f"uncertainties.{name}", spec, resolved.get("__config_dir"))

    sweep = resolved.get("sweep", {})
    if sweep:
        if not isinstance(sweep, dict):
            errors.append("sweep: expected object")
        else:
            mode = sweep.get("mode", "grid")
            if mode not in SUPPORTED_SWEEP_MODES:
                errors.append("sweep.mode: expected grid, latin, random, sobol, nested, or evolutionary")
            if mode in {"grid", "nested"}:
                if not isinstance(sweep.get("values"), dict) or not sweep.get("values"):
                    errors.append("sweep.values: expected nonempty object mapping parameter names to value lists")
                else:
                    for name, values in sweep["values"].items():
                        _check_value_list(errors, f"sweep.values.{name}", values)
            if mode in {"latin", "random", "sobol", "evolutionary"}:
                if not isinstance(sweep.get("bounds"), dict) or not sweep.get("bounds"):
                    errors.append("sweep.bounds: expected nonempty object mapping parameter names to [low, high]")
                else:
                    for name, bounds in sweep["bounds"].items():
                        _check_range_pair(errors, f"sweep.bounds.{name}", bounds)
            for count_key in ("samples", "population", "generations", "uncertainty_runs", "replicates"):
                if count_key in sweep and (not isinstance(sweep[count_key], int) or sweep[count_key] <= 0):
                    errors.append(f"sweep.{count_key}: expected a positive integer")
            for i, objective in enumerate(sweep.get("objectives", [])):
                if not isinstance(objective, dict):
                    errors.append(f"sweep.objectives[{i}]: expected object")
                    continue
                if not objective.get("metric"):
                    errors.append(f"sweep.objectives[{i}].metric: required")
                if objective.get("sense", "min") not in {"min", "max"}:
                    errors.append(f"sweep.objectives[{i}].sense: expected min or max")

    return errors


def validate_or_raise(config: dict[str, Any]) -> None:
    errors = validate_config(config)
    if errors:
        raise ConfigError("\n".join(errors))
