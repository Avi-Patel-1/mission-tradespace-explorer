from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import config_hash, load_config, starter_config, strip_internal


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        rows: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.update(_flatten(child, child_prefix))
        return rows
    return {prefix: value}


def diff_configs(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    left = strip_internal(load_config(left_path))
    right = strip_internal(load_config(right_path))
    lf = _flatten(left)
    rf = _flatten(right)
    added = sorted(path for path in rf if path not in lf)
    removed = sorted(path for path in lf if path not in rf)
    changed = [
        {"path": path, "left": lf[path], "right": rf[path]}
        for path in sorted(set(lf) & set(rf))
        if lf[path] != rf[path]
    ]
    return {"left": str(left_path), "right": str(right_path), "added": added, "removed": removed, "changed": changed}


def freeze_config(config_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    resolved = strip_internal(load_config(config_path))
    payload = {"config_hash": config_hash(resolved), "resolved_config": resolved}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return {"written": str(out_path), "config_hash": payload["config_hash"]}


def generate_template(out_path: str | Path, *, kind: str = "monte_carlo", name: str = "template", model: str = "point_mass") -> dict[str, Any]:
    config = starter_config("sweep" if kind == "sweep" else "monte_carlo", name)
    config.setdefault("mission", {})
    config["mission"]["model"] = model
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(config, indent=2) + "\n")
    return {"written": str(out_path), "kind": kind, "model": model}
