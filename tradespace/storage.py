from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteStudy:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            create table if not exists metadata (
                key text primary key,
                value text not null
            )
            """
        )
        self.conn.execute(
            """
            create table if not exists runs (
                run_index integer primary key,
                seed integer not null,
                inputs_json text not null,
                outputs_json text not null,
                status text not null,
                config_hash text,
                updated_utc text default current_timestamp
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SQLiteStudy":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def set_metadata(self, key: str, value: Any) -> None:
        self.conn.execute(
            "insert or replace into metadata(key, value) values(?, ?)",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def record_run(
        self,
        run_index: int,
        seed: int,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        *,
        status: str = "complete",
        config_hash: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            insert or replace into runs(run_index, seed, inputs_json, outputs_json, status, config_hash, updated_utc)
            values(?, ?, ?, ?, ?, ?, current_timestamp)
            """,
            (
                int(run_index),
                int(seed),
                json.dumps(inputs, sort_keys=True, default=str),
                json.dumps(outputs, sort_keys=True, default=str),
                status,
                config_hash,
            ),
        )
        self.conn.commit()

    def completed_runs(self, *, config_hash: str | None = None) -> dict[int, tuple[dict[str, Any], dict[str, Any], int]]:
        sql = "select run_index, seed, inputs_json, outputs_json from runs where status = 'complete'"
        params: tuple[Any, ...] = ()
        if config_hash is not None:
            sql += " and config_hash = ?"
            params = (config_hash,)
        rows: dict[int, tuple[dict[str, Any], dict[str, Any], int]] = {}
        for run_index, seed, inputs_json, outputs_json in self.conn.execute(sql, params):
            rows[int(run_index)] = (json.loads(inputs_json), json.loads(outputs_json), int(seed))
        return rows


def export_sqlite(db_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    db = Path(db_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    metadata = {row["key"]: json.loads(row["value"]) for row in conn.execute("select key, value from metadata order by key")}
    runs = []
    for row in conn.execute("select run_index, seed, inputs_json, outputs_json, status, config_hash from runs order by run_index"):
        inputs = json.loads(row["inputs_json"])
        outputs = json.loads(row["outputs_json"])
        runs.append(
            {
                "run": row["run_index"],
                "seed": row["seed"],
                "status": row["status"],
                "config_hash": row["config_hash"],
                **{f"input.{key}": value for key, value in inputs.items()},
                **{f"output.{key}": value for key, value in outputs.items()},
            }
        )
    conn.close()

    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "runs.json").write_text(json.dumps(runs, indent=2, default=str))
    if runs:
        keys = sorted({key for row in runs for key in row})
        with (out / "runs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(runs)
    else:
        (out / "runs.csv").write_text("")
    return {"runs": len(runs), "metadata_keys": sorted(metadata)}
