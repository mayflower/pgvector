#!/usr/bin/env python3
"""Smoke-test the in-backend TurboHybrid profile JSON contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "benchmarks" / "turbohybrid" / "suite.py"


def assert_number(row: dict, key: str) -> None:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        raise AssertionError(f"{row.get('query_shape')}: missing numeric {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="contrib_regression")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--shapes", default="dense_only,bm25_common")
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        command = [
            sys.executable,
            str(SUITE),
            "profile-inbackend",
            "--database",
            args.database,
            "--rows",
            str(args.rows),
            "--dimensions",
            str(args.dimensions),
            "--k",
            "10",
            "--final-k",
            "5",
            "--runs",
            str(args.runs),
            "--warmup-runs",
            str(args.warmup_runs),
            "--shapes",
            args.shapes,
            "--cli-runs",
            "1",
            "--output",
            handle.name,
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        summary = json.loads(Path(handle.name).read_text(encoding="utf-8"))

    if summary.get("suite") != "turbohybrid_profile_inbackend":
        raise AssertionError("unexpected suite name")
    if summary.get("timing_mode") != "in_backend":
        raise AssertionError("missing top-level in_backend timing mode")
    if summary.get("runs") != args.runs:
        raise AssertionError("run count was not preserved")

    rows = summary.get("results")
    if not isinstance(rows, list) or not rows:
        raise AssertionError("results must be a non-empty list")

    for row in rows:
        if row.get("timing_mode") != "in_backend":
            raise AssertionError(f"{row.get('query_shape')}: missing row timing mode")
        if row.get("sample_count") != args.runs:
            raise AssertionError(f"{row.get('query_shape')}: wrong sample count")
        for key in ("engine_avg_us", "engine_p50_us", "engine_p95_us", "engine_p99_us"):
            assert_number(row, key)
        if "cli_avg_ms" not in row:
            raise AssertionError(f"{row.get('query_shape')}: missing cli_avg_ms")
        if "psql_subprocess_ms" not in row:
            raise AssertionError(f"{row.get('query_shape')}: missing psql_subprocess_ms")

    print(json.dumps({"ok": True, "rows": len(rows), "runs": args.runs}, sort_keys=True))


if __name__ == "__main__":
    main()
