#!/usr/bin/env python3
"""TurboHybrid benchmark suite runner.

The suite intentionally separates three questions:

* IR quality: score TREC-format runs for BEIR/MTEB, MS MARCO/TREC-DL,
  MIRACL, LoTTE, BRIGHT, and small RAG datasets.
* Systems performance: run a reproducible PostgreSQL synthetic benchmark that
  captures hot/cold latency, p95/p99, QPS, build time, index size, WAL,
  insert/delete/vacuum, and filter-selectivity behavior.
* Reference comparisons: keep method metadata for SQL-level RRF, Lucene,
  ParadeDB, and external vector databases so their outputs can be normalized
  into the same result schema.

The runner has no Python package dependencies. External tools are optional and
are invoked outside this script; their result files should be imported as TREC
run files or JSON summaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"

DEFAULT_ACCEPTANCE_THRESHOLDS = {
    "hybrid_rrf_common_vs_dense_p95_ratio": 2.5,
    "bm25_only_rare_p95_ms": 20.0,
    "fusion_elapsed_pct_final_k_50": 20.0,
    "delta_10_vs_delta_0_p95_ratio": 2.0,
    "post_compaction_vs_delta_0_p95_ratio": 1.2,
}

ACCEPTANCE_SHAPES = [
    "dense_only",
    "bm25_only_rare",
    "bm25_only_common",
    "hybrid_rrf_rare",
    "hybrid_rrf_common",
    "hybrid_weighted",
    "hybrid_with_filter_1_percent",
    "hybrid_with_filter_10_percent",
    "delta_0_percent",
    "delta_1_percent",
    "delta_5_percent",
    "delta_10_percent",
    "post_compaction",
]

ACCEPTANCE_PROFILES = {
    "smoke": {
        "rows": [1000],
        "dimensions": 16,
        "methods": "turbohybrid",
        "warmup": 0,
        "runs": 1,
        "concurrency": "1",
        "validation": "fast",
        "shapes": [
            "dense_only",
            "bm25_only_rare",
            "hybrid_rrf_common",
            "delta_1_percent",
            "post_compaction",
        ],
    },
    "dev": {
        "rows": [10000],
        "dimensions": 16,
        "methods": "turbohybrid,turboquant_dense,postgres_sql_rrf",
        "warmup": 0,
        "runs": 1,
        "concurrency": "1",
        "validation": "fast",
        "shapes": ACCEPTANCE_SHAPES,
    },
    "full": {
        "rows": [10000, 100000],
        "dimensions": 64,
        "methods": "turbohybrid,turboquant_dense,postgres_sql_rrf",
        "warmup": 3,
        "runs": 30,
        "concurrency": "1,4,16",
        "validation": "standard",
        "shapes": ACCEPTANCE_SHAPES,
    },
}

ACCEPTANCE_VALIDATION_LEVELS = {"none", "fast", "standard", "full"}


@dataclass(frozen=True)
class Method:
    name: str
    layer: str
    kind: str
    description: str


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_acceptance_thresholds(path: str) -> dict[str, float]:
    thresholds = dict(DEFAULT_ACCEPTANCE_THRESHOLDS)
    if path:
        loaded = load_json(Path(path))
    else:
        default_path = CONFIG_DIR / "acceptance_thresholds.json"
        loaded = load_json(default_path) if default_path.exists() else {}
    for key, value in loaded.items():
        if key not in thresholds:
            raise ValueError(f"unknown acceptance threshold: {key}")
        thresholds[key] = float(value)
    return thresholds


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def acceptance_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile not in ACCEPTANCE_PROFILES:
        raise ValueError(f"unknown acceptance profile: {args.profile}")
    return ACCEPTANCE_PROFILES[args.profile]


def acceptance_rows(args: argparse.Namespace) -> list[int]:
    profile = acceptance_profile(args)
    if args.rows_list:
        rows_values = parse_int_list(args.rows_list)
    elif args.rows is not None:
        rows_values = [args.rows]
    else:
        rows_values = list(profile["rows"])

    if args.large:
        rows_values = [max(rows, 1_000_000) for rows in rows_values]
    return rows_values


def acceptance_shapes(args: argparse.Namespace) -> list[str]:
    profile = acceptance_profile(args)
    shapes = parse_str_list(args.shapes) if args.shapes else list(profile["shapes"])
    unknown = [shape for shape in shapes if shape not in ACCEPTANCE_SHAPES]
    if unknown:
        raise ValueError(f"unknown acceptance shape(s): {', '.join(unknown)}")
    return shapes


def acceptance_methods(args: argparse.Namespace) -> list[str]:
    profile = acceptance_profile(args)
    methods = args.methods if args.methods else profile["methods"]
    return [normalize_method(item) for item in parse_str_list(methods)]


def acceptance_concurrencies(args: argparse.Namespace) -> list[int]:
    profile = acceptance_profile(args)
    concurrency = args.concurrency if args.concurrency else profile["concurrency"]
    return parse_int_list(concurrency)


def acceptance_dimensions(args: argparse.Namespace) -> int:
    profile = acceptance_profile(args)
    return args.dimensions if args.dimensions is not None else int(profile["dimensions"])


def acceptance_runs(args: argparse.Namespace) -> int:
    profile = acceptance_profile(args)
    return args.runs if args.runs is not None else int(profile["runs"])


def acceptance_warmup(args: argparse.Namespace) -> int:
    profile = acceptance_profile(args)
    return args.warmup if args.warmup is not None else int(profile["warmup"])


def acceptance_validation(args: argparse.Namespace) -> str:
    profile = acceptance_profile(args)
    validation = args.validation or str(profile["validation"])
    if validation not in ACCEPTANCE_VALIDATION_LEVELS:
        raise ValueError(
            "unknown acceptance validation level: "
            f"{validation}; expected one of {', '.join(sorted(ACCEPTANCE_VALIDATION_LEVELS))}"
        )
    return validation


def acceptance_enforce_latency_thresholds(rows: int, runs: int, concurrencies: list[int], profile: str) -> bool:
    if profile == "full":
        return True
    return rows >= 100000 and runs >= 30 and {1, 4, 16}.issubset(set(concurrencies))


def run_psql(database: str, sql: str, *, quiet: bool = True) -> str:
    proc = subprocess.run(
        ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-d", database, "-At", "-c", sql],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if quiet else None,
    )
    return proc.stdout.strip()


def run_psql_file(database: str, sql: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as handle:
        handle.write(sql)
        path = handle.name
    try:
        subprocess.run(
            ["psql", "-X", "-v", "ON_ERROR_STOP=1", "-q", "-d", database, "-f", path],
            check=True,
            text=True,
        )
    finally:
        os.unlink(path)


def percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = math.ceil((pct / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarize_latency(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(percentile(samples, 95), 3),
        "p99_ms": round(percentile(samples, 99), 3),
        "mean_ms": round(statistics.mean(samples), 3),
    }


def timed_query(database: str, sql: str) -> float:
    start = time.perf_counter()
    run_psql(database, sql)
    return (time.perf_counter() - start) * 1000.0


def run_query_batch(database: str, sql: str, runs: int, concurrency: int) -> dict[str, Any]:
    start = time.perf_counter()
    if concurrency <= 1:
        samples = [timed_query(database, sql) for _ in range(runs)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            samples = list(pool.map(lambda _: timed_query(database, sql), range(runs)))
    elapsed = time.perf_counter() - start
    summary = summarize_latency(samples)
    summary["qps"] = round(runs / elapsed, 3) if elapsed > 0 else 0.0
    summary["runs"] = runs
    summary["concurrency"] = concurrency
    summary["sample_runs"] = [
        {"run_number": index + 1, "latency_ms": round(sample, 3)}
        for index, sample in enumerate(samples)
    ]
    return summary


def run_psql_text(database: str, sql: str) -> str:
    proc = subprocess.run(
        ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", sql],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def host_memory() -> str:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return "unavailable"
    if pages <= 0 or page_size <= 0:
        return "unavailable"
    return str(pages * page_size)


def cpu_model() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                return subprocess.check_output(
                    ["sysctl", "-n", "hw.model"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                return platform.processor() or "unknown"
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def compiler_info() -> str:
    try:
        cc = subprocess.check_output(["pg_config", "--cc"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        cc = os.environ.get("CC", "cc")
    try:
        first = subprocess.check_output(
            shlex.split(cc) + ["--version"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
        return first
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return cc


def postgres_version(database: str) -> str:
    return run_psql(database, "SHOW server_version")


def vector_literal(dimensions: int) -> str:
    values = [str(round(((i % 11) + 1) / 11.0, 6)) for i in range(dimensions)]
    return "'[" + ",".join(values) + "]'::vector"


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def synthetic_setup_sql(rows: int, dimensions: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS turbohybrid_suite_docs;
CREATE TABLE turbohybrid_suite_docs (
    id int PRIMARY KEY,
    bucket int NOT NULL,
    embedding vector({dimensions}) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector NOT NULL
);

INSERT INTO turbohybrid_suite_docs
SELECT
    i,
    i % 100,
    ('[' || array_to_string(ARRAY(
        SELECT (((i * (d + 17)) % 1000)::float8 / 1000.0)::text
        FROM generate_series(1, {dimensions}) d
    ), ',') || ']')::vector({dimensions}),
    concat_ws(' ',
        'common',
        'term' || (i % 97),
        'topic' || (i % 31)),
    to_tsvector('english', concat_ws(' ',
        'common',
        'term' || (i % 97),
        'topic' || (i % 31)))
FROM generate_series(1, {rows}) i;
ANALYZE turbohybrid_suite_docs;
"""


def drop_method_indexes_sql() -> str:
    return """
DROP INDEX IF EXISTS turbohybrid_suite_hybrid_idx;
DROP INDEX IF EXISTS turbohybrid_suite_turboquant_idx;
DROP INDEX IF EXISTS turbohybrid_suite_hnsw_idx;
DROP INDEX IF EXISTS turbohybrid_suite_fts_idx;
"""


def normalize_method(method: str) -> str:
    if method in {"sql_rrf", "pgvector", "normal_pgvector", "pgvector_sql_rrf"}:
        return "postgres_sql_rrf"
    if method in {
        "turbohybrid_exact_off",
        "turbohybrid_no_exact",
        "turbohybrid_exact_storage_off",
    }:
        return "turbohybrid_exact_storage_off"
    return method


def build_method_sql(method: str) -> tuple[str, list[str]]:
    method = normalize_method(method)
    if method == "turbohybrid":
        return (
            """
CREATE INDEX turbohybrid_suite_hybrid_idx ON turbohybrid_suite_docs
USING turbohybrid (
    embedding vector_cosine_hybrid_ops,
    body_tsv bm25_tsvector_ops
)
INCLUDE (bucket)
WITH (
    graph_m = 16,
    graph_ef_construction = 128,
    graph_ef_search = 64,
    graph_oversampling = 4,
    tq_bits = 4,
    tq_exact_storage = on,
    bm25_block_max = on
);
""",
            ["turbohybrid_suite_hybrid_idx"],
        )
    if method == "postgres_sql_rrf":
        return (
            """
CREATE INDEX turbohybrid_suite_hnsw_idx ON turbohybrid_suite_docs
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);
CREATE INDEX turbohybrid_suite_fts_idx ON turbohybrid_suite_docs
USING gin (body_tsv);
""",
            ["turbohybrid_suite_hnsw_idx", "turbohybrid_suite_fts_idx"],
        )
    if method == "turboquant_dense":
        return (
            """
CREATE INDEX turbohybrid_suite_turboquant_idx ON turbohybrid_suite_docs
USING turboquant (embedding vector_cosine_ops)
WITH (
    graph_m = 16,
    graph_ef_construction = 128,
    tq_bits = 4,
    tq_exact_storage = on
);
""",
            ["turbohybrid_suite_turboquant_idx"],
        )
    raise ValueError(f"unsupported synthetic method: {method}")


def query_sql(method: str, dimensions: int, filter_clause: str) -> str:
    method = normalize_method(method)
    vector = vector_literal(dimensions)
    query = "term1 | topic1"
    where = f"WHERE {filter_clause}" if filter_clause else ""
    if method == "turbohybrid":
        return f"""
SELECT id
FROM turbohybrid_suite_docs
{where}
ORDER BY embedding <~> hybrid_query(
    vector_query => {vector},
    text_query => to_tsquery('english', '{query}'),
    dense_k => 128,
    bm25_k => 128,
    final_k => 10
)
LIMIT 10
"""

    if method == "postgres_sql_rrf":
        return f"""
WITH q AS MATERIALIZED (
    SELECT {vector} AS v, to_tsquery('english', '{query}') AS t
),
dense AS MATERIALIZED (
    SELECT id, row_number() OVER () AS rank
    FROM (
        SELECT d.id
        FROM turbohybrid_suite_docs d, q
        {where}
        ORDER BY d.embedding <=> q.v
        LIMIT 128
    ) s
),
lexical AS MATERIALIZED (
    SELECT id, row_number() OVER () AS rank
    FROM (
        SELECT d.id
        FROM turbohybrid_suite_docs d, q
        {where} {"AND" if where else "WHERE"} d.body_tsv @@ q.t
        ORDER BY ts_rank_cd(d.body_tsv, q.t) DESC, d.id
        LIMIT 128
    ) s
)
SELECT COALESCE(dense.id, lexical.id) AS id
FROM dense
FULL OUTER JOIN lexical USING (id)
ORDER BY
    COALESCE(1.0 / (60 + dense.rank), 0.0) +
    COALESCE(1.0 / (60 + lexical.rank), 0.0) DESC,
    COALESCE(dense.id, lexical.id)
LIMIT 10
"""
    if method == "turboquant_dense":
        return f"""
SELECT id
FROM turbohybrid_suite_docs
{where}
ORDER BY embedding <=> {vector}
LIMIT 10
"""
    raise ValueError(f"unsupported synthetic method: {method}")


def relation_bytes(database: str, relations: list[str]) -> int:
    if not relations:
        return 0
    names = ",".join("'" + relation + "'::regclass" for relation in relations)
    return int(run_psql(database, f"SELECT COALESCE(sum(pg_relation_size(r)), 0)::bigint FROM (VALUES ({names.replace(',', '),(')})) v(r)"))


def wal_lsn(database: str) -> str:
    return run_psql(database, "SELECT pg_current_wal_lsn()")


def wal_diff(database: str, before: str, after: str) -> int:
    return int(float(run_psql(database, f"SELECT pg_wal_lsn_diff('{after}', '{before}')")))


def timed_sql(database: str, sql: str) -> float:
    start = time.perf_counter()
    run_psql_file(database, sql)
    return (time.perf_counter() - start) * 1000.0


def mutation_probe(database: str, dimensions: int, rows: int) -> dict[str, Any]:
    start_id = rows + 1
    insert_count = max(10, rows // 100)
    before_insert = wal_lsn(database)
    insert_sql = f"""
INSERT INTO turbohybrid_suite_docs
SELECT
    i,
    i % 100,
    ('[' || array_to_string(ARRAY(
        SELECT (((i * (d + 23)) % 1000)::float8 / 1000.0)::text
        FROM generate_series(1, {dimensions}) d
    ), ',') || ']')::vector({dimensions}),
    'mutation term1 topic1',
    to_tsvector('english', 'mutation term1 topic1')
FROM generate_series({start_id}, {start_id + insert_count - 1}) i;
"""
    insert_ms = timed_sql(database, insert_sql)
    after_insert = wal_lsn(database)

    before_delete = wal_lsn(database)
    delete_ms = timed_sql(
        database,
        f"DELETE FROM turbohybrid_suite_docs WHERE id >= {start_id} AND id < {start_id + insert_count};",
    )
    after_delete = wal_lsn(database)

    before_vacuum = wal_lsn(database)
    vacuum_ms = timed_sql(database, "VACUUM turbohybrid_suite_docs;")
    after_vacuum = wal_lsn(database)

    return {
        "insert_rows": insert_count,
        "insert_ms": round(insert_ms, 3),
        "insert_wal_bytes": wal_diff(database, before_insert, after_insert),
        "delete_ms": round(delete_ms, 3),
        "delete_wal_bytes": wal_diff(database, before_delete, after_delete),
        "vacuum_ms": round(vacuum_ms, 3),
        "vacuum_wal_bytes": wal_diff(database, before_vacuum, after_vacuum),
    }


def run_system_synthetic(args: argparse.Namespace) -> None:
    methods = [normalize_method(m.strip()) for m in args.methods.split(",") if m.strip()]
    filter_scenarios = {
        "unfiltered": "",
        "selectivity_10pct": "bucket < 10",
        "selectivity_1pct": "bucket = 0",
    }

    run_psql_file(args.database, synthetic_setup_sql(args.rows, args.dimensions))
    results: list[dict[str, Any]] = []
    for method in methods:
        run_psql_file(args.database, drop_method_indexes_sql())
        build_sql, indexes = build_method_sql(method)
        before_build = wal_lsn(args.database)
        build_ms = timed_sql(args.database, build_sql + "\nANALYZE turbohybrid_suite_docs;")
        after_build = wal_lsn(args.database)
        index_bytes = relation_bytes(args.database, indexes)
        method_result: dict[str, Any] = {
            "method": method,
            "rows": args.rows,
            "dimensions": args.dimensions,
            "build_ms": round(build_ms, 3),
            "build_wal_bytes": wal_diff(args.database, before_build, after_build),
            "index_bytes": index_bytes,
            "query": {},
        }

        for scenario, clause in filter_scenarios.items():
            sql = query_sql(method, args.dimensions, clause)
            for _ in range(args.warmup):
                run_psql(args.database, sql)
            hot = run_query_batch(args.database, sql, args.runs, args.concurrency)
            cold_samples: list[float] = []
            for _ in range(args.cold_runs):
                if args.cold_command:
                    subprocess.run(args.cold_command, shell=True, check=True)
                cold_samples.append(timed_query(args.database, sql))
            method_result["query"][scenario] = {
                "hot": hot,
                "cold": summarize_latency(cold_samples),
                "cold_mode": "external_command" if args.cold_command else "new_psql_session_only",
            }

        method_result["mutation"] = mutation_probe(args.database, args.dimensions, args.rows)
        results.append(method_result)

    summary = {
        "suite": "turbohybrid_system_synthetic",
        "layer": "systems",
        "generated_at_unix": int(time.time()),
        "database": args.database,
        "postgres_settings": {
            "shared_buffers": run_psql(args.database, "SHOW shared_buffers"),
            "work_mem": run_psql(args.database, "SHOW work_mem"),
            "maintenance_work_mem": run_psql(args.database, "SHOW maintenance_work_mem"),
        },
        "results": results,
    }

    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


def simd_profile_query(shape: str, dimensions: int, dense_k: int, bm25_k: int, final_k: int) -> str:
    vector = vector_literal(dimensions)
    if shape == "dense_only":
        return f"""
SELECT id
FROM turbohybrid_suite_docs
ORDER BY embedding <~> hybrid_query(
    vector_query => {vector},
    dense_k => {dense_k},
    final_k => {final_k}
)
LIMIT {final_k}
"""
    if shape == "bm25_rare":
        text_query = "term1"
        return f"""
SELECT id
FROM turbohybrid_suite_docs
ORDER BY embedding <~> hybrid_query(
    text_query => to_tsquery('english', '{text_query}'),
    dense_k => 0,
    bm25_k => {bm25_k},
    final_k => {final_k}
)
LIMIT {final_k}
"""
    if shape == "bm25_common":
        text_query = "common"
        return f"""
SELECT id
FROM turbohybrid_suite_docs
ORDER BY embedding <~> hybrid_query(
    text_query => to_tsquery('english', '{text_query}'),
    dense_k => 0,
    bm25_k => {bm25_k},
    final_k => {final_k}
)
LIMIT {final_k}
"""
    if shape == "hybrid_rare":
        text_query = "term1"
    elif shape == "hybrid_common" or shape == "delta_heavy":
        text_query = "common"
    elif shape == "hybrid_no_lexical_match":
        text_query = "definitelymissingterm"
    else:
        raise ValueError(f"unsupported SIMD profile query shape: {shape}")

    return f"""
SELECT id
FROM turbohybrid_suite_docs
ORDER BY embedding <~> hybrid_query(
    vector_query => {vector},
    text_query => to_tsquery('english', '{text_query}'),
    dense_k => {dense_k},
    bm25_k => {bm25_k},
    final_k => {final_k}
)
LIMIT {final_k}
"""


def apply_set_prefix(settings: list[str]) -> str:
    return "\n".join(f"SET {setting};" for setting in settings)


def simd_profile_capture_stats(database: str, measured_sql: str) -> dict[str, Any]:
    sql = f"""
{measured_sql};
SELECT COALESCE(hybrid_last_scan_stats(), '{{}}'::jsonb)::text;
SELECT COALESCE(tq_last_scan_stats(), '{{}}'::jsonb)::text;
SELECT COALESCE(tq_last_simd_stats(), '{{}}'::jsonb)::text;
"""
    lines = [line for line in run_psql(database, sql).splitlines() if line]
    stats_lines = lines[-3:]
    return {
        "hybrid_last_scan_stats": json.loads(stats_lines[0]) if len(stats_lines) > 0 else {},
        "tq_last_scan_stats": json.loads(stats_lines[1]) if len(stats_lines) > 1 else {},
        "simd_kernel_counts": json.loads(stats_lines[2]) if len(stats_lines) > 2 else {},
    }


def timed_profile_query(database: str, sql: str, perf_command: str = "") -> float:
    if not perf_command:
        return timed_query(database, sql)

    command = f"{perf_command} psql -X -q -v ON_ERROR_STOP=1 -d {shlex.quote(database)} -At -c {shlex.quote(sql)}"
    start = time.perf_counter()
    subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL)
    return (time.perf_counter() - start) * 1000.0


def run_simd_profile(args: argparse.Namespace) -> None:
    settings = list(args.set or [])
    set_prefix = apply_set_prefix(settings)
    run_psql_file(args.database, synthetic_setup_sql(args.rows, args.dimensions))
    run_psql_file(args.database, drop_method_indexes_sql())

    build_sql, indexes = build_method_sql(args.method)
    before_build = wal_lsn(args.database)
    build_ms = timed_sql(args.database, set_prefix + "\n" + build_sql + "\nANALYZE turbohybrid_suite_docs;")
    after_build = wal_lsn(args.database)

    if args.include_delta:
        acceptance_insert_delta(args.database, args.dimensions, args.rows, 10)

    simd_capabilities: dict[str, Any]
    try:
        simd_capabilities = json.loads(run_psql(args.database, "LOAD 'vector'; SELECT tq_simd_capabilities()::text;"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        simd_capabilities = {}

    cases = [
        "dense_only",
        "bm25_rare",
        "bm25_common",
        "hybrid_rare",
        "hybrid_common",
        "hybrid_no_lexical_match",
        "delta_heavy",
    ]
    results: list[dict[str, Any]] = []
    for shape in cases:
        query = simd_profile_query(shape, args.dimensions, args.dense_k, args.bm25_k, args.final_k)
        measured_sql = set_prefix + "\n" + query
        for _ in range(args.warmup_runs):
            run_psql(args.database, measured_sql)
        samples = [
            timed_profile_query(args.database, measured_sql, args.perf_command)
            for _ in range(args.runs)
        ]
        try:
            result_ids = [
                int(line)
                for line in run_psql(args.database, measured_sql).splitlines()
                if line.strip()
            ]
        except (subprocess.CalledProcessError, ValueError):
            result_ids = []
        stats = simd_profile_capture_stats(args.database, measured_sql)
        explain = run_psql_text(args.database, set_prefix + "\nEXPLAIN (ANALYZE, BUFFERS) " + query)
        row = {
            "query_shape": shape,
            "rows": args.rows,
            "dimensions": args.dimensions,
            "dense_k": args.dense_k,
            "bm25_k": args.bm25_k,
            "final_k": args.final_k,
            **summarize_latency(samples),
            "sample_runs": [
                {"run_number": index + 1, "latency_ms": round(sample, 3)}
                for index, sample in enumerate(samples)
            ],
            "result_ids": result_ids,
            "hybrid_last_scan_stats": stats["hybrid_last_scan_stats"],
            "tq_last_scan_stats": stats["tq_last_scan_stats"],
            "simd_kernel_counts": stats["simd_kernel_counts"],
            "explain_analyze": explain,
        }
        results.append(row)

    summary = {
        "suite": "turbohybrid_simd_profile",
        "generated_at_unix": int(time.time()),
        "architecture": platform.machine() or "unknown",
        "cpu_model": cpu_model(),
        "compiler": compiler_info(),
        "postgres_version": postgres_version(args.database),
        "commit": git_sha(),
        "database": args.database,
        "method": args.method,
        "rows": args.rows,
        "dimensions": args.dimensions,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "dense_k": args.dense_k,
        "bm25_k": args.bm25_k,
        "final_k": args.final_k,
        "set": settings,
        "perf_command": args.perf_command,
        "simd_build_mode": os.environ.get("SIMD_BUILD", "portable"),
        "simd_capabilities": simd_capabilities,
        "build_ms": round(build_ms, 3),
        "build_wal_bytes": wal_diff(args.database, before_build, after_build),
        "index_bytes": relation_bytes(args.database, indexes),
        "results": results,
    }

    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


def bool_sql(value: str) -> str:
    if value.lower() in {"on", "true", "1", "yes"}:
        return "on"
    if value.lower() in {"off", "false", "0", "no"}:
        return "off"
    raise ValueError(f"expected boolean value, got {value!r}")


def build_turbohybrid_index_sql(precompute_tf_norm: str = "off") -> str:
    return f"""
CREATE INDEX turbohybrid_suite_hybrid_idx ON turbohybrid_suite_docs
USING turbohybrid (
    embedding vector_cosine_hybrid_ops,
    body_tsv bm25_tsvector_ops
)
INCLUDE (bucket)
WITH (
    graph_m = 16,
    graph_ef_construction = 128,
    graph_ef_search = 64,
    graph_oversampling = 4,
    tq_bits = 4,
    tq_exact_storage = on,
    bm25_block_max = on,
    bm25_precompute_tf_norm = {bool_sql(precompute_tf_norm)}
);
"""


def vector_opclass_for_metric(metric: str) -> str:
    if metric == "l2":
        return "vector_l2_ops"
    if metric == "ip":
        return "vector_ip_ops"
    if metric == "cosine":
        return "vector_cosine_ops"
    raise ValueError(f"unsupported vector metric: {metric}")


def vector_operator_for_metric(metric: str) -> str:
    if metric == "l2":
        return "<->"
    if metric == "ip":
        return "<#>"
    if metric == "cosine":
        return "<=>"
    raise ValueError(f"unsupported vector metric: {metric}")


def build_turboquant_dense_index_sql(tq_bits: int = 4, weighted: str = "off",
                                     metric: str = "cosine") -> str:
    opclass = vector_opclass_for_metric(metric)
    return f"""
CREATE INDEX turbohybrid_suite_turboquant_idx ON turbohybrid_suite_docs
USING turboquant (embedding {opclass})
WITH (
    graph_m = 16,
    graph_ef_construction = 128,
    graph_ef_search = 64,
    graph_oversampling = 4,
    tq_bits = {int(tq_bits)},
    tq_exact_storage = on,
    graph_rescore_band = exact,
    tq_weighted = {bool_sql(weighted)}
);
"""


def dense_turboquant_query_sql(dimensions: int, limit: int, metric: str = "cosine") -> str:
    operator = vector_operator_for_metric(metric)
    return f"""
SELECT id
FROM turbohybrid_suite_docs
ORDER BY embedding {operator} {vector_literal(dimensions)}
LIMIT {limit}
"""


def tq_query_capture(database: str, measured_sql: str) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    sql = f"""
{measured_sql};
SELECT COALESCE(tq_last_scan_stats(), '{{}}'::jsonb)::text;
SELECT COALESCE(tq_last_simd_stats(), '{{}}'::jsonb)::text;
"""
    lines = [line for line in run_psql(database, sql).splitlines() if line]
    if len(lines) < 2:
        return lines, {}, {}
    return lines[:-2], json.loads(lines[-2]), json.loads(lines[-1])


def bm25_debug_query(query_shape: str) -> str:
    if query_shape in {"common-term", "common", "bm25_common"}:
        return "common"
    if query_shape in {"rare-term", "rare", "bm25_rare"}:
        return "term1"
    if query_shape in {"two-term-or", "2-or"}:
        return "term1 | topic1"
    if query_shape in {"two-term-and", "2-and"}:
        return "term1 & topic1"
    if query_shape in {"five-term-or", "5-or"}:
        return "term1 | term2 | term3 | topic1 | topic2"
    if query_shape in {"no-match", "none"}:
        return "definitelymissingterm"
    raise ValueError(f"unsupported BM25 query shape: {query_shape}")


def bm25_debug_topk_sql(query_shape: str, k: int, use_wand: bool) -> str:
    tsquery = bm25_debug_query(query_shape)
    wand_sql = "true" if use_wand else "false"
    return (
        "SELECT tq_debug_bm25_topk('turbohybrid_suite_hybrid_idx'::regclass, "
        f"to_tsquery('english', {sql_literal(tsquery)}), {k}, {wand_sql})::text"
    )


def bm25_term_stats(database: str, term: str = "common") -> dict[str, Any]:
    try:
        raw = run_psql(
            database,
            "SELECT tq_debug_bm25_term_stats("
            "'turbohybrid_suite_hybrid_idx'::regclass, "
            f"{sql_literal(term)})::text",
        )
        return json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def bm25_debug_capture(database: str, measured_sql: str) -> tuple[dict[str, Any], dict[str, Any]]:
    sql = f"""
{measured_sql};
SELECT COALESCE(tq_last_simd_stats(), '{{}}'::jsonb)::text;
"""
    lines = [line for line in run_psql(database, sql).splitlines() if line]
    if not lines:
        return {}, {}
    if len(lines) == 1:
        return json.loads(lines[0]), {}
    return json.loads(lines[-2]), json.loads(lines[-1])


def run_bm25_decode_bench(args: argparse.Namespace) -> None:
    set_prefix = apply_set_prefix(list(args.set or []))
    run_psql_file(args.database, synthetic_setup_sql(args.rows, args.dimensions))
    run_psql_file(args.database, drop_method_indexes_sql())
    build_ms = timed_sql(
        args.database,
        build_turbohybrid_index_sql("off") + "\nANALYZE turbohybrid_suite_docs;",
    )
    query_sql = set_prefix + "\nSET hybrid.enable_wand = off;\n" + bm25_debug_topk_sql(
        "common-term", args.bm25_k, False
    )
    samples = [timed_query(args.database, query_sql) for _ in range(args.runs)]
    stats, simd_stats = bm25_debug_capture(args.database, query_sql)
    p50 = summarize_latency(samples)["p50_ms"]
    decoded = int(stats.get("postings_decoded", 0))
    summary = {
        "suite": "turbohybrid_bm25_decode",
        "generated_at_unix": int(time.time()),
        "database": args.database,
        "rows": args.rows,
        "dimensions": args.dimensions,
        "encoding": args.encoding,
        "runs": args.runs,
        "build_ms": round(build_ms, 3),
        "index_bytes": relation_bytes(args.database, ["turbohybrid_suite_hybrid_idx"]),
        "postings_decoded_per_second": round(decoded / (p50 / 1000.0), 3) if p50 > 0 else 0,
        "blocks_visited": int(stats.get("blocks_visited", 0)),
        **summarize_latency(samples),
        "stats": stats,
        "simd_stats": simd_stats,
        "term_stats": bm25_term_stats(args.database, "common"),
    }
    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


def run_bm25_score_bench(args: argparse.Namespace) -> None:
    settings = list(args.set or [])
    set_prefix = apply_set_prefix(settings)
    precompute = bool_sql(args.precompute_tf_norm)
    run_psql_file(args.database, synthetic_setup_sql(args.rows, args.dimensions))
    run_psql_file(args.database, drop_method_indexes_sql())
    build_ms = timed_sql(
        args.database,
        build_turbohybrid_index_sql(precompute) + "\nANALYZE turbohybrid_suite_docs;",
    )
    query_sql = set_prefix + "\nSET hybrid.enable_wand = off;\n" + bm25_debug_topk_sql(
        args.query_shape, args.bm25_k, False
    )
    samples = [timed_query(args.database, query_sql) for _ in range(args.runs)]
    stats, simd_stats = bm25_debug_capture(args.database, query_sql)
    parity = run_psql(
        args.database,
        set_prefix
        + "\nSELECT (tq_debug_bm25_topk('turbohybrid_suite_hybrid_idx'::regclass, "
        "to_tsquery('english', 'common'), 20, true)->'results') = "
        "(tq_debug_bm25_topk('turbohybrid_suite_hybrid_idx'::regclass, "
        "to_tsquery('english', 'common'), 20, false)->'results');",
    ) == "t"
    p50 = summarize_latency(samples)["p50_ms"]
    decoded = int(stats.get("postings_decoded", 0))
    summary = {
        "suite": "turbohybrid_bm25_score",
        "generated_at_unix": int(time.time()),
        "database": args.database,
        "rows": args.rows,
        "dimensions": args.dimensions,
        "query_shape": args.query_shape,
        "bm25_k": args.bm25_k,
        "precompute_tf_norm": precompute,
        "runs": args.runs,
        "set": settings,
        "build_ms": round(build_ms, 3),
        "index_bytes": relation_bytes(args.database, ["turbohybrid_suite_hybrid_idx"]),
        "postings_decoded_per_second": round(decoded / (p50 / 1000.0), 3) if p50 > 0 else 0,
        "candidates_scored_per_second": round(int(stats.get("candidates_scored", 0)) / (p50 / 1000.0), 3) if p50 > 0 else 0,
        "topk_parity": parity,
        **summarize_latency(samples),
        "stats": stats,
        "simd_stats": simd_stats,
    }
    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


def run_dense_batch_bench(args: argparse.Namespace) -> None:
    settings = list(args.set or [])
    set_prefix = apply_set_prefix(settings)
    run_psql_file(args.database, synthetic_setup_sql(args.rows, args.dimensions))
    run_psql_file(args.database, drop_method_indexes_sql())
    before_build = wal_lsn(args.database)
    build_ms = timed_sql(
        args.database,
        build_turboquant_dense_index_sql(args.tq_bits, "off") + "\nANALYZE turbohybrid_suite_docs;",
    )
    after_build = wal_lsn(args.database)
    query = (
        set_prefix
        + "\nSET enable_seqscan = off;\n"
        + dense_turboquant_query_sql(args.dimensions, args.dense_k)
    )
    for _ in range(args.warmup_runs):
        run_psql(args.database, query)
    samples = [timed_query(args.database, query) for _ in range(args.runs)]
    _, scan_stats, simd_stats = tq_query_capture(args.database, query)
    latency = summarize_latency(samples)
    p50 = latency["p50_ms"]
    summary = {
        "suite": "turbohybrid_dense_batch",
        "generated_at_unix": int(time.time()),
        "database": args.database,
        "rows": args.rows,
        "dimensions": args.dimensions,
        "dense_k": args.dense_k,
        "tq_bits": args.tq_bits,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "set": settings,
        "build_ms": round(build_ms, 3),
        "build_wal_bytes": wal_diff(args.database, before_build, after_build),
        "index_bytes": relation_bytes(args.database, ["turbohybrid_suite_turboquant_idx"]),
        "scored_codes_per_second": round(int(scan_stats.get("graph_scored_codes", 0)) / (p50 / 1000.0), 3) if p50 > 0 else 0,
        **latency,
        "scan_stats": scan_stats,
        "simd_stats": simd_stats,
    }
    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


def run_exact_rescore_bench(args: argparse.Namespace) -> None:
    settings = list(args.set or [])
    set_prefix = apply_set_prefix(settings)
    run_psql_file(args.database, synthetic_setup_sql(args.rows, args.dimensions))
    run_psql_file(args.database, drop_method_indexes_sql())
    before_build = wal_lsn(args.database)
    build_ms = timed_sql(
        args.database,
        build_turboquant_dense_index_sql(4, "off", args.metric) + "\nANALYZE turbohybrid_suite_docs;",
    )
    after_build = wal_lsn(args.database)
    query = (
        set_prefix
        + "\nSET enable_seqscan = off;"
        + f"\nSET hnsw.ef_search = {args.rescore_count};\n"
        + dense_turboquant_query_sql(args.dimensions, min(args.rescore_count, 1000), args.metric)
    )
    for _ in range(args.warmup_runs):
        run_psql(args.database, query)
    samples = [timed_query(args.database, query) for _ in range(args.runs)]
    _, scan_stats, simd_stats = tq_query_capture(args.database, query)
    latency = summarize_latency(samples)
    p50 = latency["p50_ms"]
    summary = {
        "suite": "turbohybrid_exact_rescore",
        "generated_at_unix": int(time.time()),
        "database": args.database,
        "rows": args.rows,
        "dimensions": args.dimensions,
        "metric": args.metric,
        "rescore_count": args.rescore_count,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "set": settings,
        "build_ms": round(build_ms, 3),
        "build_wal_bytes": wal_diff(args.database, before_build, after_build),
        "index_bytes": relation_bytes(args.database, ["turbohybrid_suite_turboquant_idx"]),
        "rescored_vectors_per_second": round(int(scan_stats.get("graph_rescore_count", 0)) / (p50 / 1000.0), 3) if p50 > 0 else 0,
        **latency,
        "scan_stats": scan_stats,
        "simd_stats": simd_stats,
    }
    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


def run_weighted_tq_bench(args: argparse.Namespace) -> None:
    settings = ["hnsw.tq_weighted=on", *(args.set or [])]
    set_prefix = apply_set_prefix(settings)
    run_psql_file(args.database, synthetic_setup_sql(args.rows, args.dimensions))
    run_psql_file(args.database, drop_method_indexes_sql())
    before_build = wal_lsn(args.database)
    build_ms = timed_sql(
        args.database,
        set_prefix + "\n" + build_turboquant_dense_index_sql(args.tq_bits, args.weighted) +
        "\nANALYZE turbohybrid_suite_docs;",
    )
    after_build = wal_lsn(args.database)
    query = (
        set_prefix
        + "\nSET enable_seqscan = off;\n"
        + dense_turboquant_query_sql(args.dimensions, 20)
    )
    for _ in range(args.warmup_runs):
        run_psql(args.database, query)
    samples = [timed_query(args.database, query) for _ in range(args.runs)]
    _, scan_stats, simd_stats = tq_query_capture(args.database, query)
    latency = summarize_latency(samples)
    p50 = latency["p50_ms"]
    summary = {
        "suite": "turbohybrid_weighted_tq",
        "generated_at_unix": int(time.time()),
        "database": args.database,
        "rows": args.rows,
        "dimensions": args.dimensions,
        "tq_bits": args.tq_bits,
        "weighted": bool_sql(args.weighted),
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "set": settings,
        "build_ms": round(build_ms, 3),
        "build_wal_bytes": wal_diff(args.database, before_build, after_build),
        "index_bytes": relation_bytes(args.database, ["turbohybrid_suite_turboquant_idx"]),
        "scored_codes_per_second": round(int(scan_stats.get("graph_scored_codes", 0)) / (p50 / 1000.0), 3) if p50 > 0 else 0,
        **latency,
        "scan_stats": scan_stats,
        "simd_stats": simd_stats,
    }
    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


def simd_matrix_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TurboHybrid SIMD Matrix",
        "",
        f"- architecture: {summary.get('architecture', 'unknown')}",
        f"- cpu: {summary.get('cpu_model', 'unknown')}",
        f"- compiler: {summary.get('compiler', 'unknown')}",
        f"- postgres: {summary.get('postgres_version', 'unknown')}",
        f"- commit: {summary.get('commit', 'unknown')}",
        "",
        "| case | scalar p95 | auto p95 | speedup | kernel | parity | notes |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary["summary_rows"]:
        lines.append(
            f"| {row['case']} | {row['scalar_p95_ms']:.3f} | "
            f"{row['auto_p95_ms']:.3f} | {row['speedup']:.2f}x | "
            f"{row['kernel']} | {row['parity']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Defaults",
            "",
            "- Use `SIMD_BUILD=portable` for package builds.",
            "- Use `SIMD_BUILD=native` only for local benchmark binaries.",
            "- Keep scalar force GUCs available for parity and incident fallback.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_simd_matrix(args: argparse.Namespace) -> None:
    dimensions = parse_int_list(args.dimensions)
    rows_values = parse_int_list(args.rows)
    budgets = parse_int_list(args.candidate_budgets)
    profiles: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for rows in rows_values:
        for dimensions_value in dimensions:
            for budget in budgets:
                mode_outputs: dict[str, dict[str, Any]] = {}
                for mode in ["scalar", "auto"]:
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                        path = Path(handle.name)
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "run-simd-profile",
                        "--database",
                        args.database,
                        "--rows",
                        str(rows),
                        "--dimensions",
                        str(dimensions_value),
                        "--runs",
                        str(args.runs),
                        "--warmup-runs",
                        str(args.warmup_runs),
                        "--dense-k",
                        str(budget),
                        "--bm25-k",
                        str(budget),
                        "--final-k",
                        str(args.final_k),
                        "--output",
                        str(path),
                    ]
                    if mode == "scalar":
                        command.extend(
                            [
                                "--set",
                                "hnsw.tq_simd_force=scalar",
                                "--set",
                                "hnsw.tq_exact_simd_force=scalar",
                                "--set",
                                "hybrid.bm25_simd_force=scalar",
                            ]
                        )
                    else:
                        command.extend(
                            [
                                "--set",
                                "hnsw.tq_simd_force=auto",
                                "--set",
                                "hnsw.tq_exact_simd_force=auto",
                                "--set",
                                "hybrid.bm25_simd_force=auto",
                            ]
                        )
                    if args.perf:
                        command.extend(["--perf-command", args.perf])
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                    mode_outputs[mode] = load_json(path)
                    path.unlink(missing_ok=True)
                profiles.extend(mode_outputs.values())
                scalar_by_shape = {row["query_shape"]: row for row in mode_outputs["scalar"]["results"]}
                auto_by_shape = {row["query_shape"]: row for row in mode_outputs["auto"]["results"]}
                for shape, scalar_row in scalar_by_shape.items():
                    auto_row = auto_by_shape[shape]
                    scalar_p95 = float(scalar_row["p95_ms"])
                    auto_p95 = float(auto_row["p95_ms"])
                    scalar_ids = scalar_row.get("result_ids", [])
                    auto_ids = auto_row.get("result_ids", [])
                    parity = "pass" if scalar_ids == auto_ids else (
                        "pass_set" if sorted(scalar_ids) == sorted(auto_ids) else "fail"
                    )
                    kernel = auto_row.get("tq_last_scan_stats", {}).get(
                        "dense_simd_kernel",
                        auto_row.get("simd_kernel_counts", {}).get("bm25_score_kernel", "unknown"),
                    )
                    summary_rows.append(
                        {
                            "case": f"{shape}_{dimensions_value}d_k{budget}_rows{rows}",
                            "scalar_p95_ms": scalar_p95,
                            "auto_p95_ms": auto_p95,
                            "speedup": round(scalar_p95 / auto_p95, 3) if auto_p95 > 0 else 0,
                            "kernel": kernel,
                            "parity": parity,
                            "notes": "profile wrapper",
                        }
                    )
    summary = {
        "suite": "turbohybrid_simd_matrix",
        "generated_at_unix": int(time.time()),
        "architecture": platform.machine() or "unknown",
        "cpu_model": cpu_model(),
        "compiler": compiler_info(),
        "postgres_version": postgres_version(args.database),
        "commit": git_sha(),
        "database": args.database,
        "profiles": profiles,
        "summary_rows": summary_rows,
    }
    output_json = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(output_json + "\n", encoding="utf-8")
    else:
        print(output_json)
    if args.output_md:
        Path(args.output_md).write_text(simd_matrix_markdown(summary), encoding="utf-8")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            qid, _, docid, rel = line.split()[:4]
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def read_run(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter=" "):
            row = [item for item in row if item]
            if len(row) < 6:
                continue
            qid, docid, rank = row[0], row[2], int(row[3])
            run.setdefault(qid, []).append((rank, docid))
    return {qid: [docid for _, docid in sorted(rows)] for qid, rows in run.items()}


def dcg(gains: list[int]) -> float:
    return sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))


def score_run(args: argparse.Namespace) -> None:
    qrels = read_qrels(Path(args.qrels))
    run = read_run(Path(args.run))
    k_values = [int(value) for value in args.k.split(",")]
    metrics: dict[str, float] = {}
    query_count = len(qrels)

    for k in k_values:
        recall_total = 0.0
        ndcg_total = 0.0
        mrr_total = 0.0
        map_total = 0.0
        for qid, relevant in qrels.items():
            docs = run.get(qid, [])[:k]
            relevant_docs = {docid for docid, rel in relevant.items() if rel > 0}
            gains = [relevant.get(docid, 0) for docid in docs]
            ideal = sorted((rel for rel in relevant.values() if rel > 0), reverse=True)[:k]

            hits = 0
            precision_sum = 0.0
            reciprocal = 0.0
            for rank, docid in enumerate(docs, start=1):
                if docid in relevant_docs:
                    hits += 1
                    precision_sum += hits / rank
                    if reciprocal == 0.0:
                        reciprocal = 1.0 / rank

            recall_total += hits / len(relevant_docs) if relevant_docs else 0.0
            ndcg_total += dcg(gains) / dcg(ideal) if ideal else 0.0
            mrr_total += reciprocal
            map_total += precision_sum / len(relevant_docs) if relevant_docs else 0.0

        denom = max(query_count, 1)
        metrics[f"recall@{k}"] = round(recall_total / denom, 6)
        metrics[f"ndcg@{k}"] = round(ndcg_total / denom, 6)
        metrics[f"mrr@{k}"] = round(mrr_total / denom, 6)
        metrics[f"map@{k}"] = round(map_total / denom, 6)

    print(json.dumps({
        "suite": "turbohybrid_ir_quality",
        "layer": "ir_quality",
        "qrels": args.qrels,
        "run": args.run,
        "queries": query_count,
        "metrics": metrics,
    }, indent=2, sort_keys=True))


def jsonl_id(row: dict[str, Any]) -> str:
    if "_id" in row:
        return str(row["_id"])
    return str(row["id"])


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[jsonl_id(row)] = row
    return rows


def read_beir_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0] in {"query-id", "qid"}:
                continue
            if len(row) >= 4:
                qid, docid, rel = row[0], row[2], int(float(row[3]))
            elif len(row) >= 3:
                qid, docid, rel = row[0], row[1], int(float(row[2]))
            else:
                continue
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


def read_embedding_jsonl(
    path: Path,
    *,
    allowed_ids: set[str] | None = None,
    limit: int | None = None,
) -> tuple[dict[str, list[float]], int]:
    embeddings: dict[str, list[float]] = {}
    dimensions = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = jsonl_id(row)
            if allowed_ids is not None and row_id not in allowed_ids:
                continue
            values = [float(value) for value in row["values"]]
            if dimensions == 0:
                dimensions = len(values)
            elif len(values) != dimensions:
                raise ValueError(f"embedding dimension mismatch for id {row_id}")
            embeddings[row_id] = values
            if limit is not None and len(embeddings) >= limit:
                break
    return embeddings, dimensions


def select_real_rag_queries(
    qrels: dict[str, dict[str, int]],
    query_embeddings: dict[str, list[float]],
    queries: dict[str, dict[str, Any]],
    max_queries: int,
) -> list[str]:
    selected: list[str] = []
    for qid, relevant in qrels.items():
        if qid not in query_embeddings or qid not in queries:
            continue
        if not any(rel > 0 for rel in relevant.values()):
            continue
        selected.append(qid)
        if max_queries > 0 and len(selected) >= max_queries:
            break
    return selected


def copy_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def vector_input(values: list[float]) -> str:
    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


def vector_sql_literal_from_values(values: list[float], dimensions: int) -> str:
    return sql_literal(vector_input(values)) + f"::vector({dimensions})"


def copy_real_rag_docs(
    database: str,
    rows: list[tuple[str, list[float], str]],
) -> None:
    proc = subprocess.Popen(
        [
            "psql",
            "-X",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
            "-d",
            database,
            "-c",
            "COPY turbohybrid_real_docs (doc_id, embedding, body) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    for doc_id, embedding, body in rows:
        proc.stdin.write(
            f"{copy_escape(doc_id)}\t{copy_escape(vector_input(embedding))}\t{copy_escape(body)}\n"
        )
    proc.stdin.write("\\.\n")
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        if stdout:
            sys.stderr.write(stdout)
        if stderr:
            sys.stderr.write(stderr)
        raise subprocess.CalledProcessError(proc.returncode, "psql", output=stdout, stderr=stderr)


def copy_real_rag_queries(
    database: str,
    rows: list[tuple[str, list[float], str]],
) -> None:
    proc = subprocess.Popen(
        [
            "psql",
            "-X",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
            "-d",
            database,
            "-c",
            "COPY turbohybrid_real_queries (qid, embedding, query_text) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    for qid, embedding, query_text in rows:
        proc.stdin.write(
            f"{copy_escape(qid)}\t{copy_escape(vector_input(embedding))}\t{copy_escape(query_text)}\n"
        )
    proc.stdin.write("\\.\n")
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        if stdout:
            sys.stderr.write(stdout)
        if stderr:
            sys.stderr.write(stderr)
        raise subprocess.CalledProcessError(proc.returncode, "psql", output=stdout, stderr=stderr)


def real_rag_setup_sql(dimensions: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS turbohybrid_real_docs;
DROP TABLE IF EXISTS turbohybrid_real_queries;
CREATE TABLE turbohybrid_real_docs (
    doc_id text PRIMARY KEY,
    embedding vector({dimensions}) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);
CREATE TABLE turbohybrid_real_queries (
    qid text PRIMARY KEY,
    embedding vector({dimensions}) NOT NULL,
    query_text text NOT NULL
);
"""


def drop_real_rag_indexes_sql() -> str:
    return """
DROP INDEX IF EXISTS turbohybrid_real_hybrid_idx;
DROP INDEX IF EXISTS turbohybrid_real_hnsw_idx;
DROP INDEX IF EXISTS turbohybrid_real_fts_idx;
"""


def real_rag_build_method_sql(method: str) -> tuple[str, list[str]]:
    method = normalize_method(method)
    if method in {"turbohybrid", "turbohybrid_exact_storage_off"}:
        exact_storage = "off" if method == "turbohybrid_exact_storage_off" else "on"
        return (
            f"""
CREATE INDEX turbohybrid_real_hybrid_idx ON turbohybrid_real_docs
USING turbohybrid (
    embedding vector_cosine_hybrid_ops,
    body_tsv bm25_tsvector_ops
)
WITH (
    graph_m = 16,
    graph_ef_construction = 128,
    graph_ef_search = 64,
    graph_oversampling = 4,
    tq_bits = 4,
    tq_exact_storage = {exact_storage},
    bm25_block_max = on,
    hybrid_default_dense_k = 400,
    hybrid_default_bm25_k = 400,
    hybrid_default_rrf_k = 60
);
""",
            ["turbohybrid_real_hybrid_idx"],
        )
    if method == "postgres_sql_rrf":
        return (
            """
CREATE INDEX turbohybrid_real_hnsw_idx ON turbohybrid_real_docs
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);
CREATE INDEX turbohybrid_real_fts_idx ON turbohybrid_real_docs
USING gin (body_tsv);
""",
            ["turbohybrid_real_hnsw_idx", "turbohybrid_real_fts_idx"],
        )
    raise ValueError(f"unsupported real RAG method: {method}")


def real_rag_query_sql(
    method: str,
    *,
    query_embedding: list[float],
    query_text: str,
    dimensions: int,
    dense_k: int,
    bm25_k: int,
    final_k: int,
    rrf_k: int,
) -> str:
    method = normalize_method(method)
    vector = vector_sql_literal_from_values(query_embedding, dimensions)
    text = sql_literal(query_text)
    prefix = "SET enable_seqscan = off;\n"
    if method in {"turbohybrid", "turbohybrid_exact_storage_off"}:
        return f"""
{prefix}
SELECT doc_id
FROM turbohybrid_real_docs
ORDER BY embedding <~> hybrid_query(
    vector_query => {vector},
    text_query => plainto_tsquery('english', {text}),
    dense_k => {dense_k},
    bm25_k => {bm25_k},
    rrf_k => {rrf_k},
    final_k => {final_k},
    require_bm25_match => false
)
LIMIT {final_k}
"""

    if method == "postgres_sql_rrf":
        return f"""
{prefix}
SET hnsw.ef_search = {max(dense_k, final_k)};
WITH q AS MATERIALIZED (
    SELECT {vector} AS v, plainto_tsquery('english', {text}) AS t
),
dense AS MATERIALIZED (
    SELECT doc_id, row_number() OVER () AS rank
    FROM (
        SELECT d.doc_id
        FROM turbohybrid_real_docs d, q
        ORDER BY d.embedding <=> q.v
        LIMIT {dense_k}
    ) s
),
lexical AS MATERIALIZED (
    SELECT doc_id, row_number() OVER () AS rank
    FROM (
        SELECT d.doc_id
        FROM turbohybrid_real_docs d, q
        WHERE d.body_tsv @@ q.t
        ORDER BY ts_rank_cd(d.body_tsv, q.t) DESC, d.doc_id
        LIMIT {bm25_k}
    ) s
)
SELECT COALESCE(dense.doc_id, lexical.doc_id) AS doc_id
FROM dense
FULL OUTER JOIN lexical USING (doc_id)
ORDER BY
    COALESCE(1.0 / ({rrf_k} + dense.rank), 0.0) +
    COALESCE(1.0 / ({rrf_k} + lexical.rank), 0.0) DESC,
    COALESCE(dense.doc_id, lexical.doc_id)
LIMIT {final_k}
"""
    raise ValueError(f"unsupported real RAG method: {method}")


def fetch_ids(database: str, sql: str) -> list[str]:
    output = run_psql(database, sql)
    return [line for line in output.splitlines() if line]


def real_rag_batch_sql(
    method: str,
    *,
    dimensions: int,
    dense_k: int,
    bm25_k: int,
    final_k: int,
    rrf_k: int,
    warmup: int,
    runs: int,
) -> str:
    method = normalize_method(method)
    measured_first_pass = warmup + 1
    total_passes = warmup + runs
    prefix = "SET enable_seqscan = off;\n"
    if method == "postgres_sql_rrf":
        prefix += f"SET hnsw.ef_search = {max(dense_k, final_k)};\n"
        query_insert = f"""
                EXECUTE format($query$
                    WITH dense AS MATERIALIZED (
                        SELECT doc_id, row_number() OVER () AS rank
                        FROM (
                            SELECT d.doc_id
                            FROM turbohybrid_real_docs d
                            ORDER BY d.embedding <=> %L::vector({dimensions})
                            LIMIT {dense_k}
                        ) s
                    ),
                    lexical AS MATERIALIZED (
                        SELECT doc_id, row_number() OVER () AS rank
                        FROM (
                            SELECT d.doc_id
                            FROM turbohybrid_real_docs d
                            WHERE d.body_tsv @@ plainto_tsquery('english', %L)
                            ORDER BY ts_rank_cd(d.body_tsv, plainto_tsquery('english', %L)) DESC, d.doc_id
                            LIMIT {bm25_k}
                        ) s
                    ),
                    fused AS (
                        SELECT COALESCE(dense.doc_id, lexical.doc_id) AS doc_id
                        FROM dense
                        FULL OUTER JOIN lexical USING (doc_id)
                        ORDER BY
                            COALESCE(1.0 / ({rrf_k} + dense.rank), 0.0) +
                            COALESCE(1.0 / ({rrf_k} + lexical.rank), 0.0) DESC,
                            COALESCE(dense.doc_id, lexical.doc_id)
                        LIMIT {final_k}
                    )
                    INSERT INTO turbohybrid_real_results(qid, rank, doc_id)
                    SELECT %L, row_number() OVER (), doc_id
                    FROM fused
                $query$, query_row.embedding::text, query_row.query_text, query_row.query_text, query_row.qid);
"""
        query_perform = f"""
                EXECUTE format($query$
                    SELECT doc_id
                    FROM (
                        WITH dense AS MATERIALIZED (
                            SELECT doc_id, row_number() OVER () AS rank
                            FROM (
                                SELECT d.doc_id
                                FROM turbohybrid_real_docs d
                                ORDER BY d.embedding <=> %L::vector({dimensions})
                                LIMIT {dense_k}
                            ) s
                        ),
                        lexical AS MATERIALIZED (
                            SELECT doc_id, row_number() OVER () AS rank
                            FROM (
                                SELECT d.doc_id
                                FROM turbohybrid_real_docs d
                                WHERE d.body_tsv @@ plainto_tsquery('english', %L)
                                ORDER BY ts_rank_cd(d.body_tsv, plainto_tsquery('english', %L)) DESC, d.doc_id
                                LIMIT {bm25_k}
                            ) s
                        )
                        SELECT COALESCE(dense.doc_id, lexical.doc_id) AS doc_id
                        FROM dense
                        FULL OUTER JOIN lexical USING (doc_id)
                        ORDER BY
                            COALESCE(1.0 / ({rrf_k} + dense.rank), 0.0) +
                            COALESCE(1.0 / ({rrf_k} + lexical.rank), 0.0) DESC,
                            COALESCE(dense.doc_id, lexical.doc_id)
                        LIMIT {final_k}
                    ) s
                $query$, query_row.embedding::text, query_row.query_text, query_row.query_text);
"""
    elif method in {"turbohybrid", "turbohybrid_exact_storage_off"}:
        query_insert = f"""
                result_rank := 0;
                FOR result_row IN EXECUTE format($query$
                        SELECT doc_id
                        FROM turbohybrid_real_docs
                        ORDER BY embedding <~> hybrid_query(
                            vector_query => %L::vector({dimensions}),
                            text_query => plainto_tsquery('english', %L),
                            dense_k => {dense_k},
                            bm25_k => {bm25_k},
                            rrf_k => {rrf_k},
                            final_k => {final_k},
                            require_bm25_match => false
                        )
                        LIMIT {final_k}
                $query$, query_row.embedding::text, query_row.query_text)
                LOOP
                    result_rank := result_rank + 1;
                    INSERT INTO turbohybrid_real_results(qid, rank, doc_id)
                    VALUES (query_row.qid, result_rank, result_row.doc_id);
                END LOOP;
"""
        query_perform = f"""
                FOR result_row IN EXECUTE format($query$
                        SELECT doc_id
                        FROM turbohybrid_real_docs
                        ORDER BY embedding <~> hybrid_query(
                            vector_query => %L::vector({dimensions}),
                            text_query => plainto_tsquery('english', %L),
                            dense_k => {dense_k},
                            bm25_k => {bm25_k},
                            rrf_k => {rrf_k},
                            final_k => {final_k},
                            require_bm25_match => false
                        )
                        LIMIT {final_k}
                $query$, query_row.embedding::text, query_row.query_text)
                LOOP
                END LOOP;
"""
    else:
        raise ValueError(f"unsupported real RAG method: {method}")

    return f"""
{prefix}
DROP TABLE IF EXISTS turbohybrid_real_results;
DROP TABLE IF EXISTS turbohybrid_real_timings;
CREATE TEMP TABLE turbohybrid_real_results (
    qid text NOT NULL,
    rank int NOT NULL,
    doc_id text NOT NULL
);
CREATE TEMP TABLE turbohybrid_real_timings (
    sample_no bigserial PRIMARY KEY,
    qid text NOT NULL,
    latency_ms float8 NOT NULL
);

DO $benchmark$
DECLARE
    query_row record;
    result_row record;
    result_rank int;
    pass_no int;
    start_ts timestamptz;
    elapsed_ms float8;
BEGIN
    FOR pass_no IN 1..{total_passes} LOOP
        FOR query_row IN
            SELECT qid, embedding, query_text
            FROM turbohybrid_real_queries
            ORDER BY qid
        LOOP
            start_ts := clock_timestamp();
            IF pass_no = {measured_first_pass} THEN
{query_insert}
            ELSE
{query_perform}
            END IF;
            elapsed_ms := EXTRACT(EPOCH FROM (clock_timestamp() - start_ts)) * 1000.0;
            IF pass_no >= {measured_first_pass} THEN
                INSERT INTO turbohybrid_real_timings(qid, latency_ms)
                VALUES (query_row.qid, elapsed_ms);
            END IF;
        END LOOP;
    END LOOP;
END
$benchmark$;

SELECT jsonb_build_object(
    'rankings',
    COALESCE(
        (
            SELECT jsonb_object_agg(qid, docs)
            FROM (
                SELECT qid, jsonb_agg(doc_id ORDER BY rank) AS docs
                FROM turbohybrid_real_results
                GROUP BY qid
            ) ranked
        ),
        '{{}}'::jsonb
    ),
    'samples_ms',
    COALESCE(
        (
            SELECT jsonb_agg(latency_ms ORDER BY sample_no)
            FROM turbohybrid_real_timings
        ),
        '[]'::jsonb
    )
)::text;
"""


def score_rankings(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int) -> dict[str, float]:
    recall_total = 0.0
    ndcg_total = 0.0
    mrr_total = 0.0
    map_total = 0.0
    query_count = len(qrels)
    for qid, relevant in qrels.items():
        docs = rankings.get(qid, [])[:k]
        relevant_docs = {docid for docid, rel in relevant.items() if rel > 0}
        gains = [relevant.get(docid, 0) for docid in docs]
        ideal = sorted((rel for rel in relevant.values() if rel > 0), reverse=True)[:k]

        hits = 0
        precision_sum = 0.0
        reciprocal = 0.0
        for rank, docid in enumerate(docs, start=1):
            if docid in relevant_docs:
                hits += 1
                precision_sum += hits / rank
                if reciprocal == 0.0:
                    reciprocal = 1.0 / rank

        recall_total += hits / len(relevant_docs) if relevant_docs else 0.0
        ndcg_total += dcg(gains) / dcg(ideal) if ideal else 0.0
        mrr_total += reciprocal
        map_total += precision_sum / len(relevant_docs) if relevant_docs else 0.0

    denom = max(query_count, 1)
    return {
        f"recall@{k}": round(recall_total / denom, 6),
        f"ndcg@{k}": round(ndcg_total / denom, 6),
        f"mrr@{k}": round(mrr_total / denom, 6),
        f"map@{k}": round(map_total / denom, 6),
    }


def real_rag_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Real RAG Benchmark",
        "",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Dataset path: `{summary['dataset_path']}`",
        f"- Commit: `{summary['commit']}`",
        f"- PostgreSQL: `{summary['postgres_version']}`",
        f"- Rows: {summary['rows']}",
        f"- Query count: {summary['queries']}",
        f"- Dimensions: {summary['dimensions']}",
        f"- Dense candidates: {summary['dense_k']}",
        f"- BM25 candidates: {summary['bm25_k']}",
        f"- Final k: {summary['final_k']}",
        f"- Warmup passes: {summary['warmup']}",
        f"- Timed passes: {summary['runs']}",
        "",
        "## Quality And Latency",
        "",
        "| Method | nDCG@k | Recall@k | MRR@k | MAP@k | p50 ms | p95 ms | p99 ms | Mean ms | QPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_k = summary["final_k"]
    for row in summary["results"]:
        metrics = row["metrics"]
        hot = row["latency"]
        lines.append(
            f"| {row['method']} | {metrics[f'ndcg@{metric_k}']} | "
            f"{metrics[f'recall@{metric_k}']} | {metrics[f'mrr@{metric_k}']} | "
            f"{metrics[f'map@{metric_k}']} | {hot['p50_ms']} | {hot['p95_ms']} | "
            f"{hot['p99_ms']} | {hot['mean_ms']} | {hot['qps']} |"
        )
    lines.extend([
        "",
        "## Build And Storage",
        "",
        "| Method | Build ms | Build WAL bytes | Index bytes | Index MiB |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary["results"]:
        mib = round(row["index_bytes"] / (1024 * 1024), 2)
        lines.append(
            f"| {row['method']} | {row['build_ms']} | {row['build_wal_bytes']} | "
            f"{row['index_bytes']} | {mib} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `postgres_sql_rrf` is normal pgvector HNSW for dense retrieval plus PostgreSQL full-text search and SQL-level reciprocal rank fusion.",
        "- `turbohybrid` is the single TurboHybrid index over the same vector and `tsvector` columns.",
        "- `turbohybrid_exact_storage_off` is the same TurboHybrid index with `tq_exact_storage = off`.",
        "- Both methods use the same text-embedding vectors, query texts, qrels, candidate budgets, and final-k.",
        "- Latency is measured inside PostgreSQL with `clock_timestamp()` around each retrieval query.",
        "",
    ])
    return "\n".join(lines)


def run_real_rag(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    corpus_path = dataset / "corpus.jsonl"
    queries_path = dataset / "queries.jsonl"
    corpus_embeddings_path = dataset / "corpus_embeddings.jsonl"
    query_embeddings_path = dataset / "query_embeddings.jsonl"
    qrels_path = dataset / "qrels" / f"{args.split}.tsv"

    for path in [corpus_path, queries_path, corpus_embeddings_path, query_embeddings_path, qrels_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    qrels_all = read_beir_qrels(qrels_path)
    queries = read_jsonl_by_id(queries_path)
    query_embeddings, query_dimensions = read_embedding_jsonl(query_embeddings_path)
    query_ids = select_real_rag_queries(qrels_all, query_embeddings, queries, args.max_queries)
    if not query_ids:
        raise ValueError("no benchmark queries have qrels, query text, and embeddings")

    selected_qrels = {qid: qrels_all[qid] for qid in query_ids}
    relevant_doc_ids = {
        docid
        for qid in query_ids
        for docid, rel in qrels_all[qid].items()
        if rel > 0
    }
    allowed_doc_ids: set[str] | None = None
    if args.max_docs > 0:
        allowed_doc_ids = set(relevant_doc_ids)
        with corpus_embeddings_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                doc_id = jsonl_id(json.loads(line))
                allowed_doc_ids.add(doc_id)
                if len(allowed_doc_ids) >= args.max_docs:
                    break

    corpus_embeddings, dimensions = read_embedding_jsonl(
        corpus_embeddings_path,
        allowed_ids=allowed_doc_ids,
        limit=len(allowed_doc_ids) if allowed_doc_ids is not None else None,
    )
    if dimensions != query_dimensions:
        raise ValueError(f"query dimension {query_dimensions} does not match corpus dimension {dimensions}")

    corpus_docs = read_jsonl_by_id(corpus_path)
    rows: list[tuple[str, list[float], str]] = []
    for doc_id, embedding in corpus_embeddings.items():
        doc = corpus_docs.get(doc_id)
        if not doc:
            continue
        body = " ".join(str(value).strip() for value in [doc.get("title", ""), doc.get("text", "")] if str(value).strip())
        rows.append((doc_id, embedding, body))

    present_doc_ids = {doc_id for doc_id, _, _ in rows}
    selected_qrels = {
        qid: {docid: rel for docid, rel in qrels.items() if docid in present_doc_ids}
        for qid, qrels in selected_qrels.items()
    }
    selected_qrels = {
        qid: qrels for qid, qrels in selected_qrels.items() if any(rel > 0 for rel in qrels.values())
    }
    query_ids = [qid for qid in query_ids if qid in selected_qrels]
    if not query_ids:
        raise ValueError("selected document subset removed all relevant qrels")

    run_psql_file(args.database, real_rag_setup_sql(dimensions))
    copy_real_rag_docs(args.database, rows)
    copy_real_rag_queries(
        args.database,
        [
            (qid, query_embeddings[qid], str(queries[qid].get("text", "")))
            for qid in query_ids
        ],
    )
    run_psql_file(args.database, "ANALYZE turbohybrid_real_docs;")

    methods = [normalize_method(method) for method in parse_str_list(args.methods)]
    results: list[dict[str, Any]] = []
    for method in methods:
        run_psql_file(args.database, drop_real_rag_indexes_sql())
        build_sql, indexes = real_rag_build_method_sql(method)
        before_build = wal_lsn(args.database)
        build_ms = timed_sql(args.database, build_sql + "\nANALYZE turbohybrid_real_docs;")
        after_build = wal_lsn(args.database)

        batch = json.loads(run_psql(
            args.database,
            real_rag_batch_sql(
                method,
                dimensions=dimensions,
                dense_k=args.dense_k,
                bm25_k=args.bm25_k,
                final_k=args.final_k,
                rrf_k=args.rrf_k,
                warmup=args.warmup,
                runs=args.runs,
            ),
        ))
        rankings = {
            str(qid): [str(doc_id) for doc_id in docs]
            for qid, docs in batch.get("rankings", {}).items()
        }
        samples = [float(value) for value in batch.get("samples_ms", [])]
        latency = summarize_latency(samples)
        total_elapsed = sum(samples) / 1000.0
        latency["qps"] = round(len(samples) / total_elapsed, 3) if total_elapsed > 0 else 0.0
        latency["runs"] = len(samples)
        results.append({
            "method": method,
            "build_ms": round(build_ms, 3),
            "build_wal_bytes": wal_diff(args.database, before_build, after_build),
            "index_bytes": relation_bytes(args.database, indexes),
            "metrics": score_rankings(rankings, selected_qrels, args.final_k),
            "latency": latency,
        })

    summary = {
        "suite": "turbohybrid_real_rag",
        "layer": "ir_quality_and_systems",
        "generated_at_unix": int(time.time()),
        "commit": git_sha(),
        "database": args.database,
        "postgres_version": postgres_version(args.database),
        "dataset_name": args.dataset_name,
        "dataset_path": str(dataset),
        "split": args.split,
        "rows": len(rows),
        "queries": len(query_ids),
        "dimensions": dimensions,
        "dense_k": args.dense_k,
        "bm25_k": args.bm25_k,
        "rrf_k": args.rrf_k,
        "final_k": args.final_k,
        "warmup": args.warmup,
        "runs": args.runs,
        "results": results,
    }

    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(output + "\n", encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(real_rag_markdown(summary), encoding="utf-8")
    print(output)


def list_suite(_: argparse.Namespace) -> None:
    datasets = load_json(CONFIG_DIR / "datasets.json")
    methods = load_json(CONFIG_DIR / "methods.json")
    print("Datasets")
    for dataset in datasets["datasets"]:
        print(f"- {dataset['name']}: {dataset['layer']} / {dataset['family']}")
    print("\nMethods")
    for method in methods["methods"]:
        print(f"- {method['name']}: {method['layer']} / {method['kind']}")


def plan(_: argparse.Namespace) -> None:
    datasets = load_json(CONFIG_DIR / "datasets.json")["datasets"]
    methods = load_json(CONFIG_DIR / "methods.json")["methods"]
    matrix = []
    for dataset in datasets:
        layer_methods = [
            method["name"] for method in methods
            if method["name"] == "turbohybrid" or
            method["layer"] in {dataset["layer"], "reference"}
        ]
        matrix.append({
            "dataset": dataset["name"],
            "layer": dataset["layer"],
            "metrics": dataset["metrics"],
            "methods": layer_methods,
        })
    print(json.dumps({"benchmark_matrix": matrix}, indent=2, sort_keys=True))


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def acceptance_query_sql(method: str, shape: str, dimensions: int) -> str | None:
    method = normalize_method(method)
    vector = vector_literal(dimensions)
    text_query = "term1" if "rare" in shape else "common"
    filter_clause = ""
    if shape == "hybrid_with_filter_1_percent":
        filter_clause = "bucket = 0"
        text_query = "common"
    elif shape == "hybrid_with_filter_10_percent":
        filter_clause = "bucket < 10"
        text_query = "common"

    where = f"WHERE {filter_clause}" if filter_clause else ""
    if method == "turbohybrid":
        if shape == "dense_only":
            return f"""
SELECT id
FROM turbohybrid_suite_docs
{where}
ORDER BY embedding <~> hybrid_query(vector_query => {vector}, dense_k => 128, final_k => 10)
LIMIT 10
"""
        if shape.startswith("bm25_only"):
            return f"""
SELECT id
FROM turbohybrid_suite_docs
{where}
ORDER BY embedding <~> hybrid_query(
    text_query => to_tsquery('english', '{text_query}'),
    dense_k => 0,
    bm25_k => 128,
    final_k => 10
)
LIMIT 10
"""
        fusion = "weighted" if shape == "hybrid_weighted" else "rrf"
        return f"""
SELECT id
FROM turbohybrid_suite_docs
{where}
ORDER BY embedding <~> hybrid_query(
    vector_query => {vector},
    text_query => to_tsquery('english', '{text_query}'),
    fusion => '{fusion}',
    dense_k => 128,
    bm25_k => 128,
    final_k => 10
)
LIMIT 10
"""

    if method == "turboquant_dense":
        if shape != "dense_only":
            return None
        return query_sql(method, dimensions, filter_clause)

    if method == "postgres_sql_rrf":
        if shape == "dense_only":
            return f"""
SELECT id
FROM turbohybrid_suite_docs
{where}
ORDER BY embedding <=> {vector}
LIMIT 10
"""
        if shape.startswith("bm25_only"):
            return f"""
SELECT id
FROM turbohybrid_suite_docs
{where} {"AND" if where else "WHERE"} body_tsv @@ to_tsquery('english', '{text_query}')
ORDER BY ts_rank_cd(body_tsv, to_tsquery('english', '{text_query}')) DESC, id
LIMIT 10
"""
        return query_sql(method, dimensions, filter_clause)

    return None


def capture_turbohybrid_stats(database: str, query: str) -> dict[str, Any]:
    sql = f"""
{query};
SELECT hybrid_last_scan_stats()::text;
SELECT tq_index_stats('turbohybrid_suite_hybrid_idx'::regclass)::text;
"""
    lines = [line for line in run_psql(database, sql).splitlines() if line]
    if len(lines) < 2:
        return {}
    return {
        "result_count": len(lines) - 2,
        "hybrid_last_scan_stats": json.loads(lines[-2]),
        "tq_index_stats": json.loads(lines[-1]),
    }


def capture_bm25_debug_stats(database: str) -> dict[str, Any]:
    sql = """
WITH idx AS (
    SELECT to_regclass('turbohybrid_suite_hybrid_idx') AS oid
)
SELECT CASE
    WHEN oid IS NULL THEN '{}'
    ELSE tq_debug_bm25_stats(oid)::text
END
FROM idx;
"""
    return json.loads(run_psql(database, sql) or "{}")


def acceptance_gucs(database: str) -> dict[str, str]:
    settings = [
        "shared_buffers",
        "work_mem",
        "maintenance_work_mem",
        "effective_cache_size",
        "enable_seqscan",
        "hybrid.fusion_hash_threshold",
        "hybrid.bm25_cache_max_mb",
        "hybrid.enable_wand",
    ]
    values: dict[str, str] = {}
    for setting in settings:
        try:
            prefix = "LOAD 'vector';\n" if setting.startswith("hybrid.") else ""
            values[setting] = run_psql(database, f"{prefix}SHOW {setting}")
        except subprocess.CalledProcessError:
            values[setting] = "unavailable"
    values["benchmark_query_prefix"] = "SET enable_seqscan = off"
    return values


def acceptance_insert_delta(database: str, dimensions: int, base_rows: int, percent: int) -> dict[str, Any]:
    target_rows = max(1, base_rows * percent // 100)
    current_max = int(run_psql(database, "SELECT max(id) FROM turbohybrid_suite_docs"))
    before = wal_lsn(database)
    sql = f"""
INSERT INTO turbohybrid_suite_docs
SELECT
    i,
    i % 100,
    ('[' || array_to_string(ARRAY(
        SELECT (((i * (d + 29)) % 1000)::float8 / 1000.0)::text
        FROM generate_series(1, {dimensions}) d
    ), ',') || ']')::vector({dimensions}),
    'common delta term1 topic1',
    to_tsvector('english', 'common delta term1 topic1')
FROM generate_series({current_max + 1}, {current_max + target_rows}) i;
"""
    elapsed_ms = timed_sql(database, sql)
    after = wal_lsn(database)
    return {
        "inserted_rows": target_rows,
        "insert_ms": round(elapsed_ms, 3),
        "wal_bytes": wal_diff(database, before, after),
        "bm25_stats_after_insert": capture_bm25_debug_stats(database),
    }


def acceptance_correctness_probes(database: str, dimensions: int, validation: str) -> dict[str, Any]:
    if validation == "none":
        return {
            "validation": validation,
            "skipped": True,
            "wand_matches_daat": None,
            "hybrid_topk_deterministic": None,
        }

    vector = vector_literal(dimensions)
    deterministic_query = "common | term1" if validation in {"standard", "full"} else "term1 | topic1"
    if validation in {"standard", "full"}:
        wand_queries = """
                (to_tsquery('english', 'common'), 10),
                (to_tsquery('english', 'common | term1 | term2'), 10),
                (to_tsquery('english', 'term1 & topic1'), 10)
"""
    else:
        wand_queries = """
                (to_tsquery('english', 'term1 & topic1'), 10)
"""
    sql = f"""
DROP TABLE IF EXISTS turbohybrid_acceptance_first;
DROP TABLE IF EXISTS turbohybrid_acceptance_second;

CREATE TEMP TABLE turbohybrid_acceptance_first AS
SELECT id
FROM turbohybrid_suite_docs
ORDER BY embedding <~> hybrid_query(
    vector_query => {vector},
    text_query => to_tsquery('english', '{deterministic_query}'),
    dense_k => 128,
    bm25_k => 128,
    final_k => 10
)
LIMIT 10;

CREATE TEMP TABLE turbohybrid_acceptance_second AS
SELECT id
FROM turbohybrid_suite_docs
ORDER BY embedding <~> hybrid_query(
    vector_query => {vector},
    text_query => to_tsquery('english', '{deterministic_query}'),
    dense_k => 128,
    bm25_k => 128,
    final_k => 10
)
LIMIT 10;

SELECT
    (
        WITH wand_queries(tsq, k) AS (
            VALUES
{wand_queries}
        )
        SELECT bool_and(
            (tq_debug_bm25_topk('turbohybrid_suite_hybrid_idx'::regclass, tsq, k, true)->'results') =
            (tq_debug_bm25_topk('turbohybrid_suite_hybrid_idx'::regclass, tsq, k, false)->'results')
        )
        FROM wand_queries
    ),
    (SELECT array_agg(id ORDER BY ctid) FROM turbohybrid_acceptance_first) =
    (SELECT array_agg(id ORDER BY ctid) FROM turbohybrid_acceptance_second);
"""
    values = run_psql(database, "SET enable_seqscan = off;\n" + sql).split("|")
    return {
        "validation": validation,
        "skipped": False,
        "wand_matches_daat": values[0] == "t" if len(values) > 0 else False,
        "hybrid_topk_deterministic": values[1] == "t" if len(values) > 1 else False,
    }


def acceptance_evaluate(summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    thresholds = summary.get("thresholds", DEFAULT_ACCEPTANCE_THRESHOLDS)
    by_key = {
        (row["method"], row["shape"], row["concurrency"]): row
        for row in summary["results"]
    }

    def add(name: str, status: str, value: Any, threshold: Any, detail: str) -> None:
        checks.append({
            "name": name,
            "status": status,
            "value": value,
            "threshold": threshold,
            "detail": detail,
        })

    correctness = summary.get("correctness", {})
    validation = correctness.get("validation", summary.get("validation", "standard"))
    if correctness.get("skipped"):
        add(
            "validation_probes",
            "PASS",
            "skipped",
            "validation != none",
            "correctness probes were skipped by --validation none",
        )
    else:
        add(
            "wand_matches_daat",
            "PASS" if correctness.get("wand_matches_daat") else "FAIL",
            correctness.get("wand_matches_daat"),
            True,
            f"{validation} validation queries return identical BM25 results with WAND and DAAT",
        )
        add(
            "hybrid_topk_deterministic",
            "PASS" if correctness.get("hybrid_topk_deterministic") else "FAIL",
            correctness.get("hybrid_topk_deterministic"),
            True,
            f"{validation} validation query returns the same ordered top-k twice",
        )

    if not summary.get("latency_thresholds_enforced", False):
        add(
            "latency_thresholds",
            "PASS",
            "skipped",
            "full profile or full-equivalent explicit run",
            "smoke/dev profiles validate behavior and emit timings; full profile or explicit 100k-row/30-run/concurrency 1,4,16 runs enforce latency thresholds",
        )
    else:
        dense = by_key.get(("turboquant_dense", "dense_only", 1))
        hybrid = by_key.get(("turbohybrid", "hybrid_rrf_common", 1))
        if dense and hybrid and dense["hot"]["p95_ms"] > 0:
            ratio = hybrid["hot"]["p95_ms"] / dense["hot"]["p95_ms"]
            add(
                "hybrid_rrf_common_vs_dense_p95",
                "PASS" if ratio <= thresholds["hybrid_rrf_common_vs_dense_p95_ratio"] else "WARN",
                round(ratio, 3),
                thresholds["hybrid_rrf_common_vs_dense_p95_ratio"],
                "turbohybrid hot p95 divided by turboquant dense-only hot p95",
            )
        else:
            add("hybrid_rrf_common_vs_dense_p95", "WARN", None, None, "missing comparable dense or hybrid result")

        rare = by_key.get(("turbohybrid", "bm25_only_rare", 1))
        if rare:
            p95 = rare["hot"]["p95_ms"]
            add(
                "bm25_only_rare_p95",
                "PASS" if p95 <= thresholds["bm25_only_rare_p95_ms"] else "WARN",
                p95,
                thresholds["bm25_only_rare_p95_ms"],
                "turbohybrid rare-term BM25 hot p95",
            )

        for shape in ("hybrid_rrf_common", "hybrid_weighted"):
            row = by_key.get(("turbohybrid", shape, 1))
            stats = (row or {}).get("scan_stats", {}).get("hybrid_last_scan_stats", {})
            elapsed = stats.get("elapsed_us", 0)
            fusion = stats.get("fusion_elapsed_us", 0)
            if elapsed:
                pct = fusion * 100.0 / elapsed
                add(
                    f"{shape}_fusion_elapsed_pct",
                    "PASS" if pct <= thresholds["fusion_elapsed_pct_final_k_50"] else "WARN",
                    round(pct, 3),
                    thresholds["fusion_elapsed_pct_final_k_50"],
                    "fusion time share for final_k <= 50",
                )

        delta0 = by_key.get(("turbohybrid", "delta_0_percent", 1))
        delta10 = by_key.get(("turbohybrid", "delta_10_percent", 1))
        post = by_key.get(("turbohybrid", "post_compaction", 1))
        if delta0 and delta10 and delta0["hot"]["p95_ms"] > 0:
            ratio = delta10["hot"]["p95_ms"] / delta0["hot"]["p95_ms"]
            add(
                "delta_10_vs_delta_0_p95",
                "PASS" if ratio <= thresholds["delta_10_vs_delta_0_p95_ratio"] else "WARN",
                round(ratio, 3),
                thresholds["delta_10_vs_delta_0_p95_ratio"],
                "delta 10 percent p95 divided by delta 0 percent p95",
            )
        if delta0 and post and delta0["hot"]["p95_ms"] > 0:
            ratio = post["hot"]["p95_ms"] / delta0["hot"]["p95_ms"]
            add(
                "post_compaction_vs_delta_0_p95",
                "PASS" if ratio <= thresholds["post_compaction_vs_delta_0_p95_ratio"] else "WARN",
                round(ratio, 3),
                thresholds["post_compaction_vs_delta_0_p95_ratio"],
                "post-compaction p95 divided by delta 0 percent p95",
            )

    cache_rows = [
        row for row in summary["results"]
        if row["method"] == "turbohybrid"
        and row.get("scan_stats", {}).get("hybrid_last_scan_stats", {}).get("bm25_cache_bytes", 0) > 0
    ]
    add(
        "bm25_cache_bytes_reported",
        "PASS" if cache_rows else "WARN",
        len(cache_rows),
        ">=1",
        "at least one turbohybrid scan reported backend-local BM25 cache bytes",
    )
    return checks

def write_acceptance_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# TurboHybrid Acceptance Benchmark",
        "",
        f"- Commit: `{summary['commit']}`",
        f"- PostgreSQL: `{summary['postgres_version']}`",
        f"- Host: `{summary['host']}`",
        f"- RAM bytes: `{summary.get('host_memory_bytes', 'unavailable')}`",
        f"- Profile: `{summary.get('profile', 'unknown')}`",
        f"- Rows: {summary['rows']}",
        f"- Dimensions: {summary['dimensions']}",
        f"- Runs: {summary['runs']}",
        f"- Warmup: {summary.get('warmup', '')}",
        f"- Concurrency: {summary.get('concurrency', [])}",
        f"- Validation: `{summary.get('validation', 'standard')}`",
        f"- Latency thresholds enforced: {summary.get('latency_thresholds_enforced', False)}",
        f"- Shapes: {', '.join(summary.get('shapes', []))}",
        f"- Generated at Unix time: {summary['generated_at_unix']}",
        "",
        "## Thresholds",
        "",
        "| Check | Status | Value | Threshold | Detail |",
        "|---|---:|---:|---:|---|",
    ]
    for check in summary["checks"]:
        lines.append(
            f"| {check['name']} | {check['status']} | {check['value']} | {check['threshold']} | {check['detail']} |"
        )
    lines.extend([
        "",
        "## Correctness",
        "",
        f"- Validation: {summary.get('validation', 'standard')}",
        f"- Skipped: {summary.get('correctness', {}).get('skipped', False)}",
        f"- WAND matches DAAT: {summary.get('correctness', {}).get('wand_matches_daat')}",
        f"- Hybrid top-k deterministic: {summary.get('correctness', {}).get('hybrid_topk_deterministic')}",
    ])
    lines.extend([
        "",
        "## Latency",
        "",
        "| Method | Shape | Concurrency | p50 ms | p95 ms | p99 ms | Mean ms | QPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["results"]:
        hot = row["hot"]
        lines.append(
            f"| {row['method']} | {row['shape']} | {row['concurrency']} | "
            f"{hot['p50_ms']} | {hot['p95_ms']} | {hot['p99_ms']} | {hot['mean_ms']} | {hot['qps']} |"
        )
    lines.extend([
        "",
        "## Build And Storage",
        "",
        "| Method | Build ms | Build WAL bytes | Index bytes |",
        "|---|---:|---:|---:|",
    ])
    for row in summary["builds"]:
        lines.append(
            f"| {row['method']} | {row['build_ms']} | {row['build_wal_bytes']} | {row['index_bytes']} |"
        )
    lines.extend([
        "",
        "## GUCs",
        "",
        "| Setting | Value |",
        "|---|---|",
    ])
    for name, value in summary.get("gucs", {}).items():
        lines.append(f"| `{name}` | `{value}` |")
    lines.extend([
        "",
        "## DDL",
        "",
    ])
    for method, ddl in summary.get("ddl", {}).items():
        lines.extend([
            f"### {method}",
            "",
            "```sql",
            ddl.strip(),
            "```",
            "",
        ])
    lines.extend([
        "## Scan Stats",
        "",
        "| Shape | Elapsed us | Dense us | BM25 us | Fusion us | Strategy | Union | Final | Postings decoded | Blocks visited | Blocks skipped | Cache bytes |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["results"]:
        if row["method"] != "turbohybrid" or row["concurrency"] != 1:
            continue
        stats = row.get("scan_stats", {}).get("hybrid_last_scan_stats", {})
        if not stats:
            continue
        lines.append(
            f"| {row['shape']} | {stats.get('elapsed_us', '')} | {stats.get('dense_elapsed_us', '')} | "
            f"{stats.get('bm25_elapsed_us', '')} | {stats.get('fusion_elapsed_us', '')} | "
            f"{stats.get('fusion_strategy', '')} | {stats.get('union_candidates', '')} | "
            f"{stats.get('final_results', '')} | {stats.get('bm25_postings_decoded', '')} | "
            f"{stats.get('bm25_blocks_visited', '')} | {stats.get('bm25_blocks_skipped', '')} | "
            f"{stats.get('bm25_cache_bytes', '')} |"
        )
    lines.extend([
        "",
        "## Delta And Compaction",
        "",
        "| Shape | Inserted rows | Insert ms | Vacuum/compaction ms | WAL bytes | Delta docs before | Postings pages after | BlockMax pages after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary.get("mutations", []):
        lines.append(
            f"| {row['shape']} | {row.get('inserted_rows', '')} | {row.get('insert_ms', '')} | "
            f"{row.get('compaction_ms', row.get('vacuum_ms', ''))} | {row.get('wal_bytes', '')} | "
            f"{row.get('delta_doc_count_before', '')} | {row.get('postings_pages_after', '')} | "
            f"{row.get('blockmax_pages_after', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_acceptance_one(args: argparse.Namespace, rows: int, output_suffix: str = "") -> dict[str, Any]:
    concurrencies = acceptance_concurrencies(args)
    methods = acceptance_methods(args)
    shapes = acceptance_shapes(args)
    dimensions = acceptance_dimensions(args)
    runs = acceptance_runs(args)
    warmup = acceptance_warmup(args)
    validation = acceptance_validation(args)
    thresholds = load_acceptance_thresholds(args.thresholds)
    enforce_latency_thresholds = acceptance_enforce_latency_thresholds(
        rows, runs, concurrencies, args.profile
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_psql_file(args.database, synthetic_setup_sql(rows, dimensions))
    run_psql_file(args.database, drop_method_indexes_sql())

    builds: list[dict[str, Any]] = []
    build_by_method: dict[str, dict[str, Any]] = {}
    ddl_by_method: dict[str, str] = {}
    index_map: dict[str, list[str]] = {}
    for method in methods:
        build_sql, indexes = build_method_sql(method)
        ddl_by_method[method] = build_sql
        before = wal_lsn(args.database)
        build_ms = timed_sql(args.database, build_sql)
        after = wal_lsn(args.database)
        index_map[method] = indexes
        build_row = {
            "method": method,
            "build_ms": round(build_ms, 3),
            "build_wal_bytes": wal_diff(args.database, before, after),
            "index_bytes": relation_bytes(args.database, indexes),
        }
        builds.append(build_row)
        build_by_method[method] = build_row
    run_psql(args.database, "ANALYZE turbohybrid_suite_docs")

    results: list[dict[str, Any]] = []
    mutation_events: list[dict[str, Any]] = []
    for shape in shapes:
        if shape == "delta_1_percent":
            mutation_events.append({"shape": shape, **acceptance_insert_delta(args.database, dimensions, rows, 1)})
        elif shape == "delta_5_percent":
            mutation_events.append({"shape": shape, **acceptance_insert_delta(args.database, dimensions, rows, 4)})
        elif shape == "delta_10_percent":
            mutation_events.append({"shape": shape, **acceptance_insert_delta(args.database, dimensions, rows, 5)})
        elif shape == "post_compaction":
            bm25_before = capture_bm25_debug_stats(args.database)
            before = wal_lsn(args.database)
            vacuum_ms = timed_sql(args.database, "VACUUM turbohybrid_suite_docs;")
            after = wal_lsn(args.database)
            bm25_after = capture_bm25_debug_stats(args.database)
            mutation_events.append({
                "shape": shape,
                "vacuum_ms": round(vacuum_ms, 3),
                "compaction_ms": round(vacuum_ms, 3),
                "wal_bytes": wal_diff(args.database, before, after),
                "delta_doc_count_before": bm25_before.get("delta_doc_count"),
                "postings_pages_after": bm25_after.get("postings_pages"),
                "blockmax_pages_after": bm25_after.get("blockmax_pages"),
                "bm25_stats_before_compaction": bm25_before,
                "bm25_stats_after_compaction": bm25_after,
            })

        for method in methods:
            sql = acceptance_query_sql(method, shape, dimensions)
            if sql is None:
                continue
            sql = "SET enable_seqscan = off;\n" + sql
            for _ in range(warmup):
                run_psql(args.database, sql)
            scan_stats = capture_turbohybrid_stats(args.database, sql) if method == "turbohybrid" else {}
            for concurrency in concurrencies:
                hot = run_query_batch(args.database, sql, runs, concurrency)
                method_build = build_by_method.get(method, {})
                row_scan_stats = scan_stats if concurrency == 1 else {}
                index_size_bytes = relation_bytes(args.database, index_map.get(method, []))
                results.append({
                    "method": method,
                    "shape": shape,
                    "query_shape": shape,
                    "rows": rows,
                    "dimensions": dimensions,
                    "run_number": "aggregate",
                    "concurrency": concurrency,
                    "hot": hot,
                    "p50_ms": hot["p50_ms"],
                    "p95_ms": hot["p95_ms"],
                    "p99_ms": hot["p99_ms"],
                    "mean_ms": hot["mean_ms"],
                    "scan_stats": row_scan_stats,
                    "hybrid_last_scan_stats": row_scan_stats.get("hybrid_last_scan_stats", {}),
                    "tq_index_stats": row_scan_stats.get("tq_index_stats", {}),
                    "index_bytes": index_size_bytes,
                    "index_size_bytes": index_size_bytes,
                    "wal_bytes_if_available": method_build.get("build_wal_bytes"),
                })

    summary: dict[str, Any] = {
        "suite": "turbohybrid_acceptance",
        "generated_at_unix": int(time.time()),
        "commit": git_sha(),
        "database": args.database,
        "postgres_version": run_psql(args.database, "SHOW server_version"),
        "host": f"{platform.platform()} / {platform.processor()}",
        "host_memory_bytes": host_memory(),
        "profile": args.profile,
        "shapes": shapes,
        "rows": rows,
        "dimensions": dimensions,
        "runs": runs,
        "warmup": warmup,
        "concurrency": concurrencies,
        "validation": validation,
        "latency_thresholds_enforced": enforce_latency_thresholds,
        "ddl": ddl_by_method,
        "gucs": acceptance_gucs(args.database),
        "thresholds": thresholds,
        "builds": builds,
        "mutations": mutation_events,
        "correctness": acceptance_correctness_probes(args.database, dimensions, validation),
        "results": results,
    }
    summary["checks"] = acceptance_evaluate(summary)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{timestamp}{output_suffix}_summary.json"
    md_path = output_dir / f"{timestamp}{output_suffix}_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_acceptance_markdown(summary, md_path)
    return {"json": str(json_path), "markdown": str(md_path), "checks": summary["checks"], "rows": rows}


def run_acceptance(args: argparse.Namespace) -> None:
    rows_values = acceptance_rows(args)
    outputs = []
    multi = len(rows_values) > 1
    for rows in rows_values:
        suffix = f"_rows{rows}" if multi else ""
        outputs.append(run_acceptance_one(args, rows, suffix))

    print(json.dumps(outputs[0] if len(outputs) == 1 else {"runs": outputs}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    list_parser = sub.add_parser("list", help="List configured datasets and methods")
    list_parser.set_defaults(func=list_suite)

    plan_parser = sub.add_parser("plan", help="Emit the benchmark matrix as JSON")
    plan_parser.set_defaults(func=plan)

    system_parser = sub.add_parser("run-system-synthetic", help="Run PostgreSQL synthetic systems benchmark")
    system_parser.add_argument("--database", default="contrib_regression")
    system_parser.add_argument("--rows", type=int, default=10000)
    system_parser.add_argument("--dimensions", type=int, default=64)
    system_parser.add_argument("--methods", default="turbohybrid,postgres_sql_rrf")
    system_parser.add_argument("--warmup", type=int, default=3)
    system_parser.add_argument("--runs", type=int, default=30)
    system_parser.add_argument("--cold-runs", type=int, default=3)
    system_parser.add_argument("--cold-command", default="")
    system_parser.add_argument("--concurrency", type=int, default=1)
    system_parser.add_argument("--output", default="")
    system_parser.set_defaults(func=run_system_synthetic)

    real_rag_parser = sub.add_parser(
        "run-real-rag",
        help="Run a real BEIR-style RAG retrieval benchmark with text embeddings",
    )
    real_rag_parser.add_argument("--database", default="contrib_regression")
    real_rag_parser.add_argument("--dataset", default=".cache/beir/fiqa")
    real_rag_parser.add_argument("--dataset-name", default="fiqa-openai")
    real_rag_parser.add_argument("--split", default="test")
    real_rag_parser.add_argument(
        "--methods",
        default="postgres_sql_rrf,turbohybrid,turbohybrid_exact_storage_off",
    )
    real_rag_parser.add_argument("--max-docs", type=int, default=0,
                                 help="Optional corpus subset size; 0 uses all documents")
    real_rag_parser.add_argument("--max-queries", type=int, default=0,
                                 help="Optional query subset size; 0 uses all qrel queries")
    real_rag_parser.add_argument("--dense-k", type=int, default=400)
    real_rag_parser.add_argument("--bm25-k", type=int, default=400)
    real_rag_parser.add_argument("--rrf-k", type=int, default=60)
    real_rag_parser.add_argument("--final-k", type=int, default=10)
    real_rag_parser.add_argument("--warmup", type=int, default=1)
    real_rag_parser.add_argument("--runs", type=int, default=1)
    real_rag_parser.add_argument("--output-json", default="")
    real_rag_parser.add_argument("--output-md", default="")
    real_rag_parser.set_defaults(func=run_real_rag)

    simd_profile_parser = sub.add_parser("run-simd-profile", help="Run architecture-aware TurboHybrid SIMD profile benchmark")
    simd_profile_parser.add_argument("--database", default="contrib_regression")
    simd_profile_parser.add_argument("--rows", type=int, default=100000)
    simd_profile_parser.add_argument("--dimensions", type=int, default=768)
    simd_profile_parser.add_argument("--runs", type=int, default=50)
    simd_profile_parser.add_argument("--warmup-runs", type=int, default=10)
    simd_profile_parser.add_argument("--dense-k", type=int, default=100)
    simd_profile_parser.add_argument("--bm25-k", type=int, default=100)
    simd_profile_parser.add_argument("--final-k", type=int, default=20)
    simd_profile_parser.add_argument("--method", choices=["turbohybrid"], default="turbohybrid")
    simd_profile_parser.add_argument("--output", default="")
    simd_profile_parser.add_argument("--perf-command", default="")
    simd_profile_parser.add_argument("--include-delta", action="store_true",
                                     help="Insert a 10 percent delta segment before profiling delta-heavy queries")
    simd_profile_parser.add_argument("--set", action="append", default=[],
                                     help="Session GUC assignment, for example hnsw.tq_simd_force=scalar")
    simd_profile_parser.set_defaults(func=run_simd_profile)

    bm25_decode_parser = sub.add_parser("bench-bm25-decode", help="Run a focused BM25 postings decode benchmark")
    bm25_decode_parser.add_argument("--database", default="contrib_regression")
    bm25_decode_parser.add_argument("--rows", type=int, default=100000)
    bm25_decode_parser.add_argument("--dimensions", type=int, default=768)
    bm25_decode_parser.add_argument("--encoding", choices=["auto", "offset16"], default="auto")
    bm25_decode_parser.add_argument("--runs", type=int, default=30)
    bm25_decode_parser.add_argument("--bm25-k", type=int, default=100)
    bm25_decode_parser.add_argument("--output", default="")
    bm25_decode_parser.add_argument("--set", action="append", default=[],
                                    help="Session GUC assignment, for example hybrid.bm25_simd_force=scalar")
    bm25_decode_parser.set_defaults(func=run_bm25_decode_bench)

    bm25_score_parser = sub.add_parser("bench-bm25-score", help="Run a focused BM25 decode plus score benchmark")
    bm25_score_parser.add_argument("--database", default="contrib_regression")
    bm25_score_parser.add_argument("--rows", type=int, default=100000)
    bm25_score_parser.add_argument("--dimensions", type=int, default=768)
    bm25_score_parser.add_argument("--query-shape", default="common-term",
                                   choices=["common-term", "rare-term", "two-term-or", "two-term-and", "five-term-or", "no-match"])
    bm25_score_parser.add_argument("--bm25-k", type=int, default=100)
    bm25_score_parser.add_argument("--precompute-tf-norm", default="on",
                                   choices=["on", "off", "true", "false", "1", "0", "yes", "no"])
    bm25_score_parser.add_argument("--runs", type=int, default=30)
    bm25_score_parser.add_argument("--output", default="")
    bm25_score_parser.add_argument("--set", action="append", default=[],
                                   help="Session GUC assignment, for example hybrid.bm25_simd_force=auto")
    bm25_score_parser.set_defaults(func=run_bm25_score_bench)

    exact_parser = sub.add_parser("bench-exact-rescore", help="Run a focused exact-rescore SIMD benchmark")
    exact_parser.add_argument("--database", default="contrib_regression")
    exact_parser.add_argument("--rows", type=int, default=100000)
    exact_parser.add_argument("--dimensions", type=int, default=1536)
    exact_parser.add_argument("--metric", choices=["cosine", "l2", "ip"], default="cosine")
    exact_parser.add_argument("--rescore-count", type=int, default=400)
    exact_parser.add_argument("--runs", type=int, default=30)
    exact_parser.add_argument("--warmup-runs", type=int, default=5)
    exact_parser.add_argument("--output", default="")
    exact_parser.add_argument("--set", action="append", default=[],
                              help="Session GUC assignment, for example hnsw.tq_exact_simd_force=scalar")
    exact_parser.set_defaults(func=run_exact_rescore_bench)

    weighted_parser = sub.add_parser("bench-weighted-tq", help="Run a focused weighted TurboQuant+ dense benchmark")
    weighted_parser.add_argument("--database", default="contrib_regression")
    weighted_parser.add_argument("--rows", type=int, default=100000)
    weighted_parser.add_argument("--dimensions", type=int, default=1536)
    weighted_parser.add_argument("--tq-bits", type=int, default=4)
    weighted_parser.add_argument("--weighted", default="on", choices=["on", "off", "true", "false", "1", "0", "yes", "no"])
    weighted_parser.add_argument("--runs", type=int, default=30)
    weighted_parser.add_argument("--warmup-runs", type=int, default=5)
    weighted_parser.add_argument("--output", default="")
    weighted_parser.add_argument("--set", action="append", default=[],
                                 help="Session GUC assignment, for example hnsw.tq_graph_avx512_weighted=off")
    weighted_parser.set_defaults(func=run_weighted_tq_bench)

    dense_batch_parser = sub.add_parser("bench-dense-batch", help="Run a focused dense graph batch-scoring benchmark")
    dense_batch_parser.add_argument("--database", default="contrib_regression")
    dense_batch_parser.add_argument("--rows", type=int, default=100000)
    dense_batch_parser.add_argument("--dimensions", type=int, default=1536)
    dense_batch_parser.add_argument("--dense-k", type=int, default=400)
    dense_batch_parser.add_argument("--tq-bits", type=int, default=4)
    dense_batch_parser.add_argument("--runs", type=int, default=30)
    dense_batch_parser.add_argument("--warmup-runs", type=int, default=5)
    dense_batch_parser.add_argument("--output", default="")
    dense_batch_parser.add_argument("--set", action="append", default=[],
                                    help="Session GUC assignment, for example hnsw.tq_graph_batch_scoring=off")
    dense_batch_parser.set_defaults(func=run_dense_batch_bench)

    simd_matrix_parser = sub.add_parser("run-simd-matrix", help="Run scalar-vs-auto SIMD profiles across rows, dimensions, and candidate budgets")
    simd_matrix_parser.add_argument("--database", default="contrib_regression")
    simd_matrix_parser.add_argument("--rows", default="10000,100000")
    simd_matrix_parser.add_argument("--dimensions", default="128,768")
    simd_matrix_parser.add_argument("--candidate-budgets", default="50,100,200")
    simd_matrix_parser.add_argument("--runs", type=int, default=10)
    simd_matrix_parser.add_argument("--warmup-runs", type=int, default=3)
    simd_matrix_parser.add_argument("--final-k", type=int, default=20)
    simd_matrix_parser.add_argument("--output-json", default="")
    simd_matrix_parser.add_argument("--output-md", default="")
    simd_matrix_parser.add_argument("--perf", default="", help="Optional perf wrapper command")
    simd_matrix_parser.set_defaults(func=run_simd_matrix)

    acceptance_parser = sub.add_parser("acceptance", help="Run the TurboHybrid beta acceptance benchmark gate")
    acceptance_parser.add_argument("--database", default="contrib_regression")
    acceptance_parser.add_argument("--profile", choices=sorted(ACCEPTANCE_PROFILES), default="dev",
                                   help="Validation tier. smoke is fastest, dev is the default iteration gate, full is publishable.")
    acceptance_parser.add_argument("--rows", type=int, default=None,
                                   help="Override the selected profile's dataset size")
    acceptance_parser.add_argument("--rows-list", default="", help="Comma-separated dataset sizes, for example 10000,100000")
    acceptance_parser.add_argument("--dimensions", type=int, default=None,
                                   help="Override the selected profile's vector dimensions")
    acceptance_parser.add_argument("--large", action="store_true")
    acceptance_parser.add_argument("--methods", default="",
                                   help="Override the selected profile's method list")
    acceptance_parser.add_argument("--shapes", default="",
                                   help="Override the selected profile's query/mutation shapes")
    acceptance_parser.add_argument("--warmup", type=int, default=None)
    acceptance_parser.add_argument("--runs", type=int, default=None)
    acceptance_parser.add_argument("--concurrency", default="",
                                   help="Override the selected profile's concurrency list")
    acceptance_parser.add_argument("--validation", choices=sorted(ACCEPTANCE_VALIDATION_LEVELS), default="",
                                   help="Correctness probe tier. Profile defaults keep smoke/dev fast; standard/full include high-df DAAT/WAND probes.")
    acceptance_parser.add_argument("--output-dir", default="benchmarks/turbohybrid/results")
    acceptance_parser.add_argument("--thresholds", default="", help="JSON file overriding acceptance thresholds")
    acceptance_parser.set_defaults(func=run_acceptance)

    score_parser = sub.add_parser("score-run", help="Score a TREC run against qrels")
    score_parser.add_argument("--qrels", required=True)
    score_parser.add_argument("--run", required=True)
    score_parser.add_argument("--k", default="10,100")
    score_parser.set_defaults(func=score_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
