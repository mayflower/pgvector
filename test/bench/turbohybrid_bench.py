#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import time


def run_psql(database: str, sql: str) -> str:
    proc = subprocess.run(
        ["psql", "-X", "-q", "-d", database, "-At", "-c", sql],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def timed(database: str, sql: str, runs: int) -> tuple[float, float, float, float]:
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        run_psql(database, sql)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    p50 = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
    p99 = samples[min(len(samples) - 1, int(len(samples) * 0.99))]
    qps = 1000.0 / statistics.mean(samples)
    return p50, p95, p99, qps


def timed_result(database: str, sql: str, runs: int) -> dict[str, float]:
    p50, p95, p99, qps = timed(database, sql, runs)
    return {
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "qps_serial": round(qps, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="contrib_regression")
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--skip-vacuum-stress", action="store_true")
    args = parser.parse_args()

    setup_sql = open("test/bench/turbohybrid_bench.sql", encoding="utf-8").read()
    setup_sql = setup_sql.replace("\\set rows 1000", f"\\set rows {args.rows}")
    start = time.perf_counter()
    subprocess.run(
        ["psql", "-X", "-q", "-d", args.database],
        input=setup_sql,
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    build_ms = (time.perf_counter() - start) * 1000.0

    vacuum_ms = 0.0
    if not args.skip_vacuum_stress:
        start = time.perf_counter()
        run_psql(args.database, "DELETE FROM turbohybrid_bench_docs WHERE id % 97 = 0")
        run_psql(args.database, "VACUUM turbohybrid_bench_docs")
        vacuum_ms = (time.perf_counter() - start) * 1000.0

    vector = "'[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.2,0.3,0.4,0.5,0.6,0.7]'::vector"
    dense_sql = f"SELECT id FROM turbohybrid_bench_docs ORDER BY embedding <~> hybrid_query(vector_query => {vector}, dense_k => 64) LIMIT 10"
    bm25_sql = "SELECT id FROM turbohybrid_bench_docs ORDER BY embedding <~> hybrid_query(text_query => to_tsquery('english', 'term1 | common'), dense_k => 0, bm25_k => 64) LIMIT 10"
    bm25_rare_sql = "SELECT id FROM turbohybrid_bench_docs ORDER BY embedding <~> hybrid_query(text_query => to_tsquery('english', 'raremarker'), dense_k => 0, bm25_k => 64) LIMIT 10"
    hybrid_sql = f"SELECT id FROM turbohybrid_bench_docs ORDER BY embedding <~> hybrid_query(vector_query => {vector}, text_query => to_tsquery('english', 'term1 | common'), dense_k => 64, bm25_k => 64) LIMIT 10"
    hybrid_weighted_sql = f"SELECT id FROM turbohybrid_bench_docs ORDER BY embedding <~> hybrid_query(vector_query => {vector}, text_query => to_tsquery('english', 'term1 | common'), dense_k => 64, bm25_k => 64, fusion => 'weighted') LIMIT 10"
    hybrid_three_term_sql = f"SELECT id FROM turbohybrid_bench_docs ORDER BY embedding <~> hybrid_query(vector_query => {vector}, text_query => to_tsquery('english', 'term1 | term2 | common'), dense_k => 64, bm25_k => 64) LIMIT 10"
    hybrid_and_sql = f"SELECT id FROM turbohybrid_bench_docs ORDER BY embedding <~> hybrid_query(vector_query => {vector}, text_query => to_tsquery('english', 'term1 & common'), dense_k => 64, bm25_k => 64) LIMIT 10"

    dense_p50, dense_p95, dense_p99, dense_qps = timed(args.database, dense_sql, args.runs)
    bm25_p50, bm25_p95, bm25_p99, bm25_qps = timed(args.database, bm25_sql, args.runs)
    bm25_rare_p50, bm25_rare_p95, bm25_rare_p99, bm25_rare_qps = timed(args.database, bm25_rare_sql, args.runs)
    hybrid_p50, hybrid_p95, hybrid_p99, hybrid_qps = timed(args.database, hybrid_sql, args.runs)
    hybrid_weighted = timed_result(args.database, hybrid_weighted_sql, args.runs)
    hybrid_three_p50, hybrid_three_p95, hybrid_three_p99, hybrid_three_qps = timed(args.database, hybrid_three_term_sql, args.runs)
    hybrid_and = timed_result(args.database, hybrid_and_sql, args.runs)
    stats_output = run_psql(args.database, hybrid_sql + "; SELECT hybrid_last_scan_stats()")
    stats = json.loads(stats_output.splitlines()[-1])
    rare_stats_output = run_psql(args.database, bm25_rare_sql + "; SELECT hybrid_last_scan_stats()")
    rare_stats = json.loads(rare_stats_output.splitlines()[-1])
    run_psql(
        args.database,
        "INSERT INTO turbohybrid_bench_docs (id, tenant_id, embedding, body_tsv) "
        "VALUES (2000000001, 0, '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.2,0.3,0.4,0.5,0.6,0.7]'::vector, "
        "to_tsvector('english', 'deltamarker term1 common'))",
    )
    delta_sql = f"SELECT id FROM turbohybrid_bench_docs ORDER BY embedding <~> hybrid_query(vector_query => {vector}, text_query => to_tsquery('english', 'deltamarker'), dense_k => 64, bm25_k => 64) LIMIT 10"
    post_insert_delta = timed_result(args.database, delta_sql, args.runs)
    post_insert_stats_output = run_psql(args.database, delta_sql + "; SELECT hybrid_last_scan_stats()")
    post_insert_stats = json.loads(post_insert_stats_output.splitlines()[-1])
    post_vacuum_compacted = None
    post_vacuum_stats = None
    if not args.skip_vacuum_stress:
        run_psql(args.database, "VACUUM turbohybrid_bench_docs")
        post_vacuum_compacted = timed_result(args.database, delta_sql, args.runs)
        post_vacuum_stats_output = run_psql(args.database, delta_sql + "; SELECT hybrid_last_scan_stats()")
        post_vacuum_stats = json.loads(post_vacuum_stats_output.splitlines()[-1])
    wand_matches = run_psql(
        args.database,
        "SELECT (tq_debug_bm25_topk('turbohybrid_bench_idx'::regclass, "
        "to_tsquery('english', 'term1 | term2 | common'), 64, true)->'results') = "
        "(tq_debug_bm25_topk('turbohybrid_bench_idx'::regclass, "
        "to_tsquery('english', 'term1 | term2 | common'), 64, false)->'results')",
    )
    size = run_psql(args.database, "SELECT pg_relation_size('turbohybrid_bench_idx')")

    print(json.dumps({
        "rows": args.rows,
        "build_ms": round(build_ms, 2),
        "delete_vacuum_ms": round(vacuum_ms, 2),
        "index_bytes": int(size),
        "dense_p50_ms": round(dense_p50, 2),
        "dense_p95_ms": round(dense_p95, 2),
        "dense_p99_ms": round(dense_p99, 2),
        "dense_qps_serial": round(dense_qps, 2),
        "bm25_p50_ms": round(bm25_p50, 2),
        "bm25_p95_ms": round(bm25_p95, 2),
        "bm25_p99_ms": round(bm25_p99, 2),
        "bm25_qps_serial": round(bm25_qps, 2),
        "bm25_rare_p50_ms": round(bm25_rare_p50, 2),
        "bm25_rare_p95_ms": round(bm25_rare_p95, 2),
        "bm25_rare_p99_ms": round(bm25_rare_p99, 2),
        "bm25_rare_qps_serial": round(bm25_rare_qps, 2),
        "hybrid_p50_ms": round(hybrid_p50, 2),
        "hybrid_p95_ms": round(hybrid_p95, 2),
        "hybrid_p99_ms": round(hybrid_p99, 2),
        "hybrid_qps_serial": round(hybrid_qps, 2),
        "hybrid_weighted": hybrid_weighted,
        "hybrid_three_term_p50_ms": round(hybrid_three_p50, 2),
        "hybrid_three_term_p95_ms": round(hybrid_three_p95, 2),
        "hybrid_three_term_p99_ms": round(hybrid_three_p99, 2),
        "hybrid_three_term_qps_serial": round(hybrid_three_qps, 2),
        "hybrid_two_term_and": hybrid_and,
        "post_insert_delta": post_insert_delta,
        "post_insert_delta_scan_stats": post_insert_stats,
        "post_vacuum_compacted": post_vacuum_compacted,
        "post_vacuum_compacted_scan_stats": post_vacuum_stats,
        "wand_matches_daat": wand_matches == "t",
        "last_scan_stats": stats,
        "rare_scan_stats": rare_stats,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
