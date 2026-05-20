# Turbohybrid Benchmark Harness

This harness is for local acceptance of the native `turbohybrid` access method.
It complements the small SQL regression tests with a larger synthetic corpus that
has one term in every row, one rare term, two-term and three-term queries, and a
delete plus `VACUUM` pass before measuring warmed-cache latency.

Run from the repository root after installing the extension into the target
PostgreSQL instance:

```bash
python3 test/bench/turbohybrid_bench.py --database contrib_regression --rows 100000 --runs 20
```

The JSON report includes build time, delete/vacuum time, index size, p50/p95/p99
latency, serial QPS for dense-only, BM25-only, rare-term BM25, hybrid RRF,
hybrid weighted, two-term OR, two-term AND, post-insert delta, and post-vacuum
compacted queries, WAND-vs-DAAT top-k parity, and `hybrid_last_scan_stats()`
counters for postings decoded/skipped, graph work, exact rescore count, and
final candidate counts.

Use `--skip-vacuum-stress` when isolating pure query latency from maintenance
effects.

For the slow high-df and compaction storage smoke test, run:

```bash
psql -X -v ON_ERROR_STOP=1 -d contrib_regression -f test/bench/turbohybrid_highdf_wal.sql
```

That script builds a 70,000-row corpus with one term in every document, checks
that the postings list spans multiple chunks, verifies WAND returns top-k
results, inserts delta rows, runs `VACUUM`, and checks compaction page counters.

For deterministic WAND-vs-DAAT stress coverage, run:

```bash
psql -X -v ON_ERROR_STOP=1 -d contrib_regression -f test/bench/turbohybrid_wand_stress.sql
```

That script builds a 1,000-row synthetic corpus with 100 vocabulary terms and
runs 100 deterministic OR/AND queries at `k = 1, 5, 10, 100`, comparing WAND
top-k results with DAAT for every query.

For the large-candidate fusion smoke from the speed prompt, run:

```bash
psql -X -v ON_ERROR_STOP=1 -d contrib_regression -f test/bench/turbohybrid_fusion_topn.sql
```

That script compares forced sort/merge fusion against forced hash plus bounded
top-N fusion at `dense_k = 5000`, `bm25_k = 5000`, and `final_k = 20`, prints
`fusion_elapsed_us`, and verifies the ordered top-20 results match.

For the full beta gate, including 10k and 100k synthetic rows, delta 1/5/10%
query shapes, compaction timing, page counters, WAL bytes, p95/p99, and QPS,
run:

```bash
python3 benchmarks/turbohybrid/suite.py acceptance \
  --database contrib_regression \
  --rows-list 10000,100000 \
  --runs 30 \
  --concurrency 1,4,16 \
  --output-dir benchmarks/turbohybrid/results
```

Validation smoke on May 20, 2026, with 10k rows, 16 dimensions, `--runs 1`,
and concurrency 1 produced PASS for WAND parity, deterministic hybrid top-k,
BM25 rare p95, fusion time share, delta 10% p95, and BM25 cache reporting. The
100k smoke produced the same PASS checks except `post_compaction_vs_delta_0_p95`
reported WARN at 1.597 against the default 1.2 threshold. This is a low-run
gate validation, not a publishable latency result.

The same local run of `test/bench/turbohybrid_fusion_topn.sql` reported:

- Sort/merge fusion: `fusion_elapsed_us = 2071`, `elapsed_us = 5495`,
  `union_candidates = 6853`, `final_results = 20`
- Hash plus bounded top-N fusion: `fusion_elapsed_us = 294`,
  `elapsed_us = 3244`, `union_candidates = 6853`,
  `fusion_heap_size = 20`
- Ordered result parity: true

Representative local run on May 19, 2026:

```bash
python3 test/bench/turbohybrid_bench.py --database contrib_regression --rows 100000 --runs 5
```

Results:

- Build time: 15362.19 ms
- Delete plus `VACUUM`: 179.65 ms
- Index size: 24,797,184 bytes
- Dense-only: p50 112.54 ms, p95/p99 146.44 ms, 8.78 serial QPS
- BM25 common-term: p50 380.17 ms, p95/p99 431.28 ms, 2.58 serial QPS
- BM25 rare-term: p50 38.40 ms, p95/p99 50.09 ms, 24.31 serial QPS
- Hybrid two-term: p50 523.14 ms, p95/p99 754.55 ms, 1.75 serial QPS
- Hybrid three-term: p50 453.35 ms, p95/p99 464.99 ms, 2.22 serial QPS
- WAND top-k matches DAAT: true
- Common-term hybrid scan decoded 134,421 postings, visited 223 BM25
  blocks, skipped 221 BM25 blocks, visited 91 graph nodes, scored 672 graph
  codes, exact-rescored 64 dense candidates, and produced 128 final fused
  candidates.
- Rare-term BM25 scan decoded 1 posting, scored 1 candidate, and produced 1
  final result.
