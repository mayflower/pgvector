# Real RAG Benchmark

- Dataset: `fiqa-openai-exact-off`
- Dataset path: `.cache/beir/fiqa`
- Commit: `af2802ef029b0ebc0c23477ef8aa3d8eb0188b38`
- PostgreSQL: `16.13 (Homebrew)`
- Rows: 57638
- Query count: 648
- Dimensions: 1536
- Dense candidates: 400
- BM25 candidates: 400
- Final k: 10
- Warmup passes: 1
- Timed passes: 1

## Quality And Latency

| Method | nDCG@k | Recall@k | MRR@k | MAP@k | p50 ms | p95 ms | p99 ms | Mean ms | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| turbohybrid_exact_storage_off | 0.419068 | 0.493158 | 0.487622 | 0.343457 | 4.868 | 12.655 | 15.804 | 5.85 | 170.949 |

## Build And Storage

| Method | Build ms | Build WAL bytes | Index bytes | Index MiB |
|---|---:|---:|---:|---:|
| turbohybrid_exact_storage_off | 70200.796 | 67811928 | 83894272 | 80.01 |

## Notes

- `postgres_sql_rrf` is normal pgvector HNSW for dense retrieval plus PostgreSQL full-text search and SQL-level reciprocal rank fusion.
- `turbohybrid` is the single TurboHybrid index over the same vector and `tsvector` columns.
- `turbohybrid_exact_storage_off` is the same TurboHybrid index with `tq_exact_storage = off`.
- Both methods use the same text-embedding vectors, query texts, qrels, candidate budgets, and final-k.
- Latency is measured inside PostgreSQL with `clock_timestamp()` around each retrieval query.
