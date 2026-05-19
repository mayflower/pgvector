# FIQA RAG Benchmark

Date: 2026-05-18
Commit: `29503668471c66968af1070948bf75674a09a5e2`
Source state: branch includes compact adjacency storage changes
Dataset: FIQA, 57,638 documents, 648 test queries, 1,536 dimensions
Metric: cosine, `k = 10`, 1 warmup pass, 3 measured passes
PostgreSQL: 16.13 Homebrew, arm64

## Index Settings

All runs used `work_mem = '256MB'` and `hnsw.ef_search = 64`.

| Method | Index options |
|---|---|
| TurboQuant, exact storage on | `routing = graph, graph_m = 16, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, graph_rescore_band = auto, graph_exact_cache = auto, graph_reorder = auto, tq_bits = 4, tq_exact_storage = on` |
| TurboQuant, exact storage off | `routing = graph, graph_m = 16, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, graph_rescore_band = none, graph_exact_cache = off, graph_reorder = auto, tq_bits = 4, tq_exact_storage = off` |
| pgvector HNSW | `m = 16, ef_construction = 128` |

## Results

| Method | Index size | Build time | Mean latency | p50 | p95 | p99 | Exact recall@10 | nDCG@10 | Qrels recall@10 | MRR@10 | MAP@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant, exact storage on | 393.453 MiB | 35.762 s | 0.655 ms | 0.644 ms | 0.764 ms | 0.853 ms | 0.9818 | 0.4410 | 0.5163 | 0.5096 | 0.3635 |
| TurboQuant, exact storage off | 53.125 MiB | 35.666 s | 0.551 ms | 0.544 ms | 0.639 ms | 0.703 ms | 0.9463 | 0.4420 | 0.5132 | 0.5130 | 0.3659 |
| pgvector HNSW | 450.039 MiB | 201.022 s | 1.864 ms | 1.745 ms | 2.948 ms | 3.733 ms | 0.9915 | 0.4438 | 0.5182 | 0.5154 | 0.3657 |

## Takeaways

With compact adjacency storage, `tq_exact_storage = off` reduces the TurboQuant graph index to 53.125 MiB. That is 86.5% smaller than TurboQuant with exact vectors and 88.2% smaller than pgvector HNSW on this FIQA run.

Exact-free TurboQuant is also the fastest query variant here: mean latency is 0.551 ms versus 1.864 ms for HNSW, and p95 is 0.639 ms versus 2.948 ms. The retrieval metrics against FIQA qrels stay close across all variants, while exact recall@10 drops from 0.9818 with exact TurboQuant storage to 0.9463 without exact storage.

TurboQuant with exact storage remains smaller and faster than HNSW in this run, but most of its 393.453 MiB footprint is the stored exact vector payload. Exact-free storage is the variant that shows the intended compression effect.

## Result Files

- `bench/results/fiqa_tq_exact_storage_on_compact_adj_worktree_20260518T152209Z.json`
- `bench/results/fiqa_tq_exact_storage_off_compact_adj_worktree_20260518T152209Z.json`
- `bench/results/fiqa_pgvector_hnsw_compact_adj_worktree_20260518T152209Z.json`
- `bench/results/fiqa_three_variants_compact_adj_worktree_20260518T152209Z.json`
