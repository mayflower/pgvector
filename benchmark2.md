# ANN-Benchmarks: Larger GloVe TurboQuant Comparison

Date: 2026-05-18

Branch: `master`

Source state: working tree after compact adjacency storage and low-dimensional
TurboQuant code-code build scoring fixes.

Harness: `../ann-benchmarks` data and PostgreSQL adapters, driven directly to
avoid the full 10,000-query ANN-Benchmarks runner cost on this local machine.

## Main Completed Run

Dataset: controlled GloVe subset derived from `glove-100-angular`

- Train vectors: 100,000
- Query vectors: 1,000
- Dimensions: 100
- Distance: angular / cosine
- Result count: `k = 10`
- Ground truth: exact top-10 recomputed against the 100k subset

Configuration:

- `M = 16`
- `efConstruction = 128`
- `ef_search = 64`
- TurboQuant: `routing = graph`, `tq_bits = 4`, `graph_oversampling = 4`

| Variant | recall@10 | mean ms | p50 ms | p95 ms | p99 ms | QPS | build s | index MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HNSW | 0.8468 | 1.157 | 0.893 | 2.495 | 6.788 | 864.1 | 16.7 | 71.0 |
| TQ exact_storage=on | 0.8130 | 0.447 | 0.441 | 0.553 | 0.644 | 2238.8 | 70.3 | 62.8 |
| TQ exact_storage=off | 0.7304 | 0.419 | 0.379 | 0.642 | 0.815 | 2385.7 | 78.8 | 23.7 |

## Interpretation

The earlier `random-s-100-angular` run was only a smoke test. It was too
friendly to all methods and too synthetic to say much about production behavior.
The 100k GloVe subset is still not million scale, but it is a real embedding
distribution and exposes the important tradeoff:

- TurboQuant with exact storage is lower-recall than HNSW at these settings
  after the fixed approximate code-code build path: 0.8130 vs 0.8468
  recall@10.
- TurboQuant improves mean query latency by about 2.6x in this local run.
- After compact adjacency storage, exact-storage TurboQuant is smaller than
  HNSW on this dataset: 62.8 MiB vs 71.0 MiB.
- Exact-free TurboQuant is much smaller than both baselines at 23.7 MiB, but
  recall drops from about 0.847 to 0.730. That is not a good default.
- HNSW still builds faster in this rerun: 16.7 s versus about 70-79 s for
  TurboQuant. The earlier 338-339 s TurboQuant build was a low-dimensional
  scoring regression where 100-dimensional build-time graph comparisons fell
  back to exact vectors instead of quantized code-code scoring.

## Larger-Scale Boundary Checks

These larger local runs were attempted before the low-dimensional code-code
build scoring fix above. They are useful as cautionary history, but should not
be cited as current build-performance numbers without rerunning them:

| Attempt | Outcome |
|---|---|
| Full `glove-100-angular`, 1,183,514 train x 100 dims | TurboQuant build was still running after about 38 minutes; cancelled. |
| `nytimes-256-angular`, 290,000 train x 256 dims | TurboQuant build was still running after about 29 minutes; cancelled. |
| GloVe 200k x 100 dims, HNSW `efConstruction=200`, `ef_search=120` | HNSW built in 1549 s, index about 142 MiB, recall@10 0.8964 on 1,000 queries, mean 89.4 ms. |
| GloVe 200k x 100 dims, TurboQuant exact_storage=on | Build was still in `CREATE INDEX` after more than 20 minutes; cancelled. |

The completed 100k rerun shows that the 100-dimensional build regression was
real and fixable. The larger cancelled runs still need to be repeated before
making a current million-scale build claim.

## Raw Output

The completed 100k direct-driver output was captured in:

```text
/tmp/pgvector_glove100k_direct_benchmark_20260518T162221Z.log
```
