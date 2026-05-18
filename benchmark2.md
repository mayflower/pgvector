# ANN-Benchmarks: Larger GloVe TurboQuant Comparison

Date: 2026-05-18

Branch: `experiment/tq-exact-free-codebook-lut`

Source state: branch `experiment/tq-exact-free-codebook-lut` with compact
adjacency storage changes after `2950366`

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
| HNSW | 0.8509 | 6.090 | 4.451 | 14.831 | 22.560 | 164.2 | 614.5 | 71.0 |
| TQ exact_storage=on | 0.8507 | 0.334 | 0.332 | 0.370 | 0.393 | 2993.2 | 478.3 | 170.1 |
| TQ exact_storage=off | 0.7563 | 0.318 | 0.314 | 0.366 | 0.407 | 3145.4 | 272.3 | 130.9 |

## Interpretation

The earlier `random-s-100-angular` run was only a smoke test. It was too
friendly to all methods and too synthetic to say much about production behavior.
The 100k GloVe subset is still not million scale, but it is a real embedding
distribution and exposes the important tradeoff:

- TurboQuant with exact storage matches HNSW recall at these settings while
  answering about 18x faster in this local run.
- The exact-storage TurboQuant index is much larger than HNSW: 170 MiB vs
  71 MiB. The exact vectors are still stored, and the native graph/code pages add
  their own overhead.
- Exact-free TurboQuant is only slightly faster than exact-storage TurboQuant
  here, but recall drops from about 0.851 to 0.756. That is not a good default.
- Exact-free does reduce index size compared with exact-storage TurboQuant
  by about 23%, but it is still larger than HNSW in this implementation.

## Larger-Scale Boundary Checks

I also tried larger local runs to see where the current implementation stops
being practical on this machine:

| Attempt | Outcome |
|---|---|
| Full `glove-100-angular`, 1,183,514 train x 100 dims | TurboQuant build was still running after about 38 minutes; cancelled. |
| `nytimes-256-angular`, 290,000 train x 256 dims | TurboQuant build was still running after about 29 minutes; cancelled. |
| GloVe 200k x 100 dims, HNSW `efConstruction=200`, `ef_search=120` | HNSW built in 1549 s, index about 142 MiB, recall@10 0.8964 on 1,000 queries, mean 89.4 ms. |
| GloVe 200k x 100 dims, TurboQuant exact_storage=on | Build was still in `CREATE INDEX` after more than 20 minutes; cancelled. |

So the larger benchmark result is not just "bigger data changes the numbers".
It also shows a build scalability problem for TurboQuant that the 90k random
smoke run hid.

## Raw Output

The completed 100k direct-driver output was captured in:

```text
/tmp/pgvector_glove100k_direct_benchmark.log
```
