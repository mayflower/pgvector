# TurboQuant Plus Architecture

This document describes the TurboQuant Plus implementation in this fork of
pgvector. It is written for maintainers who need to understand how the access
method is wired into PostgreSQL, how the native graph is stored, how query
execution works, and where the performance and correctness boundaries are.

## Goals

TurboQuant Plus adds a pgvector-native index access method named `turboquant`.
The primary goal is to keep HNSW-like top-k behavior while reducing graph scan
cost by traversing compact quantized vector codes and exact-rescoring only the
final candidate band.

The implementation targets:

- A native PostgreSQL index access method, not an external sidecar index.
- WAL-covered on-disk graph state.
- A compact native graph layout optimized for scan-time cache locality.
- Optional TurboQuant Plus correction terms for cosine and inner-product
  quality.
- Optional exact final-band rescoring from stored full-precision vector bytes.
- SQL-visible instrumentation for scan and index diagnostics.

The current native packed-code path supports plain `vector` opclasses only.
`halfvec`, `bit`, and `sparsevec` remain supported through pgvector's existing
HNSW and IVFFlat access methods, but are intentionally not exposed as
TurboQuant opclasses until they have type-specific packed graph
representations.

## Public SQL Surface

The extension install script defines:

- `CREATE ACCESS METHOD turboquant TYPE INDEX HANDLER turboquanthandler`
- `tq_last_scan_stats() RETURNS jsonb`
- `tq_index_stats(regclass) RETURNS jsonb`
- `tq_debug_weighted_self_score(regclass) RETURNS jsonb`
- `tq_debug_psquare_estimate(double precision[], double precision)`

The fresh install script and `0.8.1 -> 0.8.2` upgrade script expose these
TurboQuant opclasses:

- `vector_l2_ops`
- `vector_ip_ops`
- `vector_cosine_ops`
- `vector_l1_ops`

The access method is implemented by reusing the HNSW AM entry-point structure
and swapping in TurboQuant-specific handlers when the AM is created with
`turboquanthandler`.

## Source Map

The implementation is split by concern:

- `src/hnsw.c`
  - AM handler construction.
  - reloption and GUC registration.
  - cost estimation.
  - `tq_index_stats()`.
  - lightweight custom WAL diagnostics.
- `src/hnsw.h`
  - shared metadata, reloption, scan, and TurboQuant query structs.
  - TurboQuant defaults and enum definitions.
- `src/hnswbuild.c`, `src/hnswinsert.c`, `src/hnswscan.c`,
  `src/hnswvacuum.c`
  - AM dispatch glue between legacy HNSW, explicit flat routing, and the native
    TurboQuant graph.
- `src/hnswutils.c`
  - type/opclass support checks.
  - TurboQuant code encoding helpers.
  - query preparation.
  - SIMD dispatch naming and shared scoring utilities.
- `src/tqgraph.c`
  - native graph build, scan, traversal, exact rescore, bulk delete, and
    metadata updates.
- `src/tqgraph.h`
  - native graph tuple formats, build state, scan storage, caches, and exported
    graph helpers.
- `src/tqgraph_score.c`, `src/tqgraph_score.h`
  - quantized scoring, exact scoring, TurboQuant Plus weighted scoring, low-bit
    modes, and SIMD kernels.
- `src/tqgraph_storage.c`
  - page allocation, page chaining, block-map resolution, and tuple appends.
- `src/tqgraph_scan_cache.c`
  - per-backend native graph cache construction and lazy page loading.
- `src/tqgraph_exact.c`
  - exact vector slab storage and exact vector reads for final rescore.
- `src/tqgraph_insert.c`
  - online inserts into native graph storage and adjacency repair.
- `src/tqgraph_cache.c`
  - correction cache for TQ+ shift/scale arrays.
- `src/tqgraph_diag.c`
  - diagnostic SQL helpers.
- `src/tqstats.c`
  - `tq_last_scan_stats()` state and JSON formatting.

## Routing Modes

The `turboquant` AM has a `routing` reloption:

- `auto`
  - Default.
  - Uses the native graph when the opclass supports packed vector codes.
  - Rejects unsupported opclasses instead of silently creating a misleading
    index.
- `graph`
  - Forces the native TurboQuant graph.
  - Still requires a packed-code-compatible opclass.
- `flat`
  - Uses an exact flat scan path through the TurboQuant AM.
  - Useful for diagnostics and correctness comparisons.

The build and scan dispatch checks are centralized around:

- `HnswUseTqNativeGraph(index)`
- `HnswUseTqGraph(index)`
- `HnswUseTqFlat(index)`

If a TurboQuant index is neither native graph nor explicit flat, build, insert,
scan, and vacuum entry points raise:

```text
turboquant requires a native graph-compatible opclass
```

That error is intentional. It prevents catalog-visible indexes from silently
falling back to a different physical design.

## Reloptions

TurboQuant reloptions are parsed into `TqOptions`. The important options are:

| Option | Default | Purpose |
| --- | ---: | --- |
| `routing` | `auto` | Select native graph or explicit flat routing. |
| `graph_m` | HNSW default | Max graph connections, stored in the HNSW-compatible `m` slot. |
| `graph_ef_construction` | `128` | Candidate list size during graph construction. |
| `graph_ef_search` | `64` | Candidate list size during search. |
| `graph_oversampling` | `4` | Multiplier for final candidate collection. |
| `graph_rescore_band` | `auto` | `auto`, `none`, or `exact` final-band rescore policy. |
| `graph_exact_cache` | `auto` | Whether exact vectors are cached in backend memory. |
| `graph_reorder` | `auto` | Whether build output is reordered for locality. |
| `tq_bits` | `4` | Quantized code width. Valid values are `1`, `2`, and `4`. |
| `tq_exact_storage` | `on` | Store full-precision vectors inside the index for exact final-band rescoring. Set to `off` for exact-free, quantized-only storage. |
| `tq_weighted` | `off` | Store and use TurboQuant Plus per-vector correction. |
| `tq_quantile_fit` | `off` | Use quantile-anchored correction fitting instead of Welford fit. |
| `tq_renorm` | `off` | Store a renormalized scale factor for cosine/IP correction. |

`tq_exact_storage = off` removes the exact vector slab from native TurboQuant
graph indexes. Scans cannot exact-rescore final candidates in this mode, so the
effective result order comes from quantized code distances. This is the
smallest storage mode and is useful when index size and latency matter more
than exact recall against brute-force vector ordering. The mode still stores
heap TIDs, quantized codes, graph adjacency, optional INCLUDE payload values,
and optional TQ+ correction metadata.

`tq_renorm` is primarily useful at `tq_bits = 2`. The build code emits notices
when it is requested for bit widths where validation showed little or negative
benefit.

## GUCs

Runtime GUCs provide safety switches and CPU-kernel controls:

| GUC | Default | Purpose |
| --- | ---: | --- |
| `hnsw.tq_graph_prefetch` | `on` | Prefetch code payloads during scoring. |
| `hnsw.tq_graph_stack_scratch` | `on` | Use stack scratch buffers for small frontiers/rescore bands. |
| `hnsw.tq_graph_lowbit_popcnt` | `off` | Experimental sign-only popcount routing for 1-bit scans. |
| `hnsw.tq_graph_i8mm` | `off` | Opt-in ARM I8MM scoring path. |
| `hnsw.tq_graph_avxvnni` | `on` | Allow AVX-VNNI scoring where available. |
| `hnsw.tq_graph_avx512vnni` | `on` | Allow AVX-512 VNNI scoring where available. |
| `hnsw.tq_graph_avx512vpopcntdq` | `on` | Allow AVX-512 VPOPCNTDQ for asymmetric 1-bit scoring. |
| `hnsw.tq_weighted` | `off` | Runtime kill-switch for weighted TQ+ scoring. |
| `hnsw.tq_renorm` | `off` | Runtime kill-switch for renormalized scale use. |
| `hnsw.tq_query_1bit_asymmetric` | `off` | Use bit-plane asymmetric query scoring for 1-bit scans. |
| `hnsw.tq_query_1bit_asymmetric_bits` | `8` | Query quantization width: `8`, `12`, or `16`. |
| `hnsw.tq_build_exact_distances` | `off` | Build graph topology with exact f32 distances. |
| `hnsw.tq_hadamard_simd` | `on` | Use SIMD Hadamard preprocessing where supported. |
| `hnsw.tq_exact_avx512` | `off` | Use explicit AVX-512F exact-rescore kernels. |
| `hnsw.tq_graph_lookahead_prefetch` | `auto` | Neighbor look-ahead prefetch policy. |
| `hnsw.tq_graph_lookahead_threshold_kb` | `24576` | Working-set threshold for auto look-ahead prefetch. |

The index options decide what is stored. The GUCs decide which optional runtime
paths are allowed to run. This lets operators disable risky CPU paths or TQ+
corrections without rebuilding indexes.

## Build Pipeline

The native build entry point is `tqgraphbuild()`.

Build stages:

1. Initialize `TqGraphBuildState`.
   - Resolve type information and metric support from the opclass.
   - Read reloptions such as `graph_m`, `graph_ef_construction`, `tq_bits`,
     `tq_weighted`, `tq_quantile_fit`, and `tq_renorm`.
   - Determine score mode from the support function.
2. Scan the heap with `table_index_build_scan()`.
   - `TqGraphBuildCallback()` forms normalized index values where needed.
   - Build nodes keep heap TID, vector pointer, optional INCLUDE payloads, and
     later code/exact-page references.
3. Fit correction parameters.
   - `TqGraphFitCorrection()` computes `ecShift` and `ecScale`.
   - Welford and quantile-fit paths exist.
   - For TQ+, `mmConst` and quantized per-coordinate weights are prepared.
4. Encode vectors.
   - `TqGraphEncodeBuildNodes()` writes 1-bit, 2-bit, or 4-bit codes.
   - Exact vector bytes are preserved for final-band rescore.
   - TQ+ can store per-vector correction terms.
5. Build edges.
   - `TqGraphBuildEdges()` constructs an HNSW-like layered graph using compact
     code distances unless `hnsw.tq_build_exact_distances` is enabled.
   - Neighbor selection and pruning are handled by graph-local build helpers.
6. Reorder nodes.
   - `TqGraphReorderBuildNodesForLocality()` can rewrite node order for cache
     locality before pages are written.
7. Write pages.
   - Exact vector pages are written first.
   - Correction pages are written when correction arrays are present.
   - Code pages and adjacency pages are written.
   - The metapage records block starts, dimensions, graph parameters, flags,
     and entry-point metadata.
8. WAL-log new pages for logged relations.

The build path logs DEBUG1 timings for scan, correction fit, encode, edge
build, page write, WAL, and total time. It also logs which distance paths were
used during graph construction.

## On-Disk Layout

TurboQuant uses the existing HNSW metapage plus native graph data pages. Pages
are identified by HNSW page opaque kind values and graph operation tags.

### Metapage Fields

The metapage stores:

- Vector dimensions.
- `m` and `efConstruction`.
- Native storage kind.
- `graphEfSearch`.
- `graphOversampling`.
- `graphRescoreBand`.
- `graphMaxLevel`.
- `graphFlags`, used as a generation counter for cache invalidation.
- Entry-point node id and level.
- `tqNodeCount`.
- `tqCodeBytes`.
- `tqPayloadCount` and `tqPayloadBytes`.
- `tqFlags`.
- `tqBits`.
- Start blocks for code, adjacency, exact, and correction page chains.

`graphFlags` is incremented when build or vacuum changes graph-visible state.
Backend caches include this value in their cache key.

### Code Pages

Code pages contain `TqGraphCodeTupleData`.

Each code tuple stores:

- Tuple type.
- Node id.
- Node level.
- Heap TID.
- Exact vector page reference, or an invalid reference for exact-free indexes.
- Payload mask.
- Per-vector scale.
- Norm and code norm/correction fields.
- Optional TQ+ per-vector `ecCorrection`.
- Optional INCLUDE payload values.
- Packed quantized code bytes.

The flexible data area must be accessed through helper functions because the
offset changes when `tq_weighted` is enabled.

### Adjacency Pages

Adjacency pages contain `TqGraphAdjTupleData`.

The graph stores one adjacency tuple per node per storable level. The tuple
contains:

- Tuple type.
- Node id.
- Level.
- Neighbor count.
- Neighbor node ids.

The maximum stored level is bounded by `TQ_GRAPH_MAX_STORED_LEVEL`.

### Exact Vector Pages

Exact vector pages are slab pages. They store full-precision `vector` bytes for
final-band exact rescore. Code tuples reference exact vectors through block and
byte offset fields.

Exact vectors are intentionally part of the default index. The scan path does
not need to fetch heap tuples to compute final distances.

When `tq_exact_storage = off`, these pages are not written, the metapage carries
the exact-free flag, and code tuples store invalid exact-vector references. This
keeps the graph self-contained for approximate top-k traversal but disables
final exact rescoring from index-local full-precision values.

### Correction Pages

Correction pages store TQ+ correction arrays:

- Field 0: `ecShift`.
- Field 1: `ecScale`.

Presence of these pages sets `TQ_GRAPH_TQ_PLUS` on the metapage. When
`tq_weighted` is enabled, `TQ_GRAPH_TQ_WEIGHTED` is also set and code tuples
include per-vector `ecCorrection`.

### Page Chaining

Data page chains use `nextblkno` in the HNSW page opaque. Build usually writes
page groups contiguously, but insert can append pages at the end of the
relation. Scan and vacuum resolve chains rather than assuming contiguous block
numbers.

## Backend Cache

Native graph scans use a backend-local cache in `CacheMemoryContext`.

The cache key includes:

- relation OID
- relfilenumber
- dimensions
- `m`
- max level
- `graphFlags`
- node count
- entry node id
- code bytes and bit width
- payload shape
- start blocks for all graph page chains

`TqGraphBuildCache()` loads code pages, adjacency pages, payload references, and
exact vectors when the index stores them and the cache policy allows it.
Subsequent scans copy the cached `TqGraphScanStorage` descriptor and reuse
loaded arrays.

Cache invalidation happens through:

- relfilenumber changes after rewrite/reindex
- metapage field changes
- `graphFlags` generation bumps after vacuum-visible changes
- explicit `TqGraphInvalidateCaches(index)` calls where needed

## Scan Pipeline

The native scan entry points are:

- `tqgraphbeginscan()`
- `tqgraphrescan()`
- `tqgraphgettuple()`
- `tqgraphendscan()`

The query execution path is:

1. Read and normalize the query datum.
   - Cosine/IP opclasses use the same normalization conventions as pgvector
     HNSW.
2. Read the native graph metapage.
3. Prepare the quantized query with `HnswPrepareTqQuery()`.
4. Seed scan context from executor controls.
   - LIMIT target rows.
   - Planner filter selectivity.
   - Supported int4 payload filters.
5. Determine `resultTarget`.
   - Starts from `graph_ef_search` and `graph_oversampling`.
   - Widens for LIMIT, selectivity, high-dimensional L2, low-bit L2, and
     unmapped scalar filters.
6. Initialize or reuse cached graph storage.
7. Traverse the graph with `TqGraphTraverse()`.
8. Optionally fill a wider candidate band when filters/selectivity require it.
9. Sort approximate candidates.
10. Exact-rescore the final band when enabled and exact vectors are stored.
11. Resort and return TIDs to the executor.
12. Record `tq_last_scan_stats()`.

The scan path rejects unordered scans. TurboQuant is an ordered top-k index path.

## Graph Traversal

Traversal is HNSW-like but optimized around compact code scoring.

Entry selection:

- Start from the metapage entry node.
- Greedily descend upper levels.
- Keep multiple high-level entry points.
- Add deterministic sampled entries for robustness, especially when graph
  topology is weaker than exact-HNSW topology.

Base-layer search:

- Uses a frontier heap and nearest heap.
- Scores neighbor batches with `TqGraphScoreNodeBatch()`.
- Tracks visited nodes with either generation arrays or a temporary bool array.
- Can use stack buffers for small frontiers.
- Can enable look-ahead prefetch for large working sets.

Candidate collection:

- Dead nodes are skipped.
- Payload predicates are evaluated from stored INCLUDE payload values where
  possible.
- `graph_oversampling` controls how many approximate candidates are collected
  before final exact rescore.

## Exact Rescore

Approximate code distances are used for traversal. Final result quality is
protected by exact vector rescoring when the index stores exact vectors.

`TqGraphExactRescore()`:

- Skips candidates already exact-scored.
- Uses cached exact vectors when available.
- Otherwise batches exact reads by block and byte offset.
- Computes the metric with the opclass support function.
- Updates result distances and marks candidates as exact.

`graph_rescore_band` controls this:

- `none`: no final exact rescore.
- `exact`: exact-rescore collected candidates.
- `auto`: currently exact-rescores candidates except where the scan already
  used exact payload-specific paths.

For indexes built with `tq_exact_storage = off`, exact vectors are not present
in the index. The scan treats the index as exact-free and does not perform
final exact rescoring even if `graph_rescore_band` would otherwise allow it.
Use `graph_rescore_band = none` and `graph_exact_cache = off` with
`tq_exact_storage = off` to make that policy explicit in DDL.

## Payload-Aware Filtering

TurboQuant supports a narrow payload acceleration path for integer INCLUDE
columns.

At build/insert time, `TqGraphCopyPayloadValues()` stores supported payload
values in code tuples. At cache-build time, `TqGraphBuildPayloadRefs()` creates
sorted payload references.

During scan:

- If an executor filter maps to a stored payload slot, the graph can restrict
  candidate collection to matching payload references.
- Small payload groups can be exact-scored directly.
- Larger payload groups still use graph traversal plus candidate widening.

For filters that cannot be mapped to payload columns, the scan widens the
candidate band using planner selectivity and `hnsw.max_scan_tuples` so the
executor has enough rows to apply the heap predicate.

## Inserts

The native insert entry point is `tqgraphinsert()`.

Insert flow:

1. Form and normalize the index value.
2. Take the HNSW update lock.
3. Read native graph metadata and correction arrays.
4. Encode the new vector using existing correction parameters.
5. Append exact vector bytes when the index stores them.
6. Append a code tuple.
7. Choose graph level deterministically.
8. Search existing graph for neighbors.
9. Append adjacency tuples for the new node.
10. Update affected existing adjacency tuples.
11. Update the metapage.

The insert path appends new pages when a page chain runs out of space. This is
why scan and vacuum code must follow page chains instead of assuming contiguous
page groups.

Inserts preserve the index's exact-storage mode. Exact-free indexes append only
the quantized code tuple and adjacency records; indexes with exact storage also
append the full-precision vector bytes used for final rescoring.

## Vacuum

`tqgraphbulkdelete()` performs lazy deletion:

- Scans code pages by following the code-page chain.
- Calls the index bulk-delete callback for each live heap TID.
- Marks dead tuples with `TQ_GRAPH_NODE_DEAD`.
- Updates tuple removal stats.
- Tags changed pages with the vacuum graph operation.
- Bumps the metapage generation if anything changed.

The current vacuum path does not physically compact graph pages or repair
adjacency lists. Scans skip dead nodes during candidate collection.

## WAL and Page Operation Tags

Physical page changes use PostgreSQL generic WAL where needed. The code also
records lightweight custom WAL diagnostics on PostgreSQL versions that support
the experimental custom rmgr slot.

Graph page operation tags are stored in high bits of the HNSW page opaque
`pageKind`. `tq_index_stats()` reports counts by operation:

- `page_init`
- `page_link`
- `meta_update`
- `element_insert`
- `neighbor_insert`
- `neighbor_update`
- `duplicate_heaptid`
- `vacuum_delete`
- `vacuum_repair`

These tags are diagnostic. Correct recovery depends on the actual page changes
being WAL-covered.

## Scoring Model

The scoring layer supports L2, inner product, cosine-via-IP, and L1 score
modes. Score mode is derived from the opclass support function.

For packed codes:

- 4-bit is the default.
- 2-bit is available for speed/size tradeoffs.
- 1-bit is experimental and has optional asymmetric query scoring.

The scoring layer includes:

- scalar fallback paths
- NEON/dot-product paths on ARM
- AVX2 paths on x86
- AVX-VNNI paths on supported x86 CPUs
- AVX-512 VNNI and VPOPCNTDQ paths where enabled and supported
- exact f32 rescore paths

All CPU-specific kernels are runtime-gated. The reported
`graph_scoring_kernel` in `tq_last_scan_stats()` is the practical way to verify
which kernel ran.

## TurboQuant Plus Correction

TurboQuant Plus adds correction terms for cosine/IP quality:

- Correction fitting computes `ecShift` and `ecScale`.
- Encoding applies shift/scale before quantization.
- Weighted scoring uses per-coordinate inverse-scale weights.
- Per-vector `ecCorrection` stores the vector-specific term needed by the
  reconstructed similarity formula.
- `mmConst` stores the constant correction term.

The weighted formula reconstructs an approximate rotated-space dot product:

```text
sim = weighted_code_dot + ecCorrection(a) + ecCorrection(b) - mmConst
```

For IP/cosine scoring, this corrected similarity is then converted to the
distance convention used by the opclass. The runtime GUC `hnsw.tq_weighted`
acts as an additional kill-switch on top of the index's `tq_weighted` storage
flag.

`tq_renorm` changes the per-vector scale field for cosine/IP by storing
`l2_length / centroid_norm` instead of plain pre-quantization L2 length. It is
intended for low-bit modes where centroid norms underrepresent magnitude.

## Diagnostics

Use `tq_last_scan_stats()` after a query to verify the executed path. Important
fields:

- `scan_orchestration`
  - `graph_native`, `flat`, or `none`.
- `graph_visited_nodes`
- `graph_scored_codes`
- `graph_candidate_count`
- `graph_rescore_count`
- `graph_rescore_pages`
- `graph_code_pages_read`
- `graph_adj_pages_read`
- `graph_entry_point_count`
- `graph_prepare_us`
- `graph_traverse_us`
- `graph_fill_us`
- `graph_rescore_us`
- `graph_sort_us`
- `graph_total_us`
- `graph_scoring_kernel`
- `graph_storage_kind`
- `graph_tq_bits`
- `graph_query_split_active`

Use `tq_index_stats(index)` to inspect persistent index metadata:

- storage kind
- graph parameters
- TQ flags and bit width
- correction-page presence
- entry level
- page operation tag counts
- WAL diagnostic mode

The regression tests use these functions to assert that native graph scans are
actually selected instead of merely checking that SQL queries return rows.

## Costing

TurboQuant reuses the HNSW cost model with the TurboQuant graph search
parameters. For native graph paths, startup and total costs are discounted
because scans operate over compact cached codes and exact-rescore only the final
candidate band.

Filtered native graph paths receive a stronger discount because candidate-band
widening can keep ordered top-k plans competitive with bitmap/filter/sort
plans.

This is intentionally heuristic. Costing should be validated against real RAG
and filtered top-k benchmarks.

## Current Limitations

- Native packed graph support is limited to plain `vector` opclasses.
- Vacuum marks nodes dead but does not compact pages or rebuild adjacency.
- Correction arrays are fit at build time. Inserts use existing correction
  parameters rather than refitting the whole index.
- Payload acceleration is deliberately narrow and currently targets supported
  int4 INCLUDE payload filters.
- `tq_bits = 1` is experimental and should not be treated as a default-quality
  mode without benchmark acceptance.
- Some GUCs are hardware and workload sensitive. They exist because the best
  p50/p95 behavior differs across ARM, AVX2, AVX-VNNI, and AVX-512 hosts.

## Example Usage

Create a default native graph index:

```sql
CREATE INDEX items_embedding_tq_idx
ON items
USING turboquant (embedding vector_cosine_ops);
```

Create a tuned graph index:

```sql
CREATE INDEX items_embedding_tq_idx
ON items
USING turboquant (embedding vector_cosine_ops)
WITH (
  routing = graph,
  graph_m = 16,
  graph_ef_construction = 128,
  graph_ef_search = 64,
  graph_oversampling = 4,
  graph_rescore_band = auto,
  tq_bits = 4
);
```

Create a compact exact-free graph index:

```sql
CREATE INDEX items_embedding_tq_compact_idx
ON items
USING turboquant (embedding vector_cosine_ops)
WITH (
  routing = graph,
  graph_m = 16,
  graph_ef_construction = 128,
  graph_ef_search = 64,
  graph_oversampling = 4,
  graph_rescore_band = none,
  graph_exact_cache = off,
  tq_bits = 4,
  tq_exact_storage = off
);
```

This form stores quantized graph data without index-local full-precision
vectors. It is substantially smaller, but exact recall against brute-force
ordering can be lower because the final ordering is not exact-rescored.

Enable TQ+ correction for an index:

```sql
SET hnsw.tq_weighted = on;

CREATE INDEX items_embedding_tq_plus_idx
ON items
USING turboquant (embedding vector_cosine_ops)
WITH (
  routing = graph,
  tq_bits = 2,
  tq_weighted = on,
  tq_quantile_fit = on,
  tq_renorm = on
);
```

Verify the scan path:

```sql
SELECT id
FROM items
ORDER BY embedding <=> '[0.1, 0.2, 0.3]'::vector
LIMIT 10;

SELECT tq_last_scan_stats();
SELECT tq_index_stats('items_embedding_tq_idx'::regclass);
```

Look for:

- `scan_orchestration = graph_native`
- `graph_scored_codes > 0`
- `graph_candidate_count > 0`
- useful `graph_rescore_count`
- expected `graph_scoring_kernel`

## Validation Strategy

For source changes:

```bash
PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config make
PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config make install
PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config PGPORT=<port> PGHOST=<host> \
  make installcheck REGRESS=turboquant_vector
```

For graph-storage, scan, insert, or vacuum changes, also run the focused
TurboQuant regression files that cover the touched behavior, for example:

```bash
make installcheck REGRESS="turboquant_vector turboquant_weighted_plus turboquant_renorm_plus turboquant_psquare"
```

For SIMD scoring changes, run the SIMD microbenchmark and parity tests:

```bash
./bench/simd/run_tqscorebench.sh
make installcheck REGRESS="turboquant_simd_parity turboquant_hadamard_simd"
```

For performance claims, use both the FIQA RAG benchmark and an ANN-style vector
benchmark. FIQA is useful because it measures end-to-end retrieval quality
against qrels; ANN-style runs are useful because they expose exact nearest
neighbor recall and storage/latency behavior without RAG label effects.

Compare at least:

- TurboQuant native graph, 4-bit.
- TurboQuant native graph, 4-bit with `tq_exact_storage = off`.
- TurboQuant native graph low-bit variants only when evaluating quality/speed
  tradeoffs.
- pgvector HNSW baseline.

For FIQA, report:

- build time
- index size
- p50, p95, and p99 latency
- nDCG@10
- qrels recall@10
- exact recall@10
- graph scan counters from `tq_last_scan_stats()`

For ANN benchmarks, report:

- dataset name, train vector count, query count, dimensions, metric, and `k`
- build time
- index size
- recall@10 against exact ground truth
- mean, p50, p95, and p99 latency
- QPS

The current exact-storage-off comparison is intentionally mixed:

- On FIQA, `tq_exact_storage = off` produces a 53.125 MiB index versus
  450.039 MiB for HNSW, with exact recall@10 dropping from 0.9915 for HNSW to
  0.9463 and qrels metrics staying close.
- On the 100k GloVe ANN subset, `tq_exact_storage = off` is still faster than
  the other variants, but recall@10 drops to 0.7563 and the 130.9 MiB index is
  larger than the 71.0 MiB HNSW index.

Do not treat a passing SQL query as proof that the native graph ran. Always
check `scan_orchestration`, storage kind, score counters, and kernel stats.

## Maintenance Rules

- Keep the SQL opclass surface aligned with implemented packed-code support.
- Any new opclass must have build, scan, insert, vacuum, and regression
  coverage before it is exposed in `sql/vector.sql`.
- Follow code-page chains for scan and vacuum; inserts can append overflow
  pages outside the original contiguous build range.
- Treat `graphFlags` as the native graph cache generation. Bump it when
  graph-visible state changes.
- Keep exact-rescore behavior explicit when changing low-bit or weighted
  scoring. Faster approximate traversal is not enough if final quality drops.
- Runtime-gate CPU-specific kernels and preserve scalar fallbacks.
- Update both fresh install SQL and versioned upgrade SQL for new SQL-visible
  functions, AM objects, or opclasses.
