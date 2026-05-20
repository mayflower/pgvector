# TurboHybrid BM25 Design

This note started as the Prompt 01 integration map and now documents the
implemented `turbohybrid` prototype: a two-key access method that stores
TurboQuant native graph data plus BM25 docstats, lexicon, postings, and
block-max pages in the same index relation, then returns fused dense/BM25
ordered TIDs from one index scan.

## Current TurboQuant Lifecycle

`sql/vector.sql` defines `turboquant` with `turboquanthandler(internal)`.
`src/hnsw.c` implements the handler by calling `BuildHnswAmRoutine()` with
TurboQuant callbacks. The routine keeps the HNSW ordered-scan shape
(`amcanorderbyop = true`) but uses TurboQuant-specific reloptions and callback
dispatch.

Build flow:

- `turboquantbuild()` in `src/hnswbuild.c` checks
  `HnswUseTqNativeGraph(index)`.
- Native graph-compatible opclasses route to `tqgraphbuild()` in
  `src/tqgraph.c`.
- `tqgraphbuild()` scans the heap with `table_index_build_scan()` and
  `TqGraphBuildCallback()`, fits correction arrays with
  `TqGraphFitCorrection()`, encodes vectors with `TqGraphEncodeBuildNodes()`,
  builds graph edges with `TqGraphBuildEdges()`, optionally reorders nodes with
  `TqGraphReorderBuildNodesForLocality()`, and writes pages with
  `TqGraphWriteIndex()`.
- `TqGraphUpdateMetaPage()` writes native graph metadata into
  `HnswMetaPageData`.

Empty build flow:

- `turboquantbuildempty()` dispatches to `tqgraphbuildempty()` for native graph
  indexes. That creates the init fork metapage and graph pages with the same
  native graph metadata shape.

Insert flow:

- `turboquantinsert()` in `src/hnswinsert.c` dispatches to `tqgraphinsert()`
  for native graph indexes.
- `tqgraphinsert()` validates and normalizes the first index value through
  `HnswFormIndexValue()`, takes `HNSW_UPDATE_LOCK`, and calls
  `TqGraphInsertValueInPlace()` from `src/tqgraph_insert.c`.
- `TqGraphInsertValueInPlace()` appends a code tuple and adjacency tuples using
  `TqGraphAppendTuple()`, updates reciprocal graph links, bumps metapage
  generation, and invalidates scan caches.

Scan flow:

- `turboquantbeginscan()`, `turboquantrescan()`, `turboquantgettuple()`, and
  `turboquantendscan()` in `src/hnswscan.c` dispatch to the native graph
  functions in `src/tqgraph.c` when `HnswUseTqNativeGraph()` is true.
- `tqgraphbeginscan()` initializes `HnswScanOpaqueData`, support functions,
  graph search settings, and a scan memory context.
- `tqgraphrescan()` resets graph stats, query state, result arrays, and the scan
  memory context.
- `tqgraphgettuple()` takes `HNSW_SCAN_LOCK`, calls `TqGraphCollectResults()` on
  first use, then returns ordered heap TIDs from the collected result array.
- `TqGraphCollectResults()` prepares the query (`HnswPrepareTqQuery()`),
  initializes cached scan storage, traverses the graph with `TqGraphTraverse()`,
  widens the candidate band with `TqGraphFillCandidateBand()` when needed,
  exact-rescores with `TqGraphExactRescore()` when exact storage is present,
  sorts `TqGraphResult` values, and records `tq_last_scan_stats()`.

Vacuum flow:

- `turboquantbulkdelete()` dispatches native graph indexes to
  `tqgraphbulkdelete()`.
- `tqgraphbulkdelete()` walks code pages by following the `nextblkno` chain,
  marks dead code tuples with `TQ_GRAPH_NODE_DEAD`, repairs adjacency lists with
  `TqGraphRepairAdjacencyForDeadNodes()`, bumps the metapage generation, and
  returns tuple counts.
- `turboquantvacuumcleanup()` dispatches to `tqgraphvacuumcleanup()`, which
  currently returns existing stats without a full rebuild.

Cost estimate flow:

- `turboquantcostestimate()` in `src/hnsw.c` opens the index, checks
  `HnswUseTqNativeGraph()`, reads graph `ef_search`, delegates to
  `hnswcostestimate_internal()`, and discounts native graph ordered top-k
  startup and total cost.

## Dense Candidate Reuse Points

The reusable dense scan surface should be extracted from the existing native
graph path rather than driving `amgettuple()` from a hybrid AM.

Candidate-producing functions and structs:

- `TqGraphResult` in `src/tqgraph.h` already carries `nodeId`, `heaptid`,
  `distance`, and `exactScored`.
- `TqGraphCollectResults()` owns the full dense candidate-band pipeline today.
- `TqGraphTraverse()` performs graph entry selection and base-layer search into
  a caller-provided `TqGraphResult` array.
- `TqGraphFillCandidateBand()` expands candidate coverage when filters or limit
  estimates require a wider band.
- `TqGraphCollectPayloadExactBand()` can use graph-owned int4 payload references
  to collect a filter-matching exact band without heap fetches.
- `TqGraphExactRescore()` exact-rescores a bounded result band when exact vector
  slabs are present.
- `TqGraphScoreNodeBatch()` and `TqGraphScoreNode()` score cached graph nodes
  from `TqGraphScanStorage`.

Prompt 07 should turn this into a public helper such as
`TqGraphCollectDenseCandidates(...)` that keeps the same behavior as
`tqgraphgettuple()` but returns a result array keyed by `nodeId`.

## Native Graph Storage Layout

The native graph uses the HNSW metapage and page opaque structures with
TurboQuant-specific page kinds.

Current page kinds in `src/hnsw.h`:

- `HNSW_PAGE_KIND_META`: metapage at block 0.
- `HNSW_PAGE_KIND_TQ_CODE`: code tuples, one per graph node.
- `HNSW_PAGE_KIND_TQ_ADJ`: adjacency tuples, one per stored node level.
- `HNSW_PAGE_KIND_TQ_EXACT`: exact vector slabs for final rescore unless
  `tq_exact_storage = off`.
- `HNSW_PAGE_KIND_TQ_CORRECTION`: TQ+ correction arrays.

Current native graph tuple types in `src/tqgraph.h`:

- `TqGraphCodeTupleData`: node id, heap TID, optional payloads, exact-vector
  pointer, scale/norm/correction values, and packed code bytes.
- `TqGraphAdjTupleData`: node id, level, and neighbor node ids.
- `TqGraphExactTupleData` and `TqGraphExactSlabPageHeaderData`: exact vectors.
- `TqGraphCorrectionTupleData`: correction arrays for TQ+ scoring.

Pages are written with `TqGraphAppendPage()`, `PageAddItem()`, and
`TqGraphFinishPage()` during build. Online insert uses `TqGraphAppendTuple()`,
which appends to an existing chain or links a new page through page opaque
`nextblkno`. Readers must follow page chains, not assume pages are contiguous.

BM25 pages should use new page kinds and separate page chains. The least risky
layout is a separate BM25 meta page referenced from new metapage fields or from
a versioned hybrid-only metapage extension, followed by independent docstats,
lexicon, postings, block-max, and delta chains. BM25 readers should reuse the
page-kind validation and chain-following patterns from `src/tqgraph_storage.c`.

## Metapage Extension Strategy

`HnswMetaPageData` currently stores:

- common HNSW identity: magic, version, dimensions, graph settings, entry
  pointer, insert page;
- storage kind: `HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE` for native graph;
- native graph counts and page roots: `tqNodeCount`, `tqEntryNodeId`,
  `tqCodeStartBlkno`, `tqAdjStartBlkno`, `tqExactStartBlkno`,
  `tqCorrectionStartBlkno`;
- generation-like invalidation field: `graphFlags`, incremented by
  `TqGraphUpdateMetaPage()` and `TqGraphBumpMetaGeneration()`;
- TQ flags and dimensions: `tqFlags`, `tqBits`, code and payload sizes.

Fresh BM25 metadata should not change how existing `hnsw` or `turboquant`
indexes are interpreted. The safe approach is:

1. Keep existing fields and meanings unchanged.
2. Add BM25 roots only for the new `turbohybrid` AM, or store a separate BM25
   meta page whose block number is valid only for hybrid indexes.
3. Include a BM25 version and generation so scan caches can invalidate without
   relying only on relation relfilenumber.
4. Treat invalid BM25 roots on `turboquant` indexes as "not hybrid", not as
   corruption.
5. Reject hybrid scans when required BM25 metadata is absent instead of falling
   back to incomplete results.

## Scan Cache Pattern

`src/tqgraph_scan_cache.c` maintains backend-local native graph caches in
`CacheMemoryContext`.

Current pattern:

- `TqGraphInitScanStorage()` finds or builds a `TqGraphNativeCache`.
- `TqGraphCacheMatches()` checks relid, relfilenumber, graph metadata,
  node counts, page roots, dimensions, bits, and payload sizes.
- `TqGraphDropStaleCaches()` removes per-relation cache entries when metadata
  no longer matches.
- `TqGraphInvalidateCaches()` drops all native graph cache entries for an index
  after mutation.
- `TqGraphBuildCache()` preloads code pages, payload refs, adjacency pages, and
  optionally exact vector slabs.

BM25 should mirror this with a separate cache keyed by relid, relfilenumber,
BM25 version/generation, doc count, term count, and page roots. Docstats and a
lexicon directory are good cache residents; postings blocks should be lazy
loaded and bounded so a backend scanning many indexes cannot grow memory
without limit.

## Payload and Filter Interaction

TurboQuant supports included int4 payload columns:

- `TqGraphIndexPayloadCount()` counts `INCLUDE` columns.
- `TqGraphCopyPayloadValues()` copies int4 payload values at build/insert time.
- `TqGraphPayloadSlotForHeapAttr()` maps heap attributes to payload slots.
- `TqGraphBuildPayloadRefs()` creates sorted payload references for cached
  scan storage.
- `TqGraphPayloadRefRange()` finds candidate node ranges for a payload value.
- `TqGraphCollectPayloadExactBand()` can collect filter-matching candidates
  without heap tuple fetches.

`src/tqgraphcontrol.c` wraps controlled TurboQuant index scans during execution
to pass active `LIMIT`, estimated scalar-filter selectivity, and simple int4
payload equality filters into the scan path.

Hybrid retrieval should preserve this behavior. Dense and BM25 branches should
use `nodeId` as the join key. Payload filters that are represented in graph
storage can be applied before final fusion; unsupported filters should either
widen candidate budgets using planner selectivity or return an explicit
unsupported error if correctness cannot be guaranteed.

## BM25 Build and Query Constraints

Bulk BM25 build must not keep an unbounded global map from term to all postings.
The preferred pipeline is:

1. Decode each `tsvector` once during heap scan.
2. Use a bounded per-document accumulator.
3. Emit compact term tuples `(termHash, termBytesRef, nodeId, tf)`.
4. Sort by `(termHash, termBytes, nodeId)`.
5. Reduce into lexicon entries, docstats, postings ranges, and block-max data.
6. Spill or fail early when memory exceeds a safe maintenance-work budget.

Query-time BM25 must not scan all postings for common terms when only a small
`bm25_k` is requested. The stored postings format should support block-level
upper bounds so the query engine can use Block-Max WAND or an equivalent
skipping strategy.

Fusion should avoid heap fetches before final ordered TID return. Dense
candidates and BM25 candidates should be unioned by `nodeId` with a compact
hash table or sort/merge. Nested loops over branch candidates are not acceptable.

## Implementation Checklist

1. Prompt 01: keep vector-only behavior covered; add this integration map.
2. Prompt 02: add a versioned `hybrid_query` varlena type, constructor,
   debug output, dense fallback distance operator, SQL definitions, and tests.
3. Prompt 03: add `turbohybrid` AM skeleton with two-key validation and clear
   unsupported errors for incomplete hybrid behavior.
4. Prompt 04: add BM25 build collection tied to final TurboQuant `nodeId`
   mapping, without creating a usable hybrid index before storage exists.
5. Prompt 05: persist BM25 meta, docstats, lexicon, postings, and block-max
   pages in separate page chains.
6. Prompt 06: add BM25 top-k with DAAT correctness and WAND/block skipping for
   larger postings lists.
7. Prompt 07: refactor dense native graph candidate collection into a reusable
   helper returning `nodeId`-keyed candidates.
8. Prompt 08: implement fused hybrid scan, RRF, weighted fusion, deterministic
   tie-breaking, ordered `amgettuple()` TID return, and a tested guard for the
   reserved BM25-only exact-rescore option.
9. Prompt 09: add append-only BM25 delta segments for online inserts and safe
   update/delete behavior.
10. Prompt 10: make vacuum BM25-aware, filter dead nodes, optionally compact
    delta pages, and expose hybrid stats.
11. Prompt 11: add hybrid reloptions, GUCs, costing, explain/debug diagnostics,
    and `hybrid_last_scan_stats()`.
12. Prompt 12: add deterministic regression coverage and a manual benchmark
    harness.
13. Prompt 13: profile and harden memory layout, cache behavior, WAND skipping,
    and fusion performance.

## Prompt 01 Safety Baseline

The existing regression suite already covers:

- HNSW vector build and ordered scans in `test/sql/hnsw_vector.sql`.
- TurboQuant vector build, ordered scans, and `tq_index_stats()` in
  `test/sql/turboquant_vector.sql`.

`test/sql/turbohybrid_safety.sql` adds a small explicit no-op guard for future
hybrid work: it builds one HNSW index and one native TurboQuant index, verifies
top-1 ordered scan behavior, checks that TurboQuant reports native graph scan
stats, and checks that `tq_index_stats()` reports live native graph nodes.

## Current TurboHybrid Surface

`turbohybrid` indexes require exactly two key columns:

- key 1: `vector` with one of the `vector_*_hybrid_ops` opclasses
- key 2: `tsvector` with `bm25_tsvector_ops`

Included int4 payload columns are accepted because the dense branch reuses the
native TurboQuant graph scan. The SQL query surface is:

```sql
ORDER BY embedding <~> hybrid_query(
  vector_query => $1::vector,
  text_query => websearch_to_tsquery('english', $2),
  fusion => 'rrf',
  dense_k => 400,
  bm25_k => 400,
  rrf_k => 60
)
```

The scan supports:

- dense-only retrieval when only `vector_query` is present
- BM25-only retrieval when only `text_query` is present
- hybrid RRF fusion by default
- weighted normalized fusion with `fusion => 'weighted'`
- `require_bm25_match` filtering after branch union
- `hybrid_last_scan_stats()` diagnostics for branch counts and work counters

Online inserts append dense graph nodes through the native TurboQuant insert
path and append compact BM25 delta records keyed by the same `nodeId`. Query
time BM25 merges base postings plus live delta records, and vacuum dead-node
filtering prevents deleted delta nodes from returning rows after graph vacuum.
Vacuum cleanup compacts the BM25 delta when the configured threshold is
exceeded by rebuilding live base postings plus live delta records into new base
BM25 page chains, atomically updating the BM25 metadata, resetting the delta
root, and invalidating reader caches.

The main scan-time knobs are exposed as `hybrid.*` GUCs:

- `hybrid.enable_wand`: force the DAAT fallback when off.
- `hybrid.max_union_candidates`: bound the fused candidate array.
- `hybrid.default_dense_k`, `hybrid.default_bm25_k`, `hybrid.default_rrf_k`:
  constructor defaults when callers pass NULL explicitly.
- `hybrid.force_fusion`: debug override for RRF vs weighted fusion.
- `hybrid.enable_exact_rescore_for_bm25_only`: reserved for exact-rescore
  experiments. The default is off. When enabled with a mixed vector + BM25
  query, the scan fails with `feature_not_supported` if BM25 returns candidates
  that were not scored by the dense branch, because exact rescoring those
  BM25-only nodes is not implemented yet.

## BM25 Storage Layout

Bulk build first creates the native TurboQuant graph, then builds BM25 over the
final graph `nodeId` namespace. The collector scans code pages to map heap TIDs
to node IDs, scans heap tuples once for the indexed `tsvector`, emits compact
term tuples, sorts them by `(termHash, termBytes, nodeId)`, and writes separate
BM25 page chains:

- `HNSW_PAGE_KIND_TQ_BM25_META`: BM25 version, document counts, avgdl, roots
- `HNSW_PAGE_KIND_TQ_BM25_DOCSTATS`: dense doc length array chunks keyed by nodeId
- `HNSW_PAGE_KIND_TQ_BM25_LEXICON`: term dictionary entries and postings pointers
- `HNSW_PAGE_KIND_TQ_BM25_POSTINGS`: nodeId/tf postings tuples
- `HNSW_PAGE_KIND_TQ_BM25_BLOCKMAX`: per-term upper-bound records
- `HNSW_PAGE_KIND_TQ_BM25_DELTA`: append-only per-document mutation records

Readers validate page kinds and follow opaque `nextblkno` chains. They do not
assume BM25 pages are contiguous.

## Query Complexity

The dense branch is the existing native TurboQuant graph candidate-band path:
graph traversal, optional candidate widening, final exact rescore when exact
storage is present, and sort by dense distance. `TqGraphCollectDenseCandidates`
exposes that path as a reusable `nodeId`-keyed array.

The BM25 branch extracts positive `tsquery` terms, rejects unsupported NOT,
prefix, and phrase operators, resolves terms through a backend-local lexicon
directory cache, copies cached doc-length/heap-TID/liveness arrays into the
scan memory context, counts live delta document frequencies, decodes matching
base postings, merges matching delta tuples, and accumulates scores in dense
arrays keyed by `nodeId`.
Single-term common queries use an in-tuple block upper-bound skip path when
WAND is enabled; `hybrid.enable_wand = off` forces the DAAT fallback for parity
tests. Unknown terms return zero BM25 candidates without heap access.

Fusion sorts the dense and BM25 branch results by `nodeId`, merges them in one
linear pass, computes RRF or weighted normalized scores, and sorts only the
union candidate array. The final scan returns heap TIDs without fetching heap
tuples in the index AM.

## Benchmark Harness

Manual benchmark files live under `test/bench/` and are intentionally excluded
from normal `installcheck`.

Example:

```bash
python3 test/bench/turbohybrid_bench.py --database contrib_regression --rows 1000 --runs 20
```

The JSON report includes build time, index size, p50/p95 dense-only latency,
p50/p95 BM25-only latency, p50/p95 hybrid latency, and
`hybrid_last_scan_stats()` counters such as graph nodes visited, postings
decoded, blocks visited/skipped, union candidates, and exact rescore count.

`tq_index_stats(regclass)` also reports BM25 fields for hybrid indexes:
`hybrid`, `bm25_doc_count`, `bm25_live_doc_count`, `bm25_dead_doc_count`,
`bm25_avgdl`, `bm25_term_count`, `bm25_postings_pages`,
`bm25_blockmax_pages`, `bm25_delta_pages`, `bm25_delta_generation`, and
`bm25_last_compaction`.

## Current Limitations

- BM25 delta compaction writes fresh live BM25 page chains and makes old chains
  unreachable through metadata, but it does not yet recycle old pages inside the
  index relation.
- The BM25 reader cache keeps docstats, heap TIDs, liveness, and the lexicon
  directory in backend-local cache memory. Postings blocks are still loaded
  lazily per query rather than cached with an eviction policy.
- The first postings format stores one postings tuple per term. The current
  WAND path skips fixed-size in-tuple blocks for single-term queries, but
  multi-term Block-Max WAND and chunked postings are still future work.
- Updates are represented as an insert of a new delta node plus graph vacuum of
  the old node. Vacuum compaction removes dead lexical entries from the active
  BM25 metadata roots once the delta threshold is reached.
- Planner costing is conservative and still delegates to generic costs; use
  `hybrid_last_scan_stats()` to inspect actual scan behavior.
