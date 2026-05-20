<p align="center">
  <img src="./logo.png" alt="TurboHybrid logo" width="720">
</p>

## Benchmark 

We ran a FIQA hybrid RAG benchmark with 57,638 `text-embedding-3-small` corpus
vectors, 648 test queries, `k = 10`, one warmup pass, one measured pass,
`dense_k = 400`, and `bm25_k = 400`. Latency is measured inside PostgreSQL
around each retrieval query.

| Method | Build ms | Index MB | p50 ms | p95 ms | p99 ms | nDCG@10 | qrels recall@10 | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pgvector HNSW + Postgres FTS SQL RRF | 176,607.923 | 468.54 | 23.364 | 60.289 | 81.790 | 0.4175 | 0.4976 | 33.434 |
| TurboHybrid, 4-bit, exact storage on | 212,029.072 | 420.34 | 13.220 | 30.169 | 38.612 | 0.4197 | 0.4971 | 65.883 |
| TurboHybrid, 4-bit, exact storage off | 70,200.796 | 80.01 | 4.868 | 12.655 | 15.804 | 0.4191 | 0.4932 | 170.949 |

TurboHybrid with exact storage on keeps retrieval quality roughly
flat against the pgvector HNSW plus Postgres FTS SQL RRF baseline while cutting
p95 latency from 60.289 ms to 30.169 ms and reducing index size from 468.54 MB
to 420.34 MB. With exact storage off, p95 latency falls to 12.655 ms and index
size to 80.01 MB, while nDCG@10 stays close and qrels recall@10 drops slightly
from 0.4976 to 0.4932.

TurboHybrid extends pgvector with a PostgreSQL-native hybrid search index that
combines dense vector retrieval and lexical BM25 retrieval in a single index
access method. Instead of maintaining a pgvector HNSW index, a PostgreSQL text
search index, and an application or SQL layer that merges both result sets,
`turbohybrid` lets one index scan collect, score, and fuse candidates for RAG
and search workloads.

The index is built over a vector column and a `tsvector` column. The vector side
uses TurboQuant graph storage: graph links provide approximate nearest-neighbor
traversal, while compact low-bit vector codes support fast candidate scoring.
The index can optionally retain full vectors for final rescoring through
`tq_exact_storage`; keeping exact vectors is the quality-first default, while
disabling them reduces index size and latency at the cost of relying on
quantized-only ranking. The lexical side stores BM25 metadata, document
statistics, a term lexicon, postings, and block maxima inside the same index
relation.

Queries are represented with the `hybrid_query(...)` SQL constructor. A query
can include the dense embedding, lexical query text, fusion mode, dense and
BM25 candidate limits, weights, RRF parameters, and whether a BM25 match is
required. The vector opclasses expose hybrid distance operators for L2, inner
product, and cosine search, while the `bm25_tsvector_ops` opclass supplies the
text side of the composite index. At execution time, TurboHybrid resolves the
lexical terms, gathers dense graph candidates, scores BM25 candidates, and
combines the two streams with reciprocal-rank fusion or weighted hybrid
scoring.

The BM25 implementation is designed for index-level execution rather than
plain SQL post-processing. It stores enough corpus statistics to score documents
directly from postings, supports block-max data, and can use WAND-style pruning
to skip postings blocks that cannot enter the top result set. Query-time caches
and SIMD-aware decode and scoring paths reduce repeated lexicon, docstats, and
postings work for hot workloads. Inserted documents are tracked through BM25
delta pages and integrated with normal PostgreSQL index maintenance paths.

Operationally, TurboHybrid keeps hybrid retrieval inside PostgreSQL: it is
created with `CREATE INDEX ... USING turbohybrid`, configured with reloptions
for graph, quantization, exact storage, and BM25 behavior, and tuned with GUC
defaults such as `hybrid.default_dense_k`, `hybrid.default_bm25_k`, and
`hybrid.default_rrf_k`. The intended benefit is lower end-to-end latency and
simpler query plans for hybrid RAG, while preserving SQL control over filters,
transactions, DDL, and VACUUM.

The main tradeoffs are explicit. Exact vector storage gives safer ranking
quality but uses more space. Exact storage off can make the index much smaller
and faster, but recall depends on compressed-code scoring. Build time, candidate
limits, BM25 pruning, and fusion defaults need workload-specific validation, so
TurboHybrid should be evaluated with both system metrics and task-level IR or
RAG quality metrics.
