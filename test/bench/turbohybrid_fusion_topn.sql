CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS turbohybrid_fusion_docs;
CREATE TABLE turbohybrid_fusion_docs (
	id int PRIMARY KEY,
	embedding vector(8),
	body_tsv tsvector
);

INSERT INTO turbohybrid_fusion_docs
SELECT
	i,
	('[' || array_to_string(ARRAY(
		SELECT (((i * (d + 11)) % 1000)::float8 / 1000.0)::text
		FROM generate_series(1, 8) d
	), ',') || ']')::vector(8),
	to_tsvector('english', concat_ws(' ',
		'common',
		'term' || (i % 257),
		'topic' || (i % 97)))
FROM generate_series(1, 10000) i;

CREATE INDEX turbohybrid_fusion_docs_idx ON turbohybrid_fusion_docs
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
	tq_exact_storage = on,
	bm25_block_max = on
);

ANALYZE turbohybrid_fusion_docs;
SET enable_seqscan = off;

SET hybrid.fusion_hash_threshold = -1;
CREATE TEMP TABLE turbohybrid_fusion_sort AS
SELECT id
FROM turbohybrid_fusion_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::vector,
	text_query => to_tsquery('english', 'common | term1 | term2'),
	dense_k => 5000,
	bm25_k => 5000,
	final_k => 20
)
LIMIT 20;

SELECT
	'sort' AS strategy,
	(hybrid_last_scan_stats()->>'fusion_elapsed_us')::bigint AS fusion_elapsed_us,
	(hybrid_last_scan_stats()->>'elapsed_us')::bigint AS elapsed_us,
	(hybrid_last_scan_stats()->>'union_candidates')::int AS union_candidates,
	(hybrid_last_scan_stats()->>'final_results')::int AS final_results;

SET hybrid.fusion_hash_threshold = 0;
CREATE TEMP TABLE turbohybrid_fusion_hash AS
SELECT id
FROM turbohybrid_fusion_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::vector,
	text_query => to_tsquery('english', 'common | term1 | term2'),
	dense_k => 5000,
	bm25_k => 5000,
	final_k => 20
)
LIMIT 20;

SELECT
	'hash_topn' AS strategy,
	(hybrid_last_scan_stats()->>'fusion_elapsed_us')::bigint AS fusion_elapsed_us,
	(hybrid_last_scan_stats()->>'elapsed_us')::bigint AS elapsed_us,
	(hybrid_last_scan_stats()->>'union_candidates')::int AS union_candidates,
	(hybrid_last_scan_stats()->>'final_results')::int AS final_results,
	(hybrid_last_scan_stats()->>'fusion_heap_size')::int AS fusion_heap_size;

SELECT
	(SELECT array_agg(id ORDER BY ctid) FROM turbohybrid_fusion_sort) =
	(SELECT array_agg(id ORDER BY ctid) FROM turbohybrid_fusion_hash) AS same_order;

RESET hybrid.fusion_hash_threshold;
RESET enable_seqscan;

DROP TABLE turbohybrid_fusion_sort;
DROP TABLE turbohybrid_fusion_hash;
DROP TABLE turbohybrid_fusion_docs;
