\set rows 1000
\set dims 16

DROP TABLE IF EXISTS turbohybrid_bench_docs;

CREATE TABLE turbohybrid_bench_docs (
	id int,
	tenant_id int,
	embedding vector(16),
	body_tsv tsvector
);

INSERT INTO turbohybrid_bench_docs
SELECT
	gs,
	gs % 16,
	('[' ||
		(gs % 101) / 101.0 || ',' ||
		(gs % 97) / 97.0 || ',' ||
		(gs % 89) / 89.0 || ',' ||
		(gs % 83) / 83.0 || ',' ||
		(gs % 79) / 79.0 || ',' ||
		(gs % 73) / 73.0 || ',' ||
		(gs % 71) / 71.0 || ',' ||
		(gs % 67) / 67.0 || ',' ||
		(gs % 61) / 61.0 || ',' ||
		(gs % 59) / 59.0 || ',' ||
		(gs % 53) / 53.0 || ',' ||
		(gs % 47) / 47.0 || ',' ||
		(gs % 43) / 43.0 || ',' ||
		(gs % 41) / 41.0 || ',' ||
		(gs % 37) / 37.0 || ',' ||
		(gs % 31) / 31.0 || ']')::vector,
	to_tsvector('english',
		'term' || (gs % 32) || ' term' || (gs % 11) || ' common common ' ||
		CASE WHEN gs = (:rows / 2) THEN ' raremarker' ELSE '' END)
FROM generate_series(1, :rows) gs;

CREATE INDEX turbohybrid_bench_idx ON turbohybrid_bench_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
INCLUDE (tenant_id)
WITH (
	graph_m = 8,
	graph_ef_construction = 32,
	graph_ef_search = 40,
	graph_oversampling = 2,
	tq_bits = 4
);

SET enable_seqscan = off;

SELECT id
FROM turbohybrid_bench_docs
ORDER BY embedding <~> hybrid_query(vector_query => '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.2,0.3,0.4,0.5,0.6,0.7]'::vector, dense_k => 64)
LIMIT 10;

SELECT id
FROM turbohybrid_bench_docs
ORDER BY embedding <~> hybrid_query(text_query => to_tsquery('english', 'term1 | common'), dense_k => 0, bm25_k => 64)
LIMIT 10;

SELECT id
FROM turbohybrid_bench_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.2,0.3,0.4,0.5,0.6,0.7]'::vector,
	text_query => to_tsquery('english', 'term1 | common'),
	dense_k => 64,
	bm25_k => 64
)
LIMIT 10;

SELECT hybrid_last_scan_stats();
SELECT pg_size_pretty(pg_relation_size('turbohybrid_bench_idx')) AS index_size;
