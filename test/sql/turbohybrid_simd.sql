SELECT jsonb_typeof(tq_simd_capabilities()) AS capabilities_type;
SELECT jsonb_typeof(tq_last_simd_stats()) AS stats_type;

SET hnsw.tq_simd_force = scalar;
SET hnsw.tq_exact_simd_force = scalar;
SET hybrid.bm25_simd_force = scalar;

DO $$
BEGIN
	PERFORM set_config('hybrid.bm25_simd_force', 'avx512f', true);
	RAISE EXCEPTION 'expected unsupported BM25 SIMD force rejection';
EXCEPTION WHEN invalid_parameter_value THEN
END
$$;

SELECT tq_simd_capabilities()->>'dense_force' AS dense_force;
SELECT tq_simd_capabilities()->>'exact_force' AS exact_force;
SELECT tq_simd_capabilities()->>'bm25_force' AS bm25_force;
SELECT tq_simd_capabilities() ? 'compile_avx512_weighted' AS has_avx512_weighted_capability;
SHOW hnsw.tq_graph_avx512_weighted;
SET hnsw.tq_graph_avx512_weighted = off;
SELECT tq_last_simd_stats()->>'avx512_weighted' AS avx512_weighted_mode;

CREATE TABLE tqh_simd_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_simd_docs VALUES
	(1, '[1,0,0]', to_tsvector('english', 'alpha common')),
	(2, '[0,1,0]', to_tsvector('english', 'beta common')),
	(3, '[0,0,1]', to_tsvector('english', 'gamma rare'));

CREATE INDEX tqh_simd_docs_idx ON tqh_simd_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

SET enable_seqscan = off;

SELECT id
FROM tqh_simd_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[1,0,0]'::vector,
	text_query => to_tsquery('english', 'common'),
	dense_k => 8,
	bm25_k => 8,
	final_k => 2
)
LIMIT 2;

SELECT hybrid_last_scan_stats() ? 'bm25_decode_kernel' AS has_bm25_decode_kernel;
SELECT hybrid_last_scan_stats()->>'bm25_decode_kernel' AS bm25_decode_kernel;
SELECT hybrid_last_scan_stats()->>'bm25_simd_force' AS bm25_simd_force;
SELECT tq_last_simd_stats()->>'bm25_score_kernel' AS bm25_score_kernel;

RESET enable_seqscan;
RESET hnsw.tq_simd_force;
RESET hnsw.tq_exact_simd_force;
RESET hybrid.bm25_simd_force;
RESET hnsw.tq_graph_avx512_weighted;

DROP TABLE tqh_simd_docs;

CREATE TABLE tqh_simd_bm25_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_simd_bm25_docs
SELECT i, ('[' || i || ',0,0]')::vector, to_tsvector('english', 'simdcommon token')
FROM generate_series(1, 12) i;

SET hybrid.debug_postings_chunk_size = 4;

CREATE INDEX tqh_simd_bm25_docs_idx ON tqh_simd_bm25_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (bm25_precompute_tf_norm = on);

SELECT tq_debug_bm25_stats('tqh_simd_bm25_docs_idx'::regclass)->>'bm25_precompute_tf_norm' AS precompute_tf_norm;
SELECT (tq_debug_bm25_term_stats('tqh_simd_bm25_docs_idx'::regclass, 'simdcommon')->'posting_encoding_counts'->>'offset16')::int > 0 AS has_offset16;
SELECT (tq_debug_bm25_term_stats('tqh_simd_bm25_docs_idx'::regclass, 'simdcommon')->'posting_encoding_counts'->>'tfnorm_q16')::int > 0 AS has_tfnorm_q16;
SELECT tq_debug_bm25_topk('tqh_simd_bm25_docs_idx'::regclass, to_tsquery('english', 'simdcommon'), 5, true)->'results' =
	tq_debug_bm25_topk('tqh_simd_bm25_docs_idx'::regclass, to_tsquery('english', 'simdcommon'), 5, false)->'results' AS precompute_wand_daat_parity;

RESET hybrid.debug_postings_chunk_size;
DROP TABLE tqh_simd_bm25_docs;
