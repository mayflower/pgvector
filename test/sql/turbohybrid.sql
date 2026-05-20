CREATE TABLE tqh_empty_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

CREATE INDEX tqh_empty_docs_idx ON tqh_empty_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

SELECT
	tq_debug_bm25_stats('tqh_empty_docs_idx'::regclass)->>'doc_count' AS empty_doc_count,
	tq_debug_bm25_stats('tqh_empty_docs_idx'::regclass)->>'unique_terms' AS empty_unique_terms,
	(tq_debug_bm25_stats('tqh_empty_docs_idx'::regclass)->>'meta_blkno')::int > 0 AS empty_has_meta;

DROP TABLE tqh_empty_docs;

CREATE TABLE tqh_docs (
	id int,
	tenant_id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_docs VALUES
	(0, 10, '[0,0,0]', to_tsvector('english', 'orphan lexical')),
	(1, 10, '[1,0,0]', to_tsvector('english', 'postgres vector search')),
	(2, 10, '[1,1,0]', to_tsvector('english', 'hybrid search')),
	(3, 20, '[0,1,0]', to_tsvector('english', 'lexical search'));

CREATE INDEX tqh_docs_idx ON tqh_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
INCLUDE (tenant_id);

CREATE TEMP TABLE tqh_bm25_meta_before AS
SELECT tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'meta_blkno' AS meta_blkno;

SELECT
	((SELECT meta_blkno FROM tqh_bm25_meta_before))::int > 0 AS has_bm25_meta_pointer,
	tq_index_stats('tqh_docs_idx'::regclass)->>'bm25_meta_start_block' =
		(SELECT meta_blkno FROM tqh_bm25_meta_before) AS stats_uses_bm25_meta_pointer;

SELECT
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'doc_count' AS doc_count,
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'total_doc_len' AS total_doc_len,
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'unique_terms' AS unique_terms,
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'term_tuple_count' AS term_tuple_count,
	(tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'lexicon_start_blkno')::int > 0 AS has_lexicon,
	(tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'postings_start_blkno')::int > 0 AS has_postings,
	(tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'blockmax_start_blkno')::int > 0 AS has_blockmax,
	(tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'postings_pages')::int > 0 AS has_postings_pages,
	(tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'blockmax_pages')::int > 0 AS has_blockmax_pages,
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'bm25_precompute_tf_norm' AS precompute_tf_norm_default;

SELECT
	tq_debug_bm25_term_stats('tqh_docs_idx'::regclass, 'search')->>'df' AS df,
	tq_debug_bm25_term_stats('tqh_docs_idx'::regclass, 'search')->>'cf' AS cf,
	tq_debug_bm25_term_stats('tqh_docs_idx'::regclass, 'search')->>'posting_count' AS posting_count,
	(tq_debug_bm25_term_stats('tqh_docs_idx'::regclass, 'search')->>'postings_blkno')::int > 0 AS has_postings,
	(tq_debug_bm25_term_stats('tqh_docs_idx'::regclass, 'search')->>'blockmax_blkno')::int > 0 AS has_blockmax;

SELECT
	tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'postgres'), 5, false)->>'result_count' AS result_count,
	tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'postgres'), 5, false)->'results'->0->>'rank' AS first_rank,
	(tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'postgres'), 5, false)->'results'->0->>'score_scaled')::int > 0 AS has_score;

SELECT
	tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'postgres & vector'), 5, false)->>'result_count' AS and_count,
	tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'postgres | lexical'), 5, false)->>'result_count' AS or_count,
	tq_debug_bm25_topk('tqh_docs_idx'::regclass, websearch_to_tsquery('english', 'postgres vector'), 5, false)->>'result_count' AS websearch_count;

SET hybrid.bm25_force_full_sort = on;
CREATE TEMP TABLE tqh_bm25_full_sort_results AS
SELECT tq_debug_bm25_topk('tqh_docs_idx'::regclass,
	to_tsquery('english', 'postgres | lexical'), 2, false)->'results' AS results;
RESET hybrid.bm25_force_full_sort;

WITH bounded AS MATERIALIZED (
	SELECT tq_debug_bm25_topk('tqh_docs_idx'::regclass,
		to_tsquery('english', 'postgres | lexical'), 2, false) AS j
)
SELECT
	(j->'results') = (SELECT results FROM tqh_bm25_full_sort_results) AS bounded_matches_full_sort,
	(j->>'full_sort_avoided')::bool AS full_sort_avoided,
	(j->>'final_sorted_count')::int <= 2 AS bounded_final_sorted
FROM bounded;

DROP TABLE tqh_bm25_full_sort_results;

SET hybrid.bm25_accumulator_mode = hash;
CREATE TEMP TABLE tqh_bm25_hash_accumulator_results AS
SELECT
	tq_debug_bm25_topk('tqh_docs_idx'::regclass,
		to_tsquery('english', 'postgres | lexical'), 3, false)->'results' AS or_results,
	tq_debug_bm25_topk('tqh_docs_idx'::regclass,
		to_tsquery('english', 'postgres & vector'), 3, false)->'results' AS and_results;
SET hybrid.bm25_accumulator_mode = node_generation_arrays;

WITH dense AS MATERIALIZED (
	SELECT tq_debug_bm25_topk('tqh_docs_idx'::regclass,
		to_tsquery('english', 'postgres | lexical'), 3, false) AS or_stats,
		tq_debug_bm25_topk('tqh_docs_idx'::regclass,
		to_tsquery('english', 'postgres & vector'), 3, false) AS and_stats
)
SELECT
	(or_stats->'results') = (SELECT or_results FROM tqh_bm25_hash_accumulator_results) AS dense_or_matches_hash,
	(and_stats->'results') = (SELECT and_results FROM tqh_bm25_hash_accumulator_results) AS dense_and_matches_hash,
	or_stats->>'accumulator_mode' AS accumulator_mode,
	(or_stats->>'accumulator_dense_updates')::int > 0 AS has_dense_updates
FROM dense;

RESET hybrid.bm25_accumulator_mode;
DROP TABLE tqh_bm25_hash_accumulator_results;

SELECT
	tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'search'), 2, true)->>'result_count' AS result_count,
	tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'search'), 2, true)->>'used_wand' AS used_wand,
	tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'search'), 2, true)->>'postings_decoded' AS postings_decoded,
	(tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'search'), 2, true)->>'blocks_visited')::int > 0 AS has_blocks;

SELECT
	(tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'postgres | hybrid'), 5, true)->'results') =
	(tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'postgres | hybrid'), 5, false)->'results') AS wand_two_term_matches_daat,
	(tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'postgres | hybrid | lexical'), 5, true)->'results') =
	(tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'postgres | hybrid | lexical'), 5, false)->'results') AS wand_three_term_matches_daat;

SELECT tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'missing'), 5, false)->>'result_count' AS missing_count;

CREATE TABLE tqh_lazy_cache_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_lazy_cache_docs VALUES
	(1, '[1,0,0]', to_tsvector('english', 'alpha')),
	(2, '[0,1,0]', to_tsvector('english', 'beta'));

CREATE INDEX tqh_lazy_cache_docs_idx ON tqh_lazy_cache_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

WITH stats AS MATERIALIZED (
	SELECT tq_debug_bm25_topk('tqh_lazy_cache_docs_idx'::regclass, to_tsquery('english', 'missing'), 5, false) AS j
)
SELECT
	j->>'result_count' AS lazy_missing_count,
	(j->>'cache_docstats_loaded')::bool AS lazy_docstats_loaded,
	(j->>'cache_liveness_loaded')::bool AS lazy_liveness_loaded,
	(j->>'cache_lexicon_entries')::int > 0 AS lazy_loaded_lexicon
FROM stats;

DROP TABLE tqh_lazy_cache_docs;

SELECT tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'orphan'), 5, false)->>'result_count' AS skipped_dense_count;

SET enable_seqscan = off;

SELECT id
FROM tqh_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[1,0,0]'::vector,
	text_query => websearch_to_tsquery('english', 'postgres vector'),
	dense_k => 3,
	bm25_k => 3
)
LIMIT 2;

RESET enable_seqscan;

CREATE TABLE tqh_final_k_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_final_k_docs
SELECT i, ('[' || i || ',0,0]')::vector, to_tsvector('english', 'common token')
FROM generate_series(1, 30) i;

CREATE INDEX tqh_final_k_docs_idx ON tqh_final_k_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

SET enable_seqscan = off;

SELECT count(*) AS omitted_final_k_limit_count
FROM (
	SELECT id
	FROM tqh_final_k_docs
	ORDER BY embedding <~> hybrid_query(
		text_query => to_tsquery('english', 'common'),
		dense_k => 0,
		bm25_k => 30
	)
	LIMIT 25
) s;

RESET enable_seqscan;

DROP TABLE tqh_final_k_docs;

CREATE TABLE tqh_sparse_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_sparse_docs
SELECT i,
	format('[%s,%s,%s]', i + 1, (i % 5) + 1, (i % 7) + 1)::vector,
	to_tsvector('english', CASE WHEN i = 77 THEN 'needle rare token' ELSE 'common filler token' END)
FROM generate_series(1, 128) i;

CREATE INDEX tqh_sparse_docs_idx ON tqh_sparse_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

SET enable_seqscan = off;

SELECT id
FROM tqh_sparse_docs
ORDER BY embedding <~> hybrid_query(
	text_query => websearch_to_tsquery('english', 'needle'),
	dense_k => 0,
	bm25_k => 5
)
LIMIT 1;

RESET enable_seqscan;

SELECT
	tq_debug_bm25_topk('tqh_sparse_docs_idx'::regclass, to_tsquery('english', 'needle'), 5, false)->>'result_count' AS rare_count,
	(tq_debug_bm25_topk('tqh_sparse_docs_idx'::regclass, to_tsquery('english', 'needle'), 5, false)->>'accumulator_entries')::int <= 2 AS sparse_accumulator;

DROP TABLE tqh_sparse_docs;

SET hybrid.debug_postings_chunk_size = 3;

CREATE TABLE tqh_chunk_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_chunk_docs
SELECT i,
	format('[%s,%s,%s]', i + 1, (i % 3) + 1, (i % 5) + 1)::vector,
	to_tsvector('english', 'chunkcommon token')
FROM generate_series(1, 12) i;

CREATE INDEX tqh_chunk_docs_idx ON tqh_chunk_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

SELECT
	tq_debug_bm25_term_stats('tqh_chunk_docs_idx'::regclass, 'chunkcommon')->>'posting_count' AS posting_count,
	tq_debug_bm25_term_stats('tqh_chunk_docs_idx'::regclass, 'chunkcommon')->>'postings_chunk_count' AS chunk_count,
	(tq_debug_bm25_stats('tqh_chunk_docs_idx'::regclass)->>'postings_pages')::int > 0 AS has_postings_pages,
	(tq_debug_bm25_stats('tqh_chunk_docs_idx'::regclass)->>'blockmax_pages')::int > 0 AS has_blockmax_pages,
	tq_debug_bm25_topk('tqh_chunk_docs_idx'::regclass, to_tsquery('english', 'chunkcommon'), 12, false)->>'result_count' AS result_count;

DROP TABLE tqh_chunk_docs;

RESET hybrid.debug_postings_chunk_size;

DO $$
BEGIN
	PERFORM tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', '!search'), 5, false);
	RAISE EXCEPTION 'expected NOT tsquery to be guarded';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

DO $$
BEGIN
	PERFORM tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'search:*'), 5, false);
	RAISE EXCEPTION 'expected prefix tsquery to be guarded';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

DO $$
BEGIN
	PERFORM tq_debug_bm25_topk('tqh_docs_idx'::regclass, phraseto_tsquery('english', 'postgres vector'), 5, false);
	RAISE EXCEPTION 'expected phrase tsquery to be guarded';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

SET enable_seqscan = off;

SELECT id
FROM tqh_docs
ORDER BY embedding <~> hybrid_query(vector_query => '[1,0,0]'::vector)
LIMIT 3;

SELECT id
FROM tqh_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[1,0,0]'::vector,
	text_query => websearch_to_tsquery('english', 'postgres'),
	dense_k => 3,
	bm25_k => 3
)
LIMIT 1;

DO $$
BEGIN
	PERFORM id,
		embedding <~> hybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'postgres'),
			dense_k => 3,
			bm25_k => 3
		) AS score
	FROM tqh_docs
	ORDER BY embedding <~> hybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres'),
		dense_k => 3,
		bm25_k => 3
	)
	LIMIT 1;
	RAISE EXCEPTION 'expected projected hybrid text score rejection';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

SELECT id
FROM tqh_docs
ORDER BY embedding <~> hybrid_query(
	text_query => websearch_to_tsquery('english', 'postgres'),
	dense_k => 0,
	bm25_k => 3
)
LIMIT 1;

SELECT count(*)
FROM (
	SELECT id
	FROM tqh_docs
	ORDER BY embedding <~> hybrid_query(
		vector_query => '[0,1,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres'),
		require_bm25_match => true,
		dense_k => 3,
		bm25_k => 3
	)
	LIMIT 3
) s;

SELECT
	hybrid_last_scan_stats()->>'fusion' AS fusion,
	(hybrid_last_scan_stats()->>'dense_candidates')::int > 0 AS has_dense,
	(hybrid_last_scan_stats()->>'bm25_candidates')::int > 0 AS has_bm25,
	(hybrid_last_scan_stats()->>'union_candidates')::int > 0 AS has_union,
	(hybrid_last_scan_stats()->>'final_results')::int > 0 AS has_final_results,
	(hybrid_last_scan_stats()->>'bm25_postings_decoded')::int > 0 AS decoded_postings,
	(hybrid_last_scan_stats()->>'bm25_cache_bytes')::bigint > 0 AS reports_cache_bytes,
	(hybrid_last_scan_stats()->>'bm25_cache_lexicon_entries')::int > 0 AS reports_cache_terms,
	(hybrid_last_scan_stats()->>'bm25_cache_hit') IS NOT NULL AS reports_cache_hit,
	(hybrid_last_scan_stats()->>'bm25_cache_build_us') IS NOT NULL AS reports_cache_build_us,
	(hybrid_last_scan_stats()->>'bm25_cache_docstats_loaded')::bool AS cache_docstats_loaded,
	(hybrid_last_scan_stats()->>'bm25_cache_liveness_loaded')::bool AS cache_liveness_loaded,
	(hybrid_last_scan_stats()->>'elapsed_us')::int > 0 AS has_elapsed;

SET hybrid.fusion_hash_threshold = -1;
CREATE TEMP TABLE tqh_sort_fusion AS
SELECT id
FROM tqh_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[1,0,0]'::vector,
	text_query => websearch_to_tsquery('english', 'postgres vector lexical'),
	dense_k => 3,
	bm25_k => 3,
	final_k => 2
)
LIMIT 2;
SELECT hybrid_last_scan_stats()->>'fusion_strategy' AS sort_strategy;

SET hybrid.fusion_hash_threshold = 0;
CREATE TEMP TABLE tqh_hash_fusion AS
SELECT id
FROM tqh_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[1,0,0]'::vector,
	text_query => websearch_to_tsquery('english', 'postgres vector lexical'),
	dense_k => 3,
	bm25_k => 3,
	final_k => 2
)
LIMIT 2;
SELECT
	hybrid_last_scan_stats()->>'fusion_strategy' AS hash_strategy,
	(hybrid_last_scan_stats()->>'fusion_candidates_seen')::int > 0 AS saw_candidates,
	(hybrid_last_scan_stats()->>'fusion_heap_size')::int = 2 AS bounded_heap;

SELECT
	(SELECT array_agg(id) FROM tqh_sort_fusion) =
	(SELECT array_agg(id) FROM tqh_hash_fusion) AS hash_matches_sort;
DROP TABLE tqh_sort_fusion;
DROP TABLE tqh_hash_fusion;
RESET hybrid.fusion_hash_threshold;

DO $$
BEGIN
	PERFORM hybrid_distance('[1,0,0]'::vector, hybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => to_tsquery('english', 'postgres')
	));
	RAISE EXCEPTION 'expected stale hybrid distance rejection';
EXCEPTION
	WHEN feature_not_supported THEN
		NULL;
END
$$;

SET hybrid.enable_exact_rescore_for_bm25_only = on;
DO $$
BEGIN
	PERFORM id
	FROM tqh_docs
	ORDER BY embedding <~> hybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'lexical'),
		fusion => 'weighted',
		dense_k => 0,
		bm25_k => 3
	)
	LIMIT 1;
	RAISE EXCEPTION 'expected BM25-only exact rescore guard';
EXCEPTION WHEN feature_not_supported THEN
END
$$;
RESET hybrid.enable_exact_rescore_for_bm25_only;

INSERT INTO tqh_docs VALUES
	(4, 30, '[0,0,1]', to_tsvector('english', 'new lexical document'));

SELECT
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'delta_doc_count' AS delta_doc_count,
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'delta_term_count' AS delta_term_count;

SET hybrid.bm25_cache_max_mb = 1;
SET hybrid.bm25_accumulator_mode = hash;
CREATE TEMP TABLE tqh_delta_hash_accumulator_results AS
SELECT tq_debug_bm25_topk('tqh_docs_idx'::regclass,
	to_tsquery('english', 'new | lexical'), 3, false)->'results' AS results;
SET hybrid.bm25_accumulator_mode = node_generation_arrays;

WITH dense AS MATERIALIZED (
	SELECT tq_debug_bm25_topk('tqh_docs_idx'::regclass,
		to_tsquery('english', 'new | lexical'), 3, false) AS stats
)
SELECT
	(stats->'results') = (SELECT results FROM tqh_delta_hash_accumulator_results) AS dense_delta_matches_hash,
	stats->>'accumulator_mode' AS accumulator_mode,
	(stats->>'delta_postings_decoded')::int > 0 AS decoded_delta_postings
FROM dense;

RESET hybrid.bm25_accumulator_mode;
RESET hybrid.bm25_cache_max_mb;
DROP TABLE tqh_delta_hash_accumulator_results;

CREATE TEMP TABLE tqh_delta_cache_probe AS
SELECT tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'new'), 3, false) AS stats;
INSERT INTO tqh_delta_cache_probe
SELECT tq_debug_bm25_topk('tqh_docs_idx'::regclass, to_tsquery('english', 'new'), 3, false);

SELECT
	(stats->>'delta_cache_hit')::bool AS delta_cache_hit,
	(stats->>'delta_cache_terms')::int > 0 AS has_delta_cache_terms,
	(stats->>'delta_postings_decoded')::int > 0 AS decoded_delta_postings,
	(stats->>'delta_blocks_visited')::int AS delta_blocks_visited
FROM tqh_delta_cache_probe
ORDER BY delta_cache_hit;

DROP TABLE tqh_delta_cache_probe;

CREATE TABLE tqh_delta_limit_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_delta_limit_docs VALUES
	(1, '[1,0,0]', to_tsvector('english', 'base lexical'));

CREATE INDEX tqh_delta_limit_docs_idx ON tqh_delta_limit_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

INSERT INTO tqh_delta_limit_docs VALUES
	(2, '[0,1,0]', to_tsvector('english', 'bounded delta cache'));

SET hybrid.bm25_cache_max_mb = 1;

CREATE TEMP TABLE tqh_delta_limit_probe AS
SELECT tq_debug_bm25_topk('tqh_delta_limit_docs_idx'::regclass, to_tsquery('english', 'bounded'), 3, false) AS stats;
INSERT INTO tqh_delta_limit_probe
SELECT tq_debug_bm25_topk('tqh_delta_limit_docs_idx'::regclass, to_tsquery('english', 'bounded'), 3, false);

SELECT
	bool_or((stats->>'delta_cache_hit')::bool) AS any_delta_cache_hit,
	bool_and((stats->>'delta_cache_terms')::int = 1) AS only_query_delta_term,
	bool_and((stats->>'delta_cache_bytes')::int > 0) AS has_query_delta_bytes,
	bool_and((stats->>'delta_blocks_visited')::int > 0) AS scanned_delta_each_time
FROM tqh_delta_limit_probe;

RESET hybrid.bm25_cache_max_mb;

DROP TABLE tqh_delta_limit_probe;
DROP TABLE tqh_delta_limit_docs;

SELECT id
FROM tqh_docs
ORDER BY embedding <~> hybrid_query(
	text_query => websearch_to_tsquery('english', 'new lexical'),
	dense_k => 0,
	bm25_k => 3
)
LIMIT 1;

SELECT id
FROM tqh_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[0,0,1]'::vector,
	text_query => websearch_to_tsquery('english', 'new lexical'),
	dense_k => 3,
	bm25_k => 3
)
LIMIT 1;

DELETE FROM tqh_docs WHERE id = 4;
VACUUM tqh_docs;

SELECT count(*)
FROM (
	SELECT id
	FROM tqh_docs
	ORDER BY embedding <~> hybrid_query(
		text_query => websearch_to_tsquery('english', 'new'),
		dense_k => 0,
		bm25_k => 3
	)
	LIMIT 3
) s;

SELECT
	tq_index_stats('tqh_docs_idx'::regclass)->>'hybrid' AS hybrid,
	(tq_index_stats('tqh_docs_idx'::regclass)->>'bm25_live_doc_count')::int >= 3 AS has_live_bm25,
	(tq_index_stats('tqh_docs_idx'::regclass)->>'bm25_dead_doc_count')::int = 0 AS compacted_dead_bm25,
	(tq_index_stats('tqh_docs_idx'::regclass)->>'bm25_delta_generation')::int > 0 AS has_delta_generation,
	tq_index_stats('tqh_docs_idx'::regclass)->>'bm25_last_compaction' <> 'never' AS has_compaction;

SELECT
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'delta_doc_count' AS compacted_delta_docs,
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'compaction_count' AS compaction_count,
	(tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'postings_pages')::int > 0 AS compacted_postings_pages,
	(tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'blockmax_pages')::int > 0 AS compacted_blockmax_pages,
	tq_debug_bm25_stats('tqh_docs_idx'::regclass)->>'meta_blkno' =
		(SELECT meta_blkno FROM tqh_bm25_meta_before) AS bm25_meta_pointer_stable;

RESET enable_seqscan;

CREATE TABLE tqh_wand_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);
INSERT INTO tqh_wand_docs
SELECT i, '[1,0,0]'::vector, to_tsvector('english', 'common')
FROM generate_series(1, 10) i;
CREATE INDEX tqh_wand_docs_idx ON tqh_wand_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (
	bm25_block_max = on,
	bm25_delta_compaction_threshold = 10,
	hybrid_default_fusion = 'rrf',
	hybrid_default_dense_k = 8,
	hybrid_default_bm25_k = 8,
	hybrid_default_rrf_k = 60
);

SELECT
	tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, true)->>'result_count' AS result_count,
	(tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, true)->>'blocks_skipped')::int > 0 AS skipped_blocks,
	(tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, true)->>'postings_decoded')::int < 10 AS decoded_less_than_full,
	(tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, true)->>'wand_iterations')::int > 0 AS wand_iterations,
	(tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, true)->>'wand_active_sorts')::int > 0 AS wand_active_sorts,
	(tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, true)->>'wand_threshold_updates')::int > 0 AS wand_threshold_updates;

SELECT
	(tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, true)->'results') =
	(tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, false)->'results') AS wand_matches_daat;

SET hybrid.enable_wand = off;
SELECT
	tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, true)->>'used_wand' AS used_wand,
	tq_debug_bm25_topk('tqh_wand_docs_idx'::regclass, to_tsquery('english', 'common'), 2, true)->>'postings_decoded' AS postings_decoded;
RESET hybrid.enable_wand;

DROP TABLE tqh_wand_docs;

CREATE TABLE tqh_impact_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);
INSERT INTO tqh_impact_docs
SELECT i,
	format('[%s,%s,%s]', i, i % 3, 0)::vector,
	to_tsvector('english',
		CASE WHEN i <= 12 THEN 'common impact' || repeat(' filler', i) ELSE 'other impact' END)
FROM generate_series(1, 20) i;
CREATE INDEX tqh_impact_docs_idx ON tqh_impact_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (
	bm25_impact_head = on,
	bm25_impact_min_df = 4,
	bm25_impact_head_k = 8
);

SET hybrid.bm25_strategy = daat_hash;
CREATE TEMP TABLE tqh_impact_daat AS
SELECT tq_debug_bm25_topk('tqh_impact_docs_idx'::regclass,
	to_tsquery('english', 'common'), 4, false)->'results' AS results;
SET hybrid.bm25_strategy = impact;
SELECT
	(tq_debug_bm25_term_stats('tqh_impact_docs_idx'::regclass, 'common')->>'bm25_impact_head_stored')::bool AS stored_impact_head,
	(tq_debug_bm25_term_stats('tqh_impact_docs_idx'::regclass, 'common')->>'impact_count')::int = 8 AS stored_impact_count;
WITH impact AS MATERIALIZED (
	SELECT tq_debug_bm25_topk('tqh_impact_docs_idx'::regclass,
		to_tsquery('english', 'common'), 4, false) AS j
)
SELECT
	j->>'strategy' AS strategy,
	(j->'results') = (SELECT results FROM tqh_impact_daat) AS impact_matches_daat,
	(j->>'impact_terms')::int = 1 AS impact_terms,
	(j->>'impact_postings_read')::int = 4 AS bounded_impact_reads,
	(j->>'impact_full_postings_avoided')::bool AS avoided_full_postings
FROM impact;

SET hybrid.bm25_strategy = daat_hash;
CREATE TEMP TABLE tqh_impact_or_daat AS
SELECT tq_debug_bm25_topk('tqh_impact_docs_idx'::regclass,
	to_tsquery('english', 'common | impact'), 6, false)->'results' AS results;
SET hybrid.bm25_strategy = impact;
WITH impact AS MATERIALIZED (
	SELECT tq_debug_bm25_topk('tqh_impact_docs_idx'::regclass,
		to_tsquery('english', 'common | impact'), 6, true) AS j
)
SELECT
	j->>'strategy' AS strategy,
	(j->'results') = (SELECT results FROM tqh_impact_or_daat) AS impact_or_matches_daat,
	(j->>'impact_terms')::int = 2 AS seeded_impact_terms,
	(j->>'impact_postings_read')::int > 0 AS seeded_impact_reads
FROM impact;

INSERT INTO tqh_impact_docs VALUES
	(21, '[21,0,0]'::vector, to_tsvector('english', 'common common common common common delta'));
SET hybrid.bm25_strategy = daat_hash;
CREATE TEMP TABLE tqh_impact_delta_daat AS
SELECT tq_debug_bm25_topk('tqh_impact_docs_idx'::regclass,
	to_tsquery('english', 'common'), 4, false)->'results' AS results;
SET hybrid.bm25_strategy = impact;
WITH impact AS MATERIALIZED (
	SELECT tq_debug_bm25_topk('tqh_impact_docs_idx'::regclass,
		to_tsquery('english', 'common'), 4, false) AS j
)
SELECT
	j->>'strategy' AS strategy,
	(j->'results') = (SELECT results FROM tqh_impact_delta_daat) AS impact_delta_matches_daat,
	(j->>'delta_postings_decoded')::int > 0 AS impact_read_delta
FROM impact;
RESET hybrid.bm25_strategy;
DROP TABLE tqh_impact_daat;
DROP TABLE tqh_impact_or_daat;
DROP TABLE tqh_impact_delta_daat;
DROP TABLE tqh_impact_docs;

CREATE TABLE tqh_impact_chunk_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);
INSERT INTO tqh_impact_chunk_docs
SELECT i,
	format('[%s,%s,%s]', i, i % 5, i % 7)::vector,
	to_tsvector('english', 'chunkimpact' || repeat(' filler', i % 11))
FROM generate_series(1, 512) i;
CREATE INDEX tqh_impact_chunk_docs_idx ON tqh_impact_chunk_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (
	bm25_impact_head = on,
	bm25_impact_min_df = 4,
	bm25_impact_head_k = 512
);
SELECT
	(tq_debug_bm25_term_stats('tqh_impact_chunk_docs_idx'::regclass, 'chunkimpact')->>'bm25_impact_head_stored')::bool AS stored_chunked_head,
	(tq_debug_bm25_term_stats('tqh_impact_chunk_docs_idx'::regclass, 'chunkimpact')->>'impact_count')::int = 512 AS stored_chunked_count;
SET hybrid.bm25_strategy = daat_hash;
CREATE TEMP TABLE tqh_impact_chunk_daat AS
SELECT tq_debug_bm25_topk('tqh_impact_chunk_docs_idx'::regclass,
	to_tsquery('english', 'chunkimpact'), 20, false)->'results' AS results;
SET hybrid.bm25_strategy = impact;
WITH impact AS MATERIALIZED (
	SELECT tq_debug_bm25_topk('tqh_impact_chunk_docs_idx'::regclass,
		to_tsquery('english', 'chunkimpact'), 20, false) AS j
)
SELECT
	j->>'strategy' AS strategy,
	(j->'results') = (SELECT results FROM tqh_impact_chunk_daat) AS chunked_impact_matches_daat,
	(j->>'impact_postings_read')::int = 20 AS bounded_chunked_reads,
	(j->>'impact_full_postings_avoided')::bool AS avoided_chunked_full_postings
FROM impact;
RESET hybrid.bm25_strategy;
DROP TABLE tqh_impact_chunk_daat;
DROP TABLE tqh_impact_chunk_docs;

CREATE TABLE tqh_bits_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);
INSERT INTO tqh_bits_docs
SELECT i,
	format('[%s,%s,%s]', i, i + 1, i + 2)::vector,
	to_tsvector('english', 'bit width dense path')
FROM generate_series(1, 8) i;

SET enable_seqscan = off;

CREATE INDEX tqh_bits_docs_1_idx ON tqh_bits_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (tq_bits = 1);

SELECT tq_index_stats('tqh_bits_docs_1_idx'::regclass)->>'tq_bits' AS tq_bits;

SELECT id
FROM tqh_bits_docs
ORDER BY embedding <~> hybrid_query(vector_query => '[1,2,3]'::vector)
LIMIT 1;

SELECT tq_last_scan_stats()->>'graph_tq_bits' AS scan_tq_bits;

DROP INDEX tqh_bits_docs_1_idx;

CREATE INDEX tqh_bits_docs_2_idx ON tqh_bits_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (tq_bits = 2);

SELECT tq_index_stats('tqh_bits_docs_2_idx'::regclass)->>'tq_bits' AS tq_bits;

SELECT id
FROM tqh_bits_docs
ORDER BY embedding <~> hybrid_query(vector_query => '[1,2,3]'::vector)
LIMIT 1;

SELECT tq_last_scan_stats()->>'graph_tq_bits' AS scan_tq_bits;

DROP INDEX tqh_bits_docs_2_idx;

CREATE INDEX tqh_bits_docs_4_idx ON tqh_bits_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (tq_bits = 4);

SELECT tq_index_stats('tqh_bits_docs_4_idx'::regclass)->>'tq_bits' AS tq_bits;

SELECT id
FROM tqh_bits_docs
ORDER BY embedding <~> hybrid_query(vector_query => '[1,2,3]'::vector)
LIMIT 1;

SELECT tq_last_scan_stats()->>'graph_tq_bits' AS scan_tq_bits;

DROP TABLE tqh_bits_docs;

RESET enable_seqscan;

CREATE TABLE tqh_dense_budget_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);
INSERT INTO tqh_dense_budget_docs
SELECT i,
	format('[%s,%s,%s]', i + 1, (i % 7) + 1, (i % 11) + 1)::vector,
	to_tsvector('english', 'dense budget row')
FROM generate_series(1, 50) i;

CREATE INDEX tqh_dense_budget_idx ON tqh_dense_budget_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

SET enable_seqscan = off;

SELECT count(*)
FROM (
	SELECT id
	FROM tqh_dense_budget_docs
	ORDER BY embedding <~> hybrid_query(
		vector_query => '[1,1,1]'::vector,
		text_query => websearch_to_tsquery('english', 'absent'),
		dense_k => 7,
		bm25_k => 0
	)
	LIMIT 3
) s;

SELECT
	hybrid_last_scan_stats()->>'dense_candidates_requested' AS dense_requested,
	hybrid_last_scan_stats()->>'dense_candidates_effective' AS dense_effective,
	(hybrid_last_scan_stats()->>'dense_k_defaulted')::bool AS dense_defaulted,
	hybrid_last_scan_stats()->>'dense_candidates' AS dense_returned,
	hybrid_last_scan_stats()->>'bm25_candidates_requested' AS bm25_requested,
	hybrid_last_scan_stats()->>'bm25_candidates_effective' AS bm25_effective,
	(hybrid_last_scan_stats()->>'bm25_k_defaulted')::bool AS bm25_defaulted,
	hybrid_last_scan_stats()->>'bm25_candidates' AS bm25_returned;

SELECT count(*)
FROM (
	SELECT id
	FROM tqh_dense_budget_docs
	ORDER BY embedding <~> hybrid_query(
		vector_query => '[1,1,1]'::vector,
		text_query => websearch_to_tsquery('english', 'absent')
	)
	LIMIT 3
) s;

SELECT
	(hybrid_last_scan_stats()->>'dense_candidates_requested')::int = 400 AS dense_requested_default,
	(hybrid_last_scan_stats()->>'dense_candidates_effective')::int = 32 AS dense_effective_limit,
	(hybrid_last_scan_stats()->>'dense_k_defaulted')::bool AS dense_defaulted,
	(hybrid_last_scan_stats()->>'bm25_candidates_requested')::int = 400 AS bm25_requested_default,
	(hybrid_last_scan_stats()->>'bm25_candidates_effective')::int = 32 AS bm25_effective_limit,
	(hybrid_last_scan_stats()->>'bm25_k_defaulted')::bool AS bm25_defaulted,
	(hybrid_last_scan_stats()->>'rrf_k_requested')::int = 60 AS rrf_requested_default,
	(hybrid_last_scan_stats()->>'rrf_k_effective')::int = 60 AS rrf_effective_preserved,
	(hybrid_last_scan_stats()->>'rrf_k_defaulted')::bool AS rrf_defaulted,
	(hybrid_last_scan_stats()->>'auto_budget_limit')::int = 3 AS detected_limit;

SELECT count(*)
FROM (
	SELECT id
	FROM tqh_dense_budget_docs
	ORDER BY embedding <~> hybrid_query(
		vector_query => '[1,1,1]'::vector,
		text_query => websearch_to_tsquery('english', 'absent'),
		dense_k => 25,
		bm25_k => 0
	)
	LIMIT 3
) s;

SELECT
	hybrid_last_scan_stats()->>'dense_candidates_requested' AS dense_requested,
	hybrid_last_scan_stats()->>'dense_candidates_effective' AS dense_effective,
	(hybrid_last_scan_stats()->>'dense_k_defaulted')::bool AS dense_defaulted,
	hybrid_last_scan_stats()->>'dense_candidates' AS dense_returned;

SELECT count(*)
FROM (
	SELECT id
	FROM tqh_dense_budget_docs
	ORDER BY embedding <~> hybrid_query(
		vector_query => '[1,1,1]'::vector,
		text_query => websearch_to_tsquery('english', 'absent'),
		dense_k => 0,
		bm25_k => 0
	)
	LIMIT 3
) s;

SELECT
	hybrid_last_scan_stats()->>'dense_candidates_requested' AS dense_requested,
	hybrid_last_scan_stats()->>'dense_candidates' AS dense_returned;

RESET enable_seqscan;

DROP TABLE tqh_dense_budget_docs;

CREATE TABLE tqh_dense_policy_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);
INSERT INTO tqh_dense_policy_docs
SELECT i,
	format('[%s,%s,%s]', i + 1, (i % 7) + 1, (i % 11) + 1)::vector,
	to_tsvector('english', 'dense policy row')
FROM generate_series(1, 180) i;

CREATE INDEX tqh_dense_policy_idx ON tqh_dense_policy_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (
	routing = graph,
	graph_ef_search = 64,
	graph_oversampling = 4
);

SET enable_seqscan = off;
SET hnsw.tq_dense_budget_policy = latency;
SET hnsw.tq_rescore_band_policy = limited;

SELECT count(*)
FROM (
	SELECT id
	FROM tqh_dense_policy_docs
	ORDER BY embedding <~> hybrid_query(
		vector_query => '[1,1,1]'::vector,
		text_query => to_tsquery('english', 'absent'),
		dense_k => 100,
		bm25_k => 0,
		final_k => 10
	)
	LIMIT 10
) s;

SELECT
	(hybrid_last_scan_stats()->>'dense_effective_result_target')::int <= 150 AS latency_target_capped,
	(hybrid_last_scan_stats()->>'dense_effective_rescore_band')::int <= 200 AS latency_rescore_capped,
	hybrid_last_scan_stats()->>'dense_budget_policy' AS dense_budget_policy,
	hybrid_last_scan_stats()->>'dense_rescore_band_policy' AS rescore_policy;

SELECT
	hybrid_last_scan_stats() ? 'dense_entry_us' AS has_entry_timer,
	hybrid_last_scan_stats() ? 'dense_base_us' AS has_base_timer,
	hybrid_last_scan_stats() ? 'dense_batch_us' AS has_batch_timer,
	hybrid_last_scan_stats() ? 'dense_heap_us' AS has_heap_timer;

SET hnsw.tq_dense_budget_policy = quality;
SET hnsw.tq_rescore_band_policy = exact;

SELECT count(*)
FROM (
	SELECT id
	FROM tqh_dense_policy_docs
	ORDER BY embedding <~> hybrid_query(
		vector_query => '[1,1,1]'::vector,
		text_query => to_tsquery('english', 'absent'),
		dense_k => 100,
		bm25_k => 0,
		final_k => 10
	)
	LIMIT 10
) s;

SELECT
	(hybrid_last_scan_stats()->>'dense_effective_result_target')::int >= 100 AS quality_preserves_target,
	(hybrid_last_scan_stats()->>'dense_effective_rescore_band')::int >= 100 AS quality_preserves_rescore,
	hybrid_last_scan_stats()->>'dense_budget_policy' AS dense_budget_policy,
	hybrid_last_scan_stats()->>'dense_rescore_band_policy' AS rescore_policy;

RESET hnsw.tq_dense_budget_policy;
RESET hnsw.tq_rescore_band_policy;
RESET enable_seqscan;

DROP TABLE tqh_dense_policy_docs;

CREATE TABLE tqh_ir_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_ir_docs VALUES
	(1, '[0.95,0.05,0]', to_tsvector('english', 'alpha exact semantic')),
	(2, '[0,1,0]', to_tsvector('english', 'alpha exact lexical')),
	(3, '[1,0,0]', to_tsvector('english', 'semantic neighbor')),
	(4, '[0,0,1]', to_tsvector('english', 'unrelated branch'));

CREATE INDEX tqh_ir_docs_idx ON tqh_ir_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

SET enable_seqscan = off;

DO $$
DECLARE
	line text;
	uses_turbohybrid bool := false;
BEGIN
	FOR line IN
		EXPLAIN (COSTS OFF)
		SELECT id
		FROM tqh_ir_docs
		ORDER BY embedding <~> hybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'alpha'),
			dense_k => 3,
			bm25_k => 3
		)
		LIMIT 3
	LOOP
		uses_turbohybrid := uses_turbohybrid OR line LIKE '%Index Scan using tqh_ir_docs_idx%';
	END LOOP;
	IF NOT uses_turbohybrid THEN
		RAISE EXCEPTION 'expected planner to choose turbohybrid index scan';
	END IF;
END
$$;

CREATE TEMP TABLE tqh_ir_dense AS
SELECT id
FROM tqh_ir_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[1,0,0]'::vector,
	dense_k => 3,
	bm25_k => 0
)
LIMIT 3;

CREATE TEMP TABLE tqh_ir_bm25 AS
SELECT id
FROM tqh_ir_docs
ORDER BY embedding <~> hybrid_query(
	text_query => websearch_to_tsquery('english', 'alpha'),
	dense_k => 0,
	bm25_k => 3
)
LIMIT 3;

CREATE TEMP TABLE tqh_ir_rrf AS
SELECT id
FROM tqh_ir_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[1,0,0]'::vector,
	text_query => websearch_to_tsquery('english', 'alpha'),
	dense_k => 3,
	bm25_k => 3
)
LIMIT 3;

CREATE TEMP TABLE tqh_ir_weighted AS
SELECT id
FROM tqh_ir_docs
ORDER BY embedding <~> hybrid_query(
	vector_query => '[1,0,0]'::vector,
	text_query => websearch_to_tsquery('english', 'alpha'),
	fusion => 'weighted',
	alpha => 0.5,
	dense_k => 3,
	bm25_k => 3
)
LIMIT 3;

SELECT
	(SELECT array_agg(id) FROM tqh_ir_dense) AS dense_order,
	(SELECT array_agg(id) FROM tqh_ir_bm25) AS bm25_order,
	(SELECT array_agg(id) FROM tqh_ir_rrf) AS rrf_order,
	(SELECT array_agg(id) FROM tqh_ir_weighted) AS weighted_order,
	(SELECT id FROM tqh_ir_rrf LIMIT 1) = 1 AS hybrid_exact_semantic_first,
	(SELECT id FROM tqh_ir_weighted LIMIT 1) = 1 AS weighted_exact_semantic_first;

DROP TABLE tqh_ir_dense;
DROP TABLE tqh_ir_bm25;
DROP TABLE tqh_ir_rrf;
DROP TABLE tqh_ir_weighted;

RESET enable_seqscan;

DROP TABLE tqh_ir_docs;

DO $$
BEGIN
	CREATE INDEX tqh_docs_bad_one_key ON tqh_docs
	USING turbohybrid (embedding vector_cosine_hybrid_ops);
	RAISE EXCEPTION 'expected one-key turbohybrid index to be rejected';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

DO $$
BEGIN
	CREATE INDEX tqh_docs_bad_second_key ON tqh_docs
	USING turbohybrid (
		embedding vector_cosine_hybrid_ops,
		tenant_id bm25_tsvector_ops
	);
	RAISE EXCEPTION 'expected non-tsvector second key to be rejected';
EXCEPTION WHEN datatype_mismatch THEN
END
$$;

DROP TABLE tqh_docs;
