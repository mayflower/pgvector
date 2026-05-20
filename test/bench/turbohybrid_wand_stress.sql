CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS tqh_wand_stress;
CREATE TABLE tqh_wand_stress (
	id int PRIMARY KEY,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_wand_stress
SELECT
	i,
	format('[%s,%s,%s]',
		((i * 17) % 100)::float8 / 100.0,
		((i * 31) % 100)::float8 / 100.0,
		((i * 47) % 100)::float8 / 100.0)::vector,
	to_tsvector('simple', concat_ws(' ',
		'common',
		'term' || (i % 100),
		'term' || ((i * 7) % 100),
		'term' || ((i * 13) % 100),
		'term' || ((i * 29) % 100)))
FROM generate_series(1, 1000) i;

CREATE INDEX tqh_wand_stress_idx ON tqh_wand_stress
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (bm25_block_max = on);

CREATE TEMP TABLE tqh_wand_stress_queries AS
WITH query_specs AS (
	SELECT
		q,
		CASE WHEN q % 2 = 0 THEN ' | ' ELSE ' & ' END AS op,
		((q - 1) % 5) + 1 AS term_count
	FROM generate_series(1, 100) q
), query_text AS (
	SELECT
		q,
		op,
		(
			SELECT string_agg('term' || (((q * 11) + (n * 17)) % 100), op)
			FROM generate_series(1, term_count) n
		) AS query
	FROM query_specs
)
SELECT q, op, k, to_tsquery('simple', query) AS tsq
FROM query_text
CROSS JOIN (VALUES (1), (5), (10), (100)) AS limits(k);

SELECT count(*) = 400 AS generated_query_count
FROM tqh_wand_stress_queries;

SELECT bool_and(
	(tq_debug_bm25_topk('tqh_wand_stress_idx'::regclass, tsq, k, true)->'results') =
	(tq_debug_bm25_topk('tqh_wand_stress_idx'::regclass, tsq, k, false)->'results')
) AS wand_matches_daat
FROM tqh_wand_stress_queries;

SELECT
	(tq_debug_bm25_topk('tqh_wand_stress_idx'::regclass, to_tsquery('simple', 'common | term1 | term2 | term3'), 10, true)->>'blocks_skipped')::int > 0 AS wand_skips_blocks,
	(tq_debug_bm25_topk('tqh_wand_stress_idx'::regclass, to_tsquery('simple', 'common | term1 | term2 | term3'), 10, true)->>'wand_iterations')::int > 0 AS wand_iterations;

DROP TABLE tqh_wand_stress;
