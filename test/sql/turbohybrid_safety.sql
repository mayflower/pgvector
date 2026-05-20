SET enable_seqscan = off;

CREATE TABLE th_safety_hnsw (id int, val vector(3));
INSERT INTO th_safety_hnsw VALUES
	(1, '[0,0,0]'),
	(2, '[1,2,3]'),
	(3, '[1,1,1]');
CREATE INDEX th_safety_hnsw_idx ON th_safety_hnsw USING hnsw (val vector_l2_ops);

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM th_safety_hnsw
	ORDER BY val <-> '[1,2,3]'
	LIMIT 1;

	IF top_id <> 2 THEN
		RAISE EXCEPTION 'expected hnsw top-1 id 2, got %', top_id;
	END IF;
END
$$;

CREATE TABLE th_safety_tq (id int, val vector(3));
INSERT INTO th_safety_tq VALUES
	(1, '[0,0,0]'),
	(2, '[1,2,3]'),
	(3, '[1,1,1]');
CREATE INDEX th_safety_tq_idx ON th_safety_tq
	USING turboquant (val vector_l2_ops)
	WITH (routing = graph);

DO $$
DECLARE
	top_id int;
	stats jsonb;
BEGIN
	SELECT id INTO top_id
	FROM th_safety_tq
	ORDER BY val <-> '[1,2,3]'
	LIMIT 1;

	IF top_id <> 2 THEN
		RAISE EXCEPTION 'expected turboquant top-1 id 2, got %', top_id;
	END IF;

	stats := tq_last_scan_stats();
	IF stats->>'scan_orchestration' <> 'graph_native' THEN
		RAISE EXCEPTION 'expected native graph scan, got %', stats;
	END IF;

	stats := tq_index_stats('th_safety_tq_idx'::regclass);
	IF stats->>'storage_kind' <> 'turboquant_graph_native' THEN
		RAISE EXCEPTION 'expected native graph storage, got %', stats;
	END IF;
	IF (stats->>'tq_live_node_count')::int <> 3 THEN
		RAISE EXCEPTION 'expected 3 live turboquant nodes, got %', stats;
	END IF;
END
$$;

DROP TABLE th_safety_hnsw;
DROP TABLE th_safety_tq;

CREATE TABLE th_safety_hybrid (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO th_safety_hybrid VALUES
	(1, '[1,0,0]', to_tsvector('english', repeat('large toasted vector lexeme ', 2000))),
	(2, '[0,1,0]', ''::tsvector),
	(3, '[0,0,1]', to_tsvector('english', 'repeat repeat repeat positions'));

CREATE INDEX th_safety_hybrid_idx ON th_safety_hybrid
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
);

SELECT
	tq_debug_bm25_topk('th_safety_hybrid_idx'::regclass, to_tsquery('english', 'large'), 5, false)->>'result_count' AS large_count,
	tq_debug_bm25_topk('th_safety_hybrid_idx'::regclass, to_tsquery('english', 'repeat'), 5, false)->>'result_count' AS repeat_count,
	tq_debug_bm25_topk('th_safety_hybrid_idx'::regclass, to_tsquery('english', 'missing'), 5, false)->>'result_count' AS missing_count;

INSERT INTO th_safety_hybrid VALUES
	(4, '[1,1,1]', to_tsvector('english', repeat('delta toasted lexeme ', 2000)));

SELECT id
FROM th_safety_hybrid
ORDER BY embedding <~> hybrid_query(
	text_query => to_tsquery('english', 'delta'),
	dense_k => 0,
	bm25_k => 5
)
LIMIT 1;

DO $$
BEGIN
	INSERT INTO th_safety_hybrid
	SELECT 5, '[1,0,1]', array_to_tsvector(ARRAY(
		SELECT repeat('oversizedbytes', 8) || i
		FROM generate_series(1, 256) i
	));

	RAISE EXCEPTION 'expected oversized BM25 delta byte-size rejection';
EXCEPTION
	WHEN program_limit_exceeded THEN
		NULL;
END
$$;

DO $$
BEGIN
	INSERT INTO th_safety_hybrid
	SELECT 6, '[1,0,1]', array_to_tsvector(ARRAY(
		SELECT 'oversized' || i
		FROM generate_series(1, 65536) i
	));

	RAISE EXCEPTION 'expected oversized BM25 delta rejection';
EXCEPTION
	WHEN program_limit_exceeded THEN
		NULL;
END
$$;

DROP TABLE th_safety_hybrid;
