SET enable_seqscan = off;

CREATE TABLE tq (val vector(3));
INSERT INTO tq (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX tq_val_idx ON tq USING turboquant (val vector_l2_ops);

INSERT INTO tq (val) VALUES ('[1,2,4]');

SELECT * FROM tq ORDER BY val <-> '[3,3,3]';
SELECT tq_last_scan_stats()->>'scan_orchestration';
SELECT tq_last_scan_stats() ? 'graph_visited_nodes';
SELECT tq_last_scan_stats() ? 'graph_scored_codes';
SELECT tq_last_scan_stats() ? 'graph_candidate_count';
SELECT tq_last_scan_stats() ? 'graph_rescore_count';
SELECT tq_last_scan_stats() ? 'graph_rescore_pages';
SELECT tq_last_scan_stats() ? 'graph_entry_point_count';
SELECT tq_last_scan_stats() ? 'graph_scoring_kernel';
SELECT tq_last_scan_stats() ? 'graph_tq_bits';
SELECT tq_last_scan_stats() ? 'graph_query_split_active';
SELECT tq_last_scan_stats()->>'graph_storage_kind';
SELECT (tq_last_scan_stats()->>'graph_scored_codes')::int > 0;
SELECT (tq_last_scan_stats()->>'graph_rescore_count')::int > 0;
SELECT (tq_last_scan_stats()->>'graph_rescore_pages')::int = 0;
SELECT length(tq_last_scan_stats()->>'graph_scoring_kernel') > 0;
SELECT (tq_last_scan_stats()->>'graph_tq_bits')::int IN (0, 1, 2, 4);
DO $$
DECLARE
	stats jsonb := tq_last_scan_stats();
BEGIN
	IF NOT stats ? 'graph_prepare_us' OR
	   NOT stats ? 'graph_traverse_us' OR
	   NOT stats ? 'graph_fill_us' OR
	   NOT stats ? 'graph_rescore_us' OR
	   NOT stats ? 'graph_sort_us' OR
	   NOT stats ? 'graph_total_us' THEN
		RAISE EXCEPTION 'expected graph phase timing counters';
	END IF;
	IF (stats->>'graph_total_us')::int < 0 THEN
		RAISE EXCEPTION 'expected non-negative graph total time';
	END IF;
END
$$;
SELECT tq_index_stats('tq_val_idx'::regclass)->>'graph_wal_mode';
SELECT tq_index_stats('tq_val_idx'::regclass) ? 'graph_custom_wal_records';
SELECT tq_index_stats('tq_val_idx'::regclass)->>'graph_page_op_tag_mode';
SELECT (tq_index_stats('tq_val_idx'::regclass)->>'meta_last_graph_op')::int > 0;
SELECT (tq_index_stats('tq_val_idx'::regclass)->>'first_graph_last_graph_op')::int > 0;
SELECT (tq_index_stats('tq_val_idx'::regclass)->>'graph_page_count')::int > 0;
SELECT (tq_index_stats('tq_val_idx'::regclass)->>'graph_tagged_page_count')::int > 0;
SELECT tq_index_stats('tq_val_idx'::regclass)->'graph_page_last_op_counts' ? 'page_init';

DROP TABLE tq;

CREATE TABLE tq (id int, val vector(3));
INSERT INTO tq
SELECT i, ARRAY[(i % 17) + 1, (i % 31) + 1, (i % 43) + 1]::vector(3)
FROM generate_series(1, 80) i;
CREATE INDEX tq_val_cos_idx ON tq USING turboquant (val vector_cosine_ops) WITH (routing = graph);
DO $$
BEGIN
	IF tq_index_stats('tq_val_cos_idx'::regclass)->>'tq_plus' <> 'true' THEN
		RAISE EXCEPTION 'expected tq_plus metadata';
	END IF;
	IF (tq_index_stats('tq_val_cos_idx'::regclass)->>'tq_flags')::int <= 0 THEN
		RAISE EXCEPTION 'expected tq_flags metadata';
	END IF;
	IF (tq_index_stats('tq_val_cos_idx'::regclass)->>'tq_correction_start_block')::int <= 0 THEN
		RAISE EXCEPTION 'expected correction start block';
	END IF;
END
$$;
DROP TABLE tq;

CREATE TABLE tq (id int, val vector(1024));
INSERT INTO tq
SELECT i, ARRAY(
	SELECT ((((i::bigint * 130363 + g::bigint * 17011 + ((i::bigint * g::bigint) % 8191) * 19) % 1000003)::float8 / 1000003.0) - 0.5)
	FROM generate_series(1, 1024) g
)::vector(1024)
FROM generate_series(1, 180) i;
CREATE INDEX tq_val_cos_hd_idx ON tq USING turboquant (val vector_cosine_ops) WITH (routing = graph, graph_ef_search = 128, graph_oversampling = 8);
DO $$
DECLARE
	query vector(1024);
	index_ids int[];
	exact_ids int[];
	stats jsonb;
BEGIN
	SELECT ARRAY(
		SELECT ((((424242::bigint + g::bigint * 49979687 + ((17::bigint * g::bigint) % 8191) * 17) % 1000003)::float8 / 1000003.0) - 0.5)
		FROM generate_series(1, 1024) g
	)::vector(1024) INTO query;

	SELECT array_agg(id ORDER BY ord) INTO index_ids
	FROM (
		SELECT id, row_number() OVER () AS ord
		FROM (
			SELECT id
			FROM tq
			ORDER BY val <=> query
			LIMIT 5
		) ordered
	) s;

	SELECT array_agg(id ORDER BY ord) INTO exact_ids
	FROM (
		SELECT id, row_number() OVER () AS ord
		FROM (
			SELECT id
			FROM tq
			ORDER BY (val <=> query) + 0, id
			LIMIT 5
		) ordered
	) s;

	IF index_ids <> exact_ids THEN
		RAISE EXCEPTION 'expected high-dimensional cosine graph order %, got %', exact_ids, index_ids;
	END IF;

	stats := tq_last_scan_stats();
	IF stats->>'scan_orchestration' <> 'graph_native' THEN
		RAISE EXCEPTION 'expected native graph scan';
	END IF;
	IF (stats->>'graph_scored_codes')::int <= 0 THEN
		RAISE EXCEPTION 'expected high-dimensional cosine code scoring';
	END IF;
END
$$;
DROP TABLE tq;

CREATE TABLE tq (id int, val vector(3));
INSERT INTO tq
SELECT i, ARRAY[(i % 17) / 17.0, (i % 31) / 31.0, (i % 43) / 43.0]::vector(3)
FROM generate_series(1, 120) i;
CREATE INDEX tq_val_bits2_idx ON tq USING turboquant (val vector_l2_ops) WITH (routing = graph, tq_bits = 2, graph_ef_search = 64);
DO $$
DECLARE
	row_count int;
BEGIN
	IF (tq_index_stats('tq_val_bits2_idx'::regclass)->>'tq_bits')::int <> 2 THEN
		RAISE EXCEPTION 'expected 2-bit metadata';
	END IF;
	SELECT count(*) INTO row_count
	FROM (SELECT id FROM tq ORDER BY val <-> '[0.2,0.3,0.4]' LIMIT 5) s;
	IF row_count <> 5 THEN
		RAISE EXCEPTION 'expected 2-bit graph scan rows';
	END IF;
	IF tq_last_scan_stats()->>'scan_orchestration' <> 'graph_native' THEN
		RAISE EXCEPTION 'expected native graph scan';
	END IF;
	IF (tq_last_scan_stats()->>'graph_rescore_count')::int <>
		(tq_last_scan_stats()->>'graph_candidate_count')::int THEN
		RAISE EXCEPTION 'expected low-bit graph scan to rescore candidate band';
	END IF;
END
$$;
INSERT INTO tq VALUES (121, '[0.2,0.3,0.4]');
DO $$
DECLARE
	row_count int;
BEGIN
	SELECT count(*) INTO row_count
	FROM (SELECT id FROM tq ORDER BY val <-> '[0.2,0.3,0.4]' LIMIT 5) s;
	IF row_count <> 5 THEN
		RAISE EXCEPTION 'expected 2-bit graph rows after insert';
	END IF;
END
$$;
DROP TABLE tq;

CREATE TABLE tq (id int, val vector(3));
INSERT INTO tq
SELECT i, ARRAY[(i % 17) / 17.0, (i % 31) / 31.0, (i % 43) / 43.0]::vector(3)
FROM generate_series(1, 200) i;
CREATE INDEX tq_val_idx ON tq USING turboquant (val vector_l2_ops) WITH (graph_ef_search = 64, graph_oversampling = 4);
SELECT count(*) FROM (SELECT id FROM tq ORDER BY val <-> '[0.2,0.3,0.4]' LIMIT 5) s;
SELECT (tq_last_scan_stats()->>'graph_rescore_count')::int = (tq_last_scan_stats()->>'graph_candidate_count')::int;
SELECT (tq_last_scan_stats()->>'graph_rescore_pages')::int <= (tq_last_scan_stats()->>'graph_rescore_count')::int;
SELECT (tq_last_scan_stats()->>'graph_candidate_count')::int > 0;
DROP TABLE tq;

CREATE TABLE tq (id int, val vector(3));
INSERT INTO tq (id, val) VALUES
	(1, '[1000,0,0]'),
	(2, '[0,1000,0]'),
	(3, '[0,0,1000]'),
	(4, '[900,30,10]'),
	(5, '[-1000,0,0]'),
	(6, '[800,100,0]');
CREATE INDEX tq_val_idx ON tq USING turboquant (val vector_l2_ops) WITH (graph_ef_search = 64, graph_rescore_band = exact);
SELECT array_agg(id) FROM (SELECT id FROM tq ORDER BY val <-> '[900,20,5]' LIMIT 4) s;
SELECT (tq_last_scan_stats()->>'graph_scored_codes')::int > 0;
DROP TABLE tq;

CREATE TABLE tq (id int, bucket int, val vector(3));
INSERT INTO tq
SELECT i, i % 10, ARRAY[(i % 17) / 17.0, (i % 31) / 31.0, (i % 43) / 43.0]::vector(3)
FROM generate_series(1, 100) i;
CREATE INDEX tq_val_payload_idx ON tq USING turboquant (val vector_l2_ops) INCLUDE (bucket) WITH (routing = graph);
DO $$
DECLARE
	row_count int;
BEGIN
	SELECT count(*) INTO row_count
	FROM (SELECT id FROM tq WHERE bucket = 3 ORDER BY val <-> '[0.2,0.3,0.4]' LIMIT 3) s;
	IF row_count <> 3 THEN
		RAISE EXCEPTION 'expected payload graph rows';
	END IF;
	IF (tq_last_scan_stats()->>'graph_entry_point_count')::int <> 0 THEN
		RAISE EXCEPTION 'expected payload exact scan to bypass graph entry points';
	END IF;
	IF (tq_last_scan_stats()->>'graph_scored_codes')::int <> 0 THEN
		RAISE EXCEPTION 'expected payload exact scan to avoid packed graph scoring';
	END IF;
	IF (tq_last_scan_stats()->>'graph_candidate_count')::int <= 0 THEN
		RAISE EXCEPTION 'expected payload exact candidates';
	END IF;
	IF (tq_last_scan_stats()->>'graph_rescore_count')::int <= 0 THEN
		RAISE EXCEPTION 'expected payload exact rescore count';
	END IF;
END
$$;
DROP TABLE tq;

CREATE TABLE tq (val vector(3));
INSERT INTO tq (val) VALUES ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), (NULL);
CREATE INDEX tq_val_flat_idx ON tq USING turboquant (val vector_l2_ops) WITH (routing = flat);
SELECT * FROM tq ORDER BY val <-> '[3,3,3]';
SELECT tq_last_scan_stats()->>'scan_orchestration';
SELECT (tq_last_scan_stats()->>'graph_scored_codes')::int;
SELECT tq_last_scan_stats()->>'graph_storage_kind';
SELECT tq_index_stats('tq_val_flat_idx'::regclass)->>'storage_kind';
DROP TABLE tq;

CREATE TABLE tq (id int, val vector(3));
INSERT INTO tq
SELECT i, ARRAY[(i % 19) / 19.0, (i % 29) / 29.0, (i % 37) / 37.0]::vector(3)
FROM generate_series(1, 150) i;
CREATE INDEX tq_val_flat_idx ON tq USING turboquant (val vector_l2_ops) WITH (routing = flat);
CREATE TEMP TABLE tq_flat_result AS
SELECT array_agg(id) AS ids
FROM (SELECT id FROM tq ORDER BY val <-> '[0.21,0.29,0.41]' LIMIT 10) s;
SELECT tq_last_scan_stats()->>'scan_orchestration';
SET enable_indexscan = off;
SET enable_seqscan = on;
SELECT ids = (SELECT array_agg(id) FROM (SELECT id FROM tq ORDER BY val <-> '[0.21,0.29,0.41]' LIMIT 10) s)
FROM tq_flat_result;
SET enable_indexscan = on;
SET enable_seqscan = off;
DROP TABLE tq_flat_result;
DROP TABLE tq;

CREATE TABLE tq_vacuum_chain (id int, val vector(1536));
INSERT INTO tq_vacuum_chain
SELECT 1, ARRAY(
    SELECT sin(0.01 * k)::real FROM generate_series(1, 1536) k
)::vector(1536);
CREATE INDEX tq_vacuum_chain_idx ON tq_vacuum_chain
  USING turboquant (val vector_l2_ops)
  WITH (routing = graph, graph_m = 4, graph_ef_construction = 16);
INSERT INTO tq_vacuum_chain
SELECT id, ARRAY(
    SELECT sin(0.01 * (k + id))::real FROM generate_series(1, 1536) k
)::vector(1536)
FROM generate_series(2, 40) id;
DELETE FROM tq_vacuum_chain WHERE id >= 11;
VACUUM tq_vacuum_chain;
DO $$
DECLARE
	vacuum_pages int;
BEGIN
	SELECT ((tq_index_stats('tq_vacuum_chain_idx'::regclass)
		-> 'graph_page_last_op_counts' ->> 'vacuum_delete')::int)
	INTO vacuum_pages;

	IF vacuum_pages < 2 THEN
		RAISE EXCEPTION 'expected vacuum to follow linked TurboQuant code pages, got % pages',
			vacuum_pages;
	END IF;
END
$$;
DROP TABLE tq_vacuum_chain;

CREATE TABLE tq (val vector(3));
CREATE INDEX tq_auto_idx ON tq USING turboquant (val vector_l2_ops) WITH (routing = auto);
CREATE INDEX tq_graph_idx ON tq USING turboquant (val vector_l2_ops) WITH (routing = graph, graph_m = 16, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, graph_rescore_band = auto);
CREATE INDEX tq_flat_idx ON tq USING turboquant (val vector_l2_ops) WITH (routing = flat);
SELECT tq_index_stats('tq_auto_idx'::regclass)->>'storage_kind';
SELECT (tq_index_stats('tq_auto_idx'::regclass)->>'graph_ef_search')::int;
SELECT (tq_index_stats('tq_auto_idx'::regclass)->>'graph_oversampling')::int;
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (routing = bad);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (routing = ivf);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (routing = legacy_hnsw);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (lists = 1);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_m = 1);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_ef_construction = 3);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_ef_search = 0);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_oversampling = 0);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_rescore_band = bad);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_exact_cache = bad);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (graph_reorder = bad);
CREATE INDEX ON tq USING turboquant (val vector_l2_ops) WITH (tq_bits = 3);

DROP TABLE tq;
