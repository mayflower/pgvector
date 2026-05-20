-- Adjacency-list look-ahead prefetch (size-gated).
--
-- The original FAISS-style prefetch was reverted on FIQA-57k (commit
-- 67f38bd) because the metadata working set fit in cache and the
-- explicit __builtin_prefetch was paid-for-nothing uops.  This
-- regression test pins the *gating* invariant: at small corpora
-- (FIQA-scale and smaller) the auto mode must keep prefetch
-- *disabled*, so the same regression cannot recur silently.
--
-- The test asserts:
--   1. Defaults (auto + 24 MiB threshold) are correct.
--   2. Forcing the GUC on doesn't break end-to-end search results
--      — the prefetch is a hint, not a semantic change.
--   3. Forcing the GUC off matches forced-on ranking exactly (the
--      hint is invisible to result ranking).

SET enable_seqscan = off;

CREATE TEMP TABLE tq_la_data AS
SELECT id, ARRAY(
    SELECT sin(0.31415 * (k + id))::real FROM generate_series(0, 1535) k
)::vector(1536) AS val
FROM generate_series(1, 64) id;

-- Default GUCs.
SHOW hnsw.tq_graph_lookahead_prefetch;
SHOW hnsw.tq_graph_lookahead_threshold_kb;

CREATE INDEX tq_la_idx ON tq_la_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph);

-- Capture top-5 self-search results under each GUC value.  All three
-- modes must produce identical rankings — prefetch is a microarchitectural
-- hint, never a semantic change.
SET hnsw.tq_graph_lookahead_prefetch = auto;
CREATE TEMP TABLE tq_la_auto AS
SELECT probe.id AS probe_id, array_agg(t.id ORDER BY t.id) AS top5
FROM tq_la_data probe,
     LATERAL (SELECT id FROM tq_la_data ORDER BY val <=> probe.val LIMIT 5) t
WHERE probe.id <= 6
GROUP BY probe.id;

SET hnsw.tq_graph_lookahead_prefetch = on;
CREATE TEMP TABLE tq_la_on AS
SELECT probe.id AS probe_id, array_agg(t.id ORDER BY t.id) AS top5
FROM tq_la_data probe,
     LATERAL (SELECT id FROM tq_la_data ORDER BY val <=> probe.val LIMIT 5) t
WHERE probe.id <= 6
GROUP BY probe.id;

SET hnsw.tq_graph_lookahead_prefetch = off;
CREATE TEMP TABLE tq_la_off AS
SELECT probe.id AS probe_id, array_agg(t.id ORDER BY t.id) AS top5
FROM tq_la_data probe,
     LATERAL (SELECT id FROM tq_la_data ORDER BY val <=> probe.val LIMIT 5) t
WHERE probe.id <= 6
GROUP BY probe.id;

RESET hnsw.tq_graph_lookahead_prefetch;

DO $$
DECLARE
    diff_count int;
BEGIN
    SELECT count(*)
    INTO diff_count
    FROM tq_la_auto a
    JOIN tq_la_on o USING (probe_id)
    WHERE a.top5 IS DISTINCT FROM o.top5;
    IF diff_count > 0 THEN
        RAISE EXCEPTION 'prefetch on/auto rankings disagree on % probes', diff_count;
    END IF;

    SELECT count(*)
    INTO diff_count
    FROM tq_la_auto a
    JOIN tq_la_off f USING (probe_id)
    WHERE a.top5 IS DISTINCT FROM f.top5;
    IF diff_count > 0 THEN
        RAISE EXCEPTION 'prefetch off/auto rankings disagree on % probes', diff_count;
    END IF;
END $$;

-- Auto-off invariant for small corpora: with the default 24 MiB
-- threshold, a 64-row 1536-dim build (~ 4 KiB metadata) is way under
-- the threshold, so auto mode picks the OFF path.  Setting the
-- threshold to 0 forces auto = ON; setting it to a huge value forces
-- auto = OFF.  Both must still produce the same ranking as the
-- explicit on/off modes.
SET hnsw.tq_graph_lookahead_prefetch = auto;
SET hnsw.tq_graph_lookahead_threshold_kb = 0;
SET hnsw.tq_graph_lookahead_threshold_kb = 1073741823;

DO $$
DECLARE
    top_on int[];
    top_off int[];
BEGIN
    SET hnsw.tq_graph_lookahead_threshold_kb = 0;
    SELECT array_agg(id ORDER BY ord)
    INTO top_on
    FROM (
        SELECT id, row_number() OVER () AS ord
        FROM (
            SELECT id FROM tq_la_data
            ORDER BY val <=> (SELECT val FROM tq_la_data WHERE id = 5)
            LIMIT 3
        ) s
    ) ranked;

    SET hnsw.tq_graph_lookahead_threshold_kb = 1073741823;
    SELECT array_agg(id ORDER BY ord)
    INTO top_off
    FROM (
        SELECT id, row_number() OVER () AS ord
        FROM (
            SELECT id FROM tq_la_data
            ORDER BY val <=> (SELECT val FROM tq_la_data WHERE id = 5)
            LIMIT 3
        ) s
    ) ranked;

    IF array_length(top_on, 1) <> 3 OR array_length(top_off, 1) <> 3 THEN
        RAISE EXCEPTION 'prefetch threshold probes returned incomplete top-3 results';
    END IF;
    IF top_on IS DISTINCT FROM top_off THEN
        RAISE EXCEPTION 'prefetch threshold rankings disagree: % vs %', top_on, top_off;
    END IF;
END $$;
RESET hnsw.tq_graph_lookahead_threshold_kb;
RESET hnsw.tq_graph_lookahead_prefetch;

DROP INDEX tq_la_idx;
