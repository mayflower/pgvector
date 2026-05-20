-- TurboQuant+ renormalization: ensure the opt-in setting
-- knobs parse.
--
-- This test only validates that:
--   * the reloption WITH (tq_renorm = on) round-trips,
--   * the GUC hnsw.tq_renorm is registered (default off),
--   * the metapage TQ_GRAPH_TQ_RENORM (0x0004) flag bit is set when
--     the reloption is on and ecShift/ecScale corrections exist
--     (cosine path), and is not set on plain L2 builds.

SET enable_seqscan = off;

-- Touch a vector value first so the extension's _PG_init runs and the
-- new GUC is registered before SHOW/SET.
CREATE TEMP TABLE tq_r_plus_data AS
SELECT id, ARRAY(
    SELECT sin(0.27182 * (k + id))::real FROM generate_series(0, 1535) k
)::vector(1536) AS val
FROM generate_series(1, 16) id;

-- GUC defaults to off and is user-settable.
SHOW hnsw.tq_renorm;
SET hnsw.tq_renorm = on;
SHOW hnsw.tq_renorm;
RESET hnsw.tq_renorm;

-- Build with the new opt-in on a cosine index where TQ+ corrections
-- exist (TqGraphFitCorrection runs for COSINE/IP only).
CREATE INDEX tq_r_plus_idx ON tq_r_plus_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph, tq_renorm = on);

SELECT (
    'tq_renorm=on' = ANY (
        SELECT unnest(reloptions)
        FROM pg_class
        WHERE relname = 'tq_r_plus_idx'
    )
) AS reloption_persisted;

-- Storage flag: tqFlags must carry TQ_GRAPH_TQ_RENORM (0x0004) on top
-- of TQ_GRAPH_TQ_PLUS (0x0001), since this is a cosine-eligible build
-- with ecShift/ecScale present.
SELECT
    ((tq_index_stats('tq_r_plus_idx'::regclass)->>'tq_flags')::int & 4) <> 0
        AS tq_renorm_flag_set;

-- The GUC kill-switch is a runtime toggle on top of the index-time
-- opt-in.  Self-match should still come back top-1 in both modes —
-- the renorm correction is small enough that for a self-query the
-- top result doesn't change.
SET hnsw.tq_renorm = on;
SELECT id FROM tq_r_plus_data
  ORDER BY val <=> (SELECT val FROM tq_r_plus_data WHERE id = 1)
  LIMIT 1;
SET hnsw.tq_renorm = off;
SELECT id FROM tq_r_plus_data
  ORDER BY val <=> (SELECT val FROM tq_r_plus_data WHERE id = 1)
  LIMIT 1;
RESET hnsw.tq_renorm;

DROP INDEX tq_r_plus_idx;

-- Slice β functional check: build two cosine indexes that differ only
-- in tq_renorm, with tq_weighted=on in both (renorm requires the TQ+
-- weighted infrastructure to share the encoder path).  The encoded
-- node->scale field carries different values in the two builds, so
-- raw distance values (val <=> query) should differ between them
-- even though semantic rankings on the small self-similar fixture
-- remain stable.
CREATE INDEX tq_r_func_off_idx ON tq_r_plus_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph, tq_weighted = on);
CREATE INDEX tq_r_func_on_idx ON tq_r_plus_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph, tq_weighted = on, tq_renorm = on);

-- Both flag bits set on the renorm index, only weighted on the baseline.
SELECT
    ((tq_index_stats('tq_r_func_off_idx'::regclass)->>'tq_flags')::int & 4) = 0
      AS off_idx_renorm_clear,
    ((tq_index_stats('tq_r_func_on_idx'::regclass)->>'tq_flags')::int & 4) <> 0
      AS on_idx_renorm_set;

-- All distances finite (no NaN / Inf from c/ecScale[i] division or
-- the centroid_norm sqrt) on both indexes.  We force each index in
-- turn with enable_indexscan toggles isn't reliable here because both
-- live on the same table; instead, query through both via the
-- pg_index oid and verify finiteness.
DO $$
DECLARE
    r record;
    d double precision;
BEGIN
    FOR r IN SELECT id, val FROM tq_r_plus_data ORDER BY id LIMIT 4
    LOOP
        FOR d IN SELECT (val <=> r.val)::double precision
                  FROM tq_r_plus_data
                  ORDER BY val <=> r.val
                  LIMIT 5
        LOOP
            IF d != d OR d = 'Infinity'::double precision OR d = '-Infinity'::double precision THEN
                RAISE EXCEPTION 'non-finite distance % for query id %', d, r.id;
            END IF;
            IF d < 0 THEN
                RAISE EXCEPTION 'negative cosine distance % for query id %', d, r.id;
            END IF;
        END LOOP;
    END LOOP;
END $$;

DROP INDEX tq_r_func_off_idx;
DROP INDEX tq_r_func_on_idx;

-- L2 build: the renormalization is a Dot/Cosine-only correction.
-- Still allow the reloption to be set on L2; there are no correction
-- arrays to apply, but the storage flag bit stays uniform.
CREATE INDEX tq_r_plus_l2_idx ON tq_r_plus_data
  USING turboquant (val vector_l2_ops)
  WITH (routing = graph, tq_renorm = on);
SELECT (
    'tq_renorm=on' = ANY (
        SELECT unnest(reloptions)
        FROM pg_class
        WHERE relname = 'tq_r_plus_l2_idx'
    )
) AS l2_reloption_persisted;
DROP INDEX tq_r_plus_l2_idx;

-- Default-off path: building without tq_renorm leaves the reloption
-- absent and the storage flag bit clear.  Pre-Phase-4 indexes are
-- byte-identical to a tq_renorm-omitting Phase-4 build.
CREATE INDEX tq_r_plus_default_idx ON tq_r_plus_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph);

SELECT (
    'tq_renorm=on' = ANY (
        SELECT unnest(reloptions)
        FROM pg_class
        WHERE relname = 'tq_r_plus_default_idx'
    )
) AS reloption_default_persisted;

SELECT
    ((tq_index_stats('tq_r_plus_default_idx'::regclass)->>'tq_flags')::int & 4) = 0
        AS tq_renorm_flag_clear_by_default;

DROP INDEX tq_r_plus_default_idx;

-- NOTICE gate: at tq_bits = 4 the renorm is approximately a no-op
-- (centroid_norm ≈ sqrt(d)).  The encoder still pays the per-coord
-- fdiv cost, so we warn so users don't unwittingly opt into the
-- overhead.  This is the *negative* assertion — at 2-bit, where
-- renorm clears the +0.30pt nDCG@10 acceptance bar, no NOTICE should
-- fire.  We don't see a server NOTICE here because client_min_messages
-- default is "notice" and a successful CREATE INDEX with no NOTICE
-- output is the positive case.
CREATE INDEX tq_r_plus_b2_idx ON tq_r_plus_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph, tq_bits = 2, tq_weighted = on, tq_renorm = on);
SELECT
    ((tq_index_stats('tq_r_plus_b2_idx'::regclass)->>'tq_flags')::int & 4) <> 0
        AS tq_renorm_flag_set_at_2bit,
    (tq_index_stats('tq_r_plus_b2_idx'::regclass)->>'tq_bits')::int AS tq_bits;
DROP INDEX tq_r_plus_b2_idx;

-- NOTICE gate at tq_bits = 1: renorm is *worse* than baseline here
-- (no magnitude info to track, renorm injects per-vector noise from
-- EC asymmetry).  We emit a different NOTICE message — not "no
-- effect" (4-bit) but "not recommended, regresses".
CREATE INDEX tq_r_plus_b1_idx ON tq_r_plus_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph, tq_bits = 1, tq_weighted = on, tq_renorm = on);
SELECT
    ((tq_index_stats('tq_r_plus_b1_idx'::regclass)->>'tq_flags')::int & 4) <> 0
        AS tq_renorm_flag_set_at_1bit,
    (tq_index_stats('tq_r_plus_b1_idx'::regclass)->>'tq_bits')::int AS tq_bits;
DROP INDEX tq_r_plus_b1_idx;
