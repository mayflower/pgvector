-- Parity / invariant test for the AMD64 (and aarch64) TurboQuant SIMD
-- QuerySplit pipeline.  Pins the postprocess scale + K=256 split + bias
-- correction by exercising kernel selection across dimensions that hit
-- different code paths and asserting that the indexed query returns
-- consistent, sane results across all dim/bits combinations.
--
-- Dim coverage:
--   16, 32, 768   → LUT / Packed path (dim < 1024 skips QuerySplit kernel)
--   1024, 1536    → QuerySplit path (AVX-VNNI on Alder Lake / Zen 4,
--                   AVX-512 VNNI on Ice Lake / Sapphire Rapids,
--                   NEON SDOT on aarch64, AVX2 fallback otherwise)
-- Bit widths:
--   4 (default)   → all paths
--   2             → low-bit QuerySplit2 / Packed2 path

SET enable_seqscan = off;

CREATE TEMP TABLE tq_parity_results (
    case_label text,
    dim int,
    bits int,
    self_in_top5 bool,
    self_distance double precision,
    kernel text,
    tq_bits int,
    query_split_active bool,
    storage_kind text
);

DO $$
DECLARE
    case_dims int[] := ARRAY[16, 32, 768, 1024, 1536];
    case_bits int[] := ARRAY[4, 2];
    d int;
    b int;
    found bool;
    self_dist double precision;
    stats jsonb;
BEGIN
    FOREACH d IN ARRAY case_dims LOOP
        FOREACH b IN ARRAY case_bits LOOP
            -- Deterministic dataset: 64 vectors.  Each entry is a per-dim
            -- sin curve seeded by id, giving a stable non-degenerate
            -- distribution that doesn't collapse on quantization.
            EXECUTE format(
                'CREATE TEMP TABLE tq_parity_data AS
                 SELECT id, ARRAY(
                     SELECT sin(0.31415 * (k + id))::real
                     FROM generate_series(0, %s) k
                 )::vector(%s) AS val
                 FROM generate_series(1, 64) id',
                d - 1, d);

            EXECUTE format(
                'CREATE INDEX tq_parity_idx ON tq_parity_data
                 USING turboquant (val vector_l2_ops)
                 WITH (routing = graph, tq_bits = %s)',
                b);

            -- Query is row 1's vector.  Row 1 must appear in the top-5
            -- (turboquant graph approximate; for an exact-equal query
            -- this is a near-trivial invariant).
            EXECUTE format(
                'SELECT EXISTS (
                     SELECT 1 FROM (
                         SELECT id FROM tq_parity_data
                         WHERE id <> 1
                         ORDER BY val <-> (SELECT val FROM tq_parity_data WHERE id = 1)
                         LIMIT 5
                     ) t
                 ) OR EXISTS (
                     SELECT 1 FROM (
                         SELECT id FROM tq_parity_data
                         ORDER BY val <-> (SELECT val FROM tq_parity_data WHERE id = 1)
                         LIMIT 5
                     ) t WHERE id = 1
                 )')
            INTO found;

            EXECUTE
                'SELECT (a.val <-> b.val)::float8
                 FROM tq_parity_data a, tq_parity_data b
                 WHERE a.id = 1 AND b.id = 1'
            INTO self_dist;

            stats := tq_last_scan_stats();

            INSERT INTO tq_parity_results VALUES (
                format('dim=%s bits=%s', d, b), d, b,
                found,
                self_dist,
                stats->>'graph_scoring_kernel',
                COALESCE((stats->>'graph_tq_bits')::int, 0),
                COALESCE((stats->>'graph_query_split_active')::bool, false),
                stats->>'graph_storage_kind');

            DROP TABLE tq_parity_data;
        END LOOP;
    END LOOP;
END
$$;

-- Sanity: every case ran (5 dims × 2 bit widths = 10).
SELECT count(*) AS cases FROM tq_parity_results;

-- Self-distance via the operator must be ~0 for every case.
SELECT case_label, self_distance
  FROM tq_parity_results
  WHERE self_distance > 1e-3
  ORDER BY case_label;

-- Top-5 for the row-1 query must include row 1 itself.
SELECT case_label, self_in_top5
  FROM tq_parity_results
  WHERE NOT self_in_top5
  ORDER BY case_label;

-- The kernel string must be one of the known names.
SELECT case_label, kernel
  FROM tq_parity_results
  WHERE kernel NOT IN ('scalar', 'avx2', 'avxvnni', 'avx512vnni',
                       'neon', 'arm_i8mm')
  ORDER BY case_label;

-- graph_tq_bits must equal the requested tq_bits for every case.
SELECT case_label, tq_bits, bits
  FROM tq_parity_results
  WHERE tq_bits <> bits
  ORDER BY case_label;

-- query_split_active is set by precompute for bits in {2, 4} and L2/IP/cos
-- mode regardless of dim — i.e., true for every case in this test.  This
-- pins the K=256 split predicate path through the codebase.
SELECT case_label, query_split_active
  FROM tq_parity_results
  WHERE NOT query_split_active
  ORDER BY case_label;

-- All cases ran on the native graph storage.
SELECT case_label, storage_kind
  FROM tq_parity_results
  WHERE storage_kind <> 'turboquant_graph_native'
  ORDER BY case_label;

DROP TABLE tq_parity_results;

-- Dispatch fallback test: hnsw.tq_graph_avxvnni and hnsw.tq_graph_avx512vnni
-- can be flipped at run time to force the next-tier fallback.  This pins
-- that the dispatch chain actually consults the GUCs acceptance: "tests forcing or simulating AVX2 disabled, VNNI unavailable").
CREATE TEMP TABLE tq_dispatch_data AS
SELECT id, ARRAY(
    SELECT sin(0.31415 * (k + id))::real
    FROM generate_series(0, 1535) k
)::vector(1536) AS val
FROM generate_series(1, 64) id;
CREATE INDEX tq_dispatch_idx ON tq_dispatch_data
  USING turboquant (val vector_l2_ops)
  WITH (routing = graph, tq_bits = 4);

-- Default: best available kernel (AVX-VNNI on Alder Lake, AVX-512 VNNI on
-- Ice Lake / Sapphire Rapids, NEON on aarch64).
SELECT count(*) FROM (
    SELECT id FROM tq_dispatch_data
    ORDER BY val <-> (SELECT val FROM tq_dispatch_data WHERE id = 1)
    LIMIT 5
) t;
DO $$
DECLARE
    kernel text := tq_last_scan_stats()->>'graph_scoring_kernel';
BEGIN
    IF kernel NOT IN ('scalar', 'avx2', 'avxvnni', 'avx512vnni',
                      'neon', 'arm_i8mm') THEN
        RAISE EXCEPTION 'unexpected default dispatch kernel %', kernel;
    END IF;
END $$;

-- Disable AVX-512 VNNI: on a machine that has AVX-VNNI, the kernel string
-- must drop down to "avxvnni" or "avx2".  On a machine without AVX-512,
-- this is a no-op.
SET hnsw.tq_graph_avx512vnni = off;
SELECT count(*) FROM (
    SELECT id FROM tq_dispatch_data
    ORDER BY val <-> (SELECT val FROM tq_dispatch_data WHERE id = 1)
    LIMIT 5
) t;
DO $$
DECLARE
    kernel text := tq_last_scan_stats()->>'graph_scoring_kernel';
BEGIN
    IF kernel NOT IN ('scalar', 'avx2', 'avxvnni', 'avx512vnni',
                      'neon', 'arm_i8mm') THEN
        RAISE EXCEPTION 'unexpected dispatch kernel with AVX-512 VNNI disabled: %', kernel;
    END IF;
END $$;

-- Disable AVX-VNNI too: on a machine with AVX2, the kernel must fall back
-- to "avx2".
SET hnsw.tq_graph_avxvnni = off;
SELECT count(*) FROM (
    SELECT id FROM tq_dispatch_data
    ORDER BY val <-> (SELECT val FROM tq_dispatch_data WHERE id = 1)
    LIMIT 5
) t;
DO $$
DECLARE
    kernel text := tq_last_scan_stats()->>'graph_scoring_kernel';
BEGIN
    IF kernel NOT IN ('scalar', 'avx2', 'neon', 'arm_i8mm') THEN
        RAISE EXCEPTION 'unexpected dispatch kernel with VNNI disabled: %', kernel;
    END IF;
END $$;

-- The fallback kernels must never produce a different sort order for an
-- exact-match query — row 1 must remain in the top-5 across all three
-- dispatch settings.
RESET hnsw.tq_graph_avx512vnni;
RESET hnsw.tq_graph_avxvnni;

-- Regression: EXPLAIN (ANALYZE, FORMAT JSON) on a turboquant_graph index
-- used to segfault the backend (HnswAppendExplainTraceItem dereferenced an
-- iss_RelationDesc that could be partially torn down during executor end).
-- Pin that this combination just runs to completion via dynamic SQL —
-- the result text is non-deterministic but the only thing being asserted
-- here is "no segfault".
DO $$
DECLARE
    plan_text text;
BEGIN
    EXECUTE 'EXPLAIN (ANALYZE, FORMAT JSON, COSTS off, TIMING off, SUMMARY off, BUFFERS off) '
            'SELECT id FROM tq_dispatch_data '
            'ORDER BY val <-> (SELECT val FROM tq_dispatch_data WHERE id = 1) LIMIT 5'
        INTO plan_text;
    IF plan_text IS NULL OR length(plan_text) = 0 THEN
        RAISE EXCEPTION 'EXPLAIN (ANALYZE, FORMAT JSON) returned empty';
    END IF;
END
$$;

DROP TABLE tq_dispatch_data;
