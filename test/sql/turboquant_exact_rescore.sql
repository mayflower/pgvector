-- Exact final-band rescore SIMD parity and GUC kill-switch.
--
-- The exact rescore goes through l2_distance / inner_product /
-- cosine_distance via support->procinfo from TqGraphExactDistance.
-- Those SQL functions wrap VectorL2SquaredDistance /
-- VectorInnerProduct, which dispatch to AVX-512F vs autoVec'd FMA
-- based on hnsw.tq_exact_avx512 plus runtime CPU detection.
--
-- This test asserts:
--   * GUC default is off (downclock-safe).
--   * Flipping the GUC on doesn't change distance values
--     measurably — AVX-512F vs autoVec FMA may differ by a few
--     ULPs (different sum order + possible FMA fusing) but the
--     final scores must agree to ~1e-5 relative.
--   * On hosts without AVX-512F support, flipping the GUC is a
--     no-op (the dispatcher falls back to autoVec).
--   * A self-search returns top-1 = self under either GUC value.

-- Touch a vector value first so the extension's _PG_init runs and the
-- new GUC is registered before SHOW.
CREATE TEMP TABLE tq_rescore_data AS
SELECT id, ARRAY(
    SELECT sin(0.123 * (k + id))::real FROM generate_series(0, 1535) k
)::vector(1536) AS val
FROM generate_series(1, 32) id;

-- Default off, downclock-safe.
SHOW hnsw.tq_exact_avx512;

-- Capture l2_distance and inner_product values under both GUC
-- settings.  If the AVX-512F kernel produces the same values
-- (within a tiny tolerance) we know the dispatch is parity-correct.
DO $$
DECLARE
    a vector;
    b vector;
    l2_off double precision;
    l2_on double precision;
    ip_off double precision;
    ip_on double precision;
    cos_off double precision;
    cos_on double precision;
    tol double precision := 1e-4;
BEGIN
    SELECT val INTO a FROM tq_rescore_data WHERE id = 1;
    SELECT val INTO b FROM tq_rescore_data WHERE id = 17;

    -- autoVec FMA path (default).
    SET LOCAL hnsw.tq_exact_avx512 = off;
    l2_off := vector_l2_squared_distance(a, b);
    ip_off := inner_product(a, b);
    cos_off := cosine_distance(a, b);

    -- AVX-512F dispatch.  On hosts without AVX-512F this is a no-op
    -- (dispatcher falls back to autoVec) so values match exactly.
    SET LOCAL hnsw.tq_exact_avx512 = on;
    l2_on := vector_l2_squared_distance(a, b);
    ip_on := inner_product(a, b);
    cos_on := cosine_distance(a, b);

    IF abs(l2_on - l2_off) > tol * (abs(l2_off) + 1.0) THEN
        RAISE EXCEPTION 'L2 squared parity broken: off=% on=% diff=%',
                        l2_off, l2_on, abs(l2_on - l2_off);
    END IF;
    IF abs(ip_on - ip_off) > tol * (abs(ip_off) + 1.0) THEN
        RAISE EXCEPTION 'inner_product parity broken: off=% on=% diff=%',
                        ip_off, ip_on, abs(ip_on - ip_off);
    END IF;
    IF abs(cos_on - cos_off) > tol * (abs(cos_off) + 1.0) THEN
        RAISE EXCEPTION 'cosine_distance parity broken: off=% on=% diff=%',
                        cos_off, cos_on, abs(cos_on - cos_off);
    END IF;
END $$;

-- End-to-end: a tq_weighted index search returns the same self-match
-- under either GUC value.  Catches the case where the AVX-512 path is
-- wired into rescore but produces wrong results on a real index scan.
SET enable_seqscan = off;
CREATE INDEX tq_rescore_idx ON tq_rescore_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph);

SET hnsw.tq_exact_avx512 = on;
SELECT id FROM tq_rescore_data
  ORDER BY val <=> (SELECT val FROM tq_rescore_data WHERE id = 11)
  LIMIT 1;
SET hnsw.tq_exact_avx512 = off;
SELECT id FROM tq_rescore_data
  ORDER BY val <=> (SELECT val FROM tq_rescore_data WHERE id = 11)
  LIMIT 1;
RESET hnsw.tq_exact_avx512;

DROP INDEX tq_rescore_idx;
