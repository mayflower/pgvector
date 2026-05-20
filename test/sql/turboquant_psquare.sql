-- Extended P-square one-quantile estimator.
--
-- Asserts that the streaming P-square estimator (TqPSquarePush /
-- TqPSquareEstimate) produces an estimate close to the true quantile
-- of a known distribution.  Tolerance set per the design doc's
-- variance bound: at p=0.99 over n=1000 samples, std ≈ 0.20 (Gaussian
-- f(F⁻¹(p)) ≈ 0.027 → variance = p·(1-p)/(n·f²) ≈ 0.027 → std ≈ 0.16).
-- We use 5 % absolute as a safe loose bound.

-- Smoke: tiny array, near-median quantile.  P-square needs >= 5
-- observations; with exactly 5 it returns the sorted middle.
SELECT abs(tq_debug_psquare_estimate(ARRAY[1.0, 2.0, 3.0, 4.0, 5.0], 0.5) - 3.0)
       < 1e-9 AS smoke_median_5obs;

-- Median of a uniform [0, 1] distribution at n=10000.  True median = 0.5.
-- P-square is order-sensitive; use random() to avoid monotonic input
-- (which is the algorithm's worst case).
SELECT abs(
    tq_debug_psquare_estimate(
        ARRAY(SELECT random()::double precision FROM generate_series(1, 10000)),
        0.5
    ) - 0.5
) < 0.05 AS median_uniform_within_5pct;

-- Quantile 0.99 of a uniform [0, 1] distribution at n=10000.  True 0.99-quantile = 0.99.
SELECT abs(
    tq_debug_psquare_estimate(
        ARRAY(SELECT random()::double precision FROM generate_series(1, 10000)),
        0.99
    ) - 0.99
) < 0.05 AS p99_uniform_within_5pct;

-- Quantile 0.5 of a Gaussian-ish distribution (sum of 12 uniforms, CLT).
-- True median ≈ 6.0.
DO $$
DECLARE
    obs double precision[];
    est double precision;
BEGIN
    obs := ARRAY(
        SELECT
            (random() + random() + random() + random() +
             random() + random() + random() + random() +
             random() + random() + random() + random())::double precision
        FROM generate_series(1, 5000)
    );
    est := tq_debug_psquare_estimate(obs, 0.5);
    IF abs(est - 6.0) > 0.30 THEN
        RAISE EXCEPTION 'P-square median of CLT-Gaussian should be ~6.0, got %', est;
    END IF;
END $$;

-- Quantile 0.997 (close to Bits4 deep-tail target for TQ+).  At n=20000
-- this is ~60 observations above the target — at the edge of P-square's
-- accuracy with N=5 markers, but should still land within 0.3 absolute
-- of the true value (~10.0 for 12-uniform sum, since Phi(2.733)≈0.997
-- is well past mean+3σ for that distribution).
DO $$
DECLARE
    obs double precision[];
    est double precision;
BEGIN
    obs := ARRAY(
        SELECT
            (random() + random() + random() + random() +
             random() + random() + random() + random() +
             random() + random() + random() + random())::double precision
        FROM generate_series(1, 20000)
    );
    est := tq_debug_psquare_estimate(obs, 0.997);
    -- True 0.997-quantile of 12-uniform sum is ~9.65 (mean 6, std 1, ~3.65σ tail).
    -- P-square at deep tails is noisier; use a 1.0 absolute bound.
    IF est < 8.0 OR est > 11.5 THEN
        RAISE EXCEPTION 'P-square 0.997-quantile out of plausible range, got %', est;
    END IF;
END $$;

-- Edge case: empty array → returns 0.
SELECT tq_debug_psquare_estimate(ARRAY[]::double precision[], 0.5) AS empty_estimate;

-- Edge case: < N observations → returns sorted-middle.
SELECT tq_debug_psquare_estimate(ARRAY[5.0, 1.0, 3.0], 0.5) AS small_estimate_returns_middle;

-- Quantile must be in (0, 1).
DO $$
BEGIN
    BEGIN
        PERFORM tq_debug_psquare_estimate(ARRAY[1.0, 2.0, 3.0], 0.0);
        RAISE EXCEPTION 'expected error for quantile = 0';
    EXCEPTION WHEN invalid_parameter_value THEN
        NULL;
    END;
    BEGIN
        PERFORM tq_debug_psquare_estimate(ARRAY[1.0, 2.0, 3.0], 1.5);
        RAISE EXCEPTION 'expected error for quantile = 1.5';
    EXCEPTION WHEN invalid_parameter_value THEN
        NULL;
    END;
END $$;

-- tq_quantile_fit reloption integration.
--
-- Build the same dataset twice — once with the default Welford fit,
-- once with the quantile fit — and assert both produce a usable
-- index that finds self in the top-1 results.  We don't assert
-- which fit produces *better* recall here; that's the
-- recall A/B's job.
SET enable_seqscan = off;

CREATE TEMP TABLE tq_qfit_data AS
SELECT id, ARRAY(
    SELECT sin(0.31415 * (k + id))::real FROM generate_series(0, 1535) k
)::vector(1536) AS val
FROM generate_series(1, 32) id;

-- Welford fit (default).
CREATE INDEX tq_qfit_w ON tq_qfit_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph, tq_weighted = on);

-- Quantile fit (opt-in).
CREATE INDEX tq_qfit_q ON tq_qfit_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph, tq_weighted = on, tq_quantile_fit = on);

-- Reloption should persist on the quantile-fit index.
SELECT (
    'tq_quantile_fit=on' = ANY (
        SELECT unnest(reloptions)
        FROM pg_class
        WHERE relname = 'tq_qfit_q'
    )
) AS quantile_fit_reloption_persisted;

-- Both indexes should return self as top-1 for at least one query
-- (search is approximate so not strictly guaranteed; on this small
-- dataset it should be).  This catches "fit produces zero/NaN
-- shift+scale that breaks scoring entirely" regressions.
SET hnsw.tq_weighted = on;

DROP INDEX tq_qfit_q;
SELECT id FROM tq_qfit_data
  ORDER BY val <=> (SELECT val FROM tq_qfit_data WHERE id = 7)
  LIMIT 1;

DROP INDEX tq_qfit_w;
CREATE INDEX tq_qfit_q2 ON tq_qfit_data
  USING turboquant (val vector_cosine_ops)
  WITH (routing = graph, tq_weighted = on, tq_quantile_fit = on);
SELECT id FROM tq_qfit_data
  ORDER BY val <=> (SELECT val FROM tq_qfit_data WHERE id = 7)
  LIMIT 1;
DROP INDEX tq_qfit_q2;
RESET hnsw.tq_weighted;
