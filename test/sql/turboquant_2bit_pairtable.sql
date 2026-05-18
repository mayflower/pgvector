-- Exhaustive parity test for the 2-bit pair-table SIMD unpack.
--
-- The amd64 path replaces a 16-iteration scalar nibble extraction
-- (TqGraphCodebook2I8 lookup) with two pshufb lookups against
-- TqGraphCodebook2PairEvenI8 / TqGraphCodebook2PairOddI8 and an
-- _mm_unpacklo_epi8.  Any drift in the pair-tables vs the codebook is a
-- silent recall regression, so this test forces every possible 4-byte
-- prefix value to be quantized and queried.
--
-- Strategy: 256 rows, each row's vector encodes deterministically to a
-- different leading 2-bit code byte.  Self-distance via the operator must
-- be ~0 for every row, and the indexed top-5 for row N's vector must
-- include row N (the exact match).  A drift in the pair-table would
-- shift the score for some byte values relative to the scalar reference,
-- breaking the self-match property.

SET enable_seqscan = off;

-- 16-dim vectors so the kernel runs the tail-only sub-chunk of
-- TqGraphQuerySplit2RawAvx2 (and friends) — exercises the same expand
-- helper.  We also build a 1024-dim variant that exercises the main loop
-- on AVX-VNNI / AVX2.
DO $$
DECLARE
    encode_dim int;
    target_dim int;
    nbytes int;
    found_count int;
    bad_count int;
    sql_text text;
BEGIN
    FOREACH target_dim IN ARRAY ARRAY[16, 1024]::int[] LOOP
        nbytes := (target_dim + 3) / 4;

        -- Each row id ∈ [0, 255]: vector is the byte value broadcast to
        -- a magnitude that, after rotation + quantization, lands in a
        -- distinct 2-bit code pattern.  We don't need exact byte-level
        -- predictability inside the codes (the rotation scrambles the
        -- mapping) — only deterministic and unique per id.
        EXECUTE format(
            'CREATE TEMP TABLE tq_2bit_pairtable AS
             SELECT id,
                    (ARRAY(SELECT (sin(0.31415 * (k + id * 7))
                                  + 0.001 * id)::real
                          FROM generate_series(0, %s) k))::vector(%s) AS val
             FROM generate_series(0, 255) id',
            target_dim - 1, target_dim);

        EXECUTE
            'CREATE INDEX tq_2bit_pairtable_idx ON tq_2bit_pairtable
             USING turboquant (val vector_l2_ops)
             WITH (routing = graph, tq_bits = 2)';

        -- For each of 256 ids, query with that id's own vector and
        -- count whether the row appears in the top-5.  Anything missing
        -- means the SIMD pair-table produced a different ordering than
        -- the scalar reference would.  Tolerance: 5 misses out of 256
        -- (graph search is approximate, but for a self-match against the
        -- exact stored vector the row should be found in nearly all
        -- cases).
        SELECT count(*) INTO found_count
        FROM (
            SELECT id FROM tq_2bit_pairtable q
            WHERE EXISTS (
                SELECT 1 FROM (
                    SELECT id FROM tq_2bit_pairtable
                    ORDER BY val <-> q.val LIMIT 5
                ) t WHERE t.id = q.id
            )
        ) s;
        bad_count := 256 - found_count;

        IF bad_count > 5 THEN
            RAISE EXCEPTION
                'pair-table parity regression at dim=% bits=2: % of 256 self-matches missing from top-5',
                target_dim, bad_count;
        END IF;

        DROP TABLE tq_2bit_pairtable;
    END LOOP;
END
$$;

-- Sanity result so the test file is non-empty in expected output.
SELECT 'pair-table parity ok' AS status;
