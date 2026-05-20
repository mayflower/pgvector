-- SQL-level reference baseline for Postgres FTS + pgvector RRF.
-- Replace :table, :id_col, :vector_col, :tsvector_col, :query_vector,
-- :tsquery, :dense_k, :bm25_k, :rrf_k, and :final_k before execution.

WITH dense AS MATERIALIZED (
    SELECT :id_col AS id, row_number() OVER () AS rank
    FROM (
        SELECT :id_col
        FROM :table
        ORDER BY :vector_col <=> :query_vector
        LIMIT :dense_k
    ) s
),
lexical AS MATERIALIZED (
    SELECT :id_col AS id, row_number() OVER () AS rank
    FROM (
        SELECT :id_col
        FROM :table
        WHERE :tsvector_col @@ :tsquery
        ORDER BY ts_rank_cd(:tsvector_col, :tsquery) DESC, :id_col
        LIMIT :bm25_k
    ) s
)
SELECT COALESCE(dense.id, lexical.id) AS id,
       COALESCE(1.0 / (:rrf_k + dense.rank), 0.0) +
       COALESCE(1.0 / (:rrf_k + lexical.rank), 0.0) AS score
FROM dense
FULL OUTER JOIN lexical USING (id)
ORDER BY score DESC, id
LIMIT :final_k;
