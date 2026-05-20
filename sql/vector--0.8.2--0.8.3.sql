\echo Use "ALTER EXTENSION vector UPDATE TO '0.8.3'" to load this file. \quit

CREATE FUNCTION turbohybridhandler(internal) RETURNS index_am_handler
	AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE ACCESS METHOD turbohybrid TYPE INDEX HANDLER turbohybridhandler;

COMMENT ON ACCESS METHOD turbohybrid IS 'turbohybrid dense and BM25 hybrid index access method';

CREATE FUNCTION tq_simd_capabilities() RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

CREATE FUNCTION tq_last_simd_stats() RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

CREATE FUNCTION tq_debug_bm25_stats(regclass) RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

CREATE FUNCTION tq_debug_bm25_term_stats(regclass, text) RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION tq_debug_bm25_topk(regclass, tsquery, integer, boolean) RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION hybrid_last_scan_stats() RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

CREATE TYPE hybrid_query;

CREATE FUNCTION hybrid_query_in(cstring) RETURNS hybrid_query
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hybrid_query_out(hybrid_query) RETURNS cstring
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE hybrid_query (
	INPUT = hybrid_query_in,
	OUTPUT = hybrid_query_out,
	INTERNALLENGTH = variable,
	STORAGE = extended,
	ALIGNMENT = double
);

CREATE FUNCTION hybrid_query(
	vector_query vector DEFAULT NULL,
	text_query tsquery DEFAULT NULL,
	fusion text DEFAULT 'rrf',
	dense_weight float8 DEFAULT 1.0,
	bm25_weight float8 DEFAULT 1.0,
	alpha float8 DEFAULT NULL,
	rrf_k int4 DEFAULT NULL,
	dense_k int4 DEFAULT NULL,
	bm25_k int4 DEFAULT NULL,
	final_k int4 DEFAULT NULL,
	require_bm25_match bool DEFAULT false
) RETURNS hybrid_query
	AS 'MODULE_PATHNAME' LANGUAGE C STABLE PARALLEL SAFE;

CREATE FUNCTION hybrid_distance(vector, hybrid_query) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C STABLE STRICT PARALLEL UNSAFE;

CREATE FUNCTION hybrid_l2_distance(vector, hybrid_query) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C STABLE STRICT PARALLEL UNSAFE;

CREATE FUNCTION hybrid_negative_inner_product(vector, hybrid_query) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C STABLE STRICT PARALLEL UNSAFE;

CREATE FUNCTION hybrid_cosine_distance(vector, hybrid_query) RETURNS float8
	AS 'MODULE_PATHNAME' LANGUAGE C STABLE STRICT PARALLEL UNSAFE;

CREATE OPERATOR <~> (
	LEFTARG = vector, RIGHTARG = hybrid_query, PROCEDURE = hybrid_cosine_distance
);

CREATE OPERATOR <~-> (
	LEFTARG = vector, RIGHTARG = hybrid_query, PROCEDURE = hybrid_l2_distance
);

CREATE OPERATOR <~#> (
	LEFTARG = vector, RIGHTARG = hybrid_query, PROCEDURE = hybrid_negative_inner_product
);

CREATE OPERATOR CLASS vector_l2_hybrid_ops
	DEFAULT FOR TYPE vector USING turbohybrid AS
	OPERATOR 1 <~-> (vector, hybrid_query) FOR ORDER BY float_ops,
	FUNCTION 1 vector_l2_squared_distance(vector, vector),
	FUNCTION 3 l2_distance(vector, vector);

CREATE OPERATOR CLASS vector_ip_hybrid_ops
	FOR TYPE vector USING turbohybrid AS
	OPERATOR 1 <~#> (vector, hybrid_query) FOR ORDER BY float_ops,
	FUNCTION 1 vector_negative_inner_product(vector, vector),
	FUNCTION 3 vector_spherical_distance(vector, vector),
	FUNCTION 4 vector_norm(vector);

CREATE OPERATOR CLASS vector_cosine_hybrid_ops
	FOR TYPE vector USING turbohybrid AS
	OPERATOR 1 <~> (vector, hybrid_query) FOR ORDER BY float_ops,
	FUNCTION 1 vector_negative_inner_product(vector, vector),
	FUNCTION 2 vector_norm(vector),
	FUNCTION 3 vector_spherical_distance(vector, vector),
	FUNCTION 4 vector_norm(vector);

CREATE OPERATOR CLASS bm25_tsvector_ops
	FOR TYPE tsvector USING turbohybrid AS
	STORAGE tsvector;
