-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.8.2'" to load this file. \quit

CREATE FUNCTION turboquanthandler(internal) RETURNS index_am_handler
	AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE ACCESS METHOD turboquant TYPE INDEX HANDLER turboquanthandler;

COMMENT ON ACCESS METHOD turboquant IS 'turboquant graph index access method';

CREATE FUNCTION tq_last_scan_stats() RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

CREATE FUNCTION tq_index_stats(regclass) RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

CREATE FUNCTION tq_debug_weighted_self_score(regclass) RETURNS jsonb
	AS 'MODULE_PATHNAME' LANGUAGE C VOLATILE;

CREATE FUNCTION tq_debug_psquare_estimate(double precision[], double precision)
	RETURNS double precision
	AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT;

CREATE OPERATOR CLASS vector_l2_ops
	DEFAULT FOR TYPE vector USING turboquant AS
	OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_l2_squared_distance(vector, vector),
	FUNCTION 3 l2_distance(vector, vector);

CREATE OPERATOR CLASS vector_ip_ops
	FOR TYPE vector USING turboquant AS
	OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_negative_inner_product(vector, vector),
	FUNCTION 3 vector_spherical_distance(vector, vector),
	FUNCTION 4 vector_norm(vector);

CREATE OPERATOR CLASS vector_cosine_ops
	FOR TYPE vector USING turboquant AS
	OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_negative_inner_product(vector, vector),
	FUNCTION 2 vector_norm(vector),
	FUNCTION 3 vector_spherical_distance(vector, vector),
	FUNCTION 4 vector_norm(vector);

CREATE OPERATOR CLASS vector_l1_ops
	FOR TYPE vector USING turboquant AS
	OPERATOR 1 <+> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 l1_distance(vector, vector);
