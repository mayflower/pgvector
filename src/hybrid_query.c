#include "postgres.h"

#include <math.h>

#include "fmgr.h"
#include "lib/stringinfo.h"
#include "nodes/execnodes.h"
#include "nodes/nodes.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "utils/builtins.h"
#include "utils/float.h"

#include "hybrid_query.h"
#include "tqhybrid.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hybrid_query_in);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hybrid_query_out);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hybrid_query);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hybrid_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hybrid_l2_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hybrid_negative_inner_product);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hybrid_cosine_distance);

static Size HybridQueryVectorOffset(void);
static Size HybridQueryTsQueryOffset(HybridQueryHeader *query);
static uint16 HybridQueryParseFusion(text *fusion);
static void HybridQueryCheckPositiveInt(const char *name, int32 value);
static void HybridQueryCheckNonNegativeInt(const char *name, int32 value);
static void HybridQueryRejectTextFallback(void);
static bool HybridQueryTextIndexOrderByContext(FunctionCallInfo fcinfo);
static void HybridCheckDims(Vector *a, Vector *b);
static double HybridL2SquaredDistance(Vector *a, Vector *b);
static double HybridNegativeInnerProduct(Vector *a, Vector *b);
static double HybridCosineDistance(Vector *a, Vector *b);

static Size
HybridQueryVectorOffset(void)
{
	return MAXALIGN(sizeof(HybridQueryHeader));
}

static Size
HybridQueryTsQueryOffset(HybridQueryHeader *query)
{
	return HybridQueryVectorOffset() + MAXALIGN(query->vectorBytes);
}

Vector *
HybridQueryGetVector(HybridQueryHeader *query)
{
	HybridQueryValidate(query);

	if ((query->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) == 0)
		return NULL;

	return (Vector *) ((char *) query + HybridQueryVectorOffset());
}

TSQuery
HybridQueryGetTsQuery(HybridQueryHeader *query)
{
	HybridQueryValidate(query);

	if ((query->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) == 0)
		return NULL;

	return (TSQuery) ((char *) query + HybridQueryTsQueryOffset(query));
}

const char *
HybridQueryFusionName(uint16 fusion)
{
	switch ((HybridFusionMode) fusion)
	{
		case HYBRID_FUSION_RRF:
			return "rrf";
		case HYBRID_FUSION_WEIGHTED:
			return "weighted";
	}

	return "unknown";
}

void
HybridQueryValidate(HybridQueryHeader *query)
{
	Size		expected;

	if (query == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("hybrid_query cannot be null")));

	if (query->version != HYBRID_QUERY_VERSION)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("unsupported hybrid_query version %u", query->version)));

	if (query->fusion != HYBRID_FUSION_RRF &&
		query->fusion != HYBRID_FUSION_WEIGHTED)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid hybrid_query fusion mode %u", query->fusion)));

	if (query->vectorBytes < 0 || query->tsqueryBytes < 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid hybrid_query payload size")));

	if ((query->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) == 0 &&
		query->vectorBytes != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("hybrid_query vector payload is inconsistent")));

	if ((query->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) == 0 &&
		query->tsqueryBytes != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("hybrid_query tsquery payload is inconsistent")));

	expected = HybridQueryTsQueryOffset(query) + MAXALIGN(query->tsqueryBytes);
	if (VARSIZE_ANY(query) < expected)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("truncated hybrid_query payload")));
}

Datum
hybrid_query_in(PG_FUNCTION_ARGS)
{
	char	   *input = PG_GETARG_CSTRING(0);

	(void) input;

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("text input is not supported for hybrid_query"),
			 errhint("Use the hybrid_query(...) constructor.")));

	PG_RETURN_NULL();
}

Datum
hybrid_query_out(PG_FUNCTION_ARGS)
{
	HybridQueryHeader *query = PG_GETARG_HYBRID_QUERY_P(0);
	StringInfoData buf;

	HybridQueryValidate(query);

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "hybrid_query(fusion=%s,vector=%s,tsquery=%s,dense_weight=%g,bm25_weight=%g,alpha=",
					 HybridQueryFusionName(query->fusion),
					 (query->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) ? "true" : "false",
					 (query->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) ? "true" : "false",
					 query->denseWeight,
					 query->bm25Weight);
	if (query->flags & HYBRID_QUERY_FLAG_ALPHA_IS_SET)
		appendStringInfo(&buf, "%g", query->alpha);
	else
		appendStringInfoString(&buf, "null");

	appendStringInfo(&buf,
					 ",rrf_k=%d,dense_k=%d,bm25_k=%d,final_k=",
					 query->rrfK,
					 query->denseK,
					 query->bm25K);
	if (query->flags & HYBRID_QUERY_FLAG_FINAL_K_IS_SET)
		appendStringInfo(&buf, "%d", query->finalK);
	else
		appendStringInfoString(&buf, "null");

	appendStringInfo(&buf,
					 ",require_bm25_match=%s)",
					 (query->flags & HYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH) ? "true" : "false");

	PG_RETURN_CSTRING(buf.data);
}

static uint16
HybridQueryParseFusion(text *fusion)
{
	char	   *name;
	uint16		result;

	if (fusion == NULL)
		return HYBRID_FUSION_RRF;

	name = text_to_cstring(fusion);
	if (pg_strcasecmp(name, "rrf") == 0)
		result = HYBRID_FUSION_RRF;
	else if (pg_strcasecmp(name, "weighted") == 0)
		result = HYBRID_FUSION_WEIGHTED;
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid hybrid fusion mode \"%s\"", name),
				 errdetail("Valid values are \"rrf\" and \"weighted\".")));

	pfree(name);
	return result;
}

static void
HybridQueryCheckPositiveInt(const char *name, int32 value)
{
	if (value <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("%s must be greater than zero", name)));
}

static void
HybridQueryCheckNonNegativeInt(const char *name, int32 value)
{
	if (value < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("%s must be greater than or equal to zero", name)));
}

typedef struct HybridQueryPlanCheck
{
	Node	   *expr;
	Oid			fnOid;
	bool		hasUserVisibleExpr;
	bool		hasIndexOrderByResjunkExpr;
} HybridQueryPlanCheck;

static List *
HybridQueryDistanceCallArgs(Node *expr, Oid *fnOid)
{
	if (expr == NULL)
		return NIL;

	if (IsA(expr, OpExpr))
	{
		OpExpr	   *op = (OpExpr *) expr;

		*fnOid = op->opfuncid;
		return op->args;
	}

	if (IsA(expr, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) expr;

		*fnOid = func->funcid;
		return func->args;
	}

	return NIL;
}

static bool
HybridQueryDistanceCallMatches(Node *candidate, HybridQueryPlanCheck *check)
{
	Oid			candidateFnOid = InvalidOid;
	Oid			exprFnOid = InvalidOid;
	List	   *candidateArgs;
	List	   *exprArgs;

	if (candidate == NULL || check->expr == NULL)
		return false;

	if (equal(candidate, check->expr))
		return true;

	candidateArgs = HybridQueryDistanceCallArgs(candidate, &candidateFnOid);
	exprArgs = HybridQueryDistanceCallArgs(check->expr, &exprFnOid);

	return OidIsValid(candidateFnOid) &&
		candidateFnOid == check->fnOid &&
		OidIsValid(exprFnOid) &&
		exprFnOid == check->fnOid &&
		equal(candidateArgs, exprArgs);
}

static bool
HybridQueryExprListContains(List *exprs, HybridQueryPlanCheck *check)
{
	ListCell   *lc;

	foreach(lc, exprs)
	{
		Node	   *candidate = (Node *) lfirst(lc);

		if (HybridQueryDistanceCallMatches(candidate, check))
			return true;
	}

	return false;
}

static void
HybridQueryInspectPlan(Plan *plan, HybridQueryPlanCheck *check)
{
	ListCell   *lc;

	if (plan == NULL)
		return;

	foreach(lc, plan->targetlist)
	{
		TargetEntry *tle = (TargetEntry *) lfirst(lc);

		if (tle == NULL || !IsA(tle, TargetEntry) ||
			!HybridQueryDistanceCallMatches((Node *) tle->expr, check))
			continue;

		if (!tle->resjunk)
		{
			check->hasUserVisibleExpr = true;
			continue;
		}

		if (IsA(plan, IndexScan) &&
			HybridQueryExprListContains(((IndexScan *) plan)->indexorderbyorig,
										check))
			check->hasIndexOrderByResjunkExpr = true;
	}

	HybridQueryInspectPlan(plan->lefttree, check);
	HybridQueryInspectPlan(plan->righttree, check);
}

static bool
HybridQueryTextIndexOrderByContext(FunctionCallInfo fcinfo)
{
	PlannedStmt *plannedstmt;
	HybridQueryPlanCheck check;

	if (fcinfo->flinfo == NULL || fcinfo->flinfo->fn_expr == NULL)
		return false;

	plannedstmt = TqHybridCurrentPlannedStmt();
	if (plannedstmt == NULL || plannedstmt->planTree == NULL)
		return false;

	check.expr = fcinfo->flinfo->fn_expr;
	check.fnOid = fcinfo->flinfo->fn_oid;
	check.hasUserVisibleExpr = false;
	check.hasIndexOrderByResjunkExpr = false;

	HybridQueryInspectPlan(plannedstmt->planTree, &check);

	return check.hasIndexOrderByResjunkExpr && !check.hasUserVisibleExpr;
}

static void
HybridQueryRejectTextFallback(void)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("hybrid text queries require a turbohybrid index scan"),
			 errdetail("The scalar hybrid distance function can only evaluate the vector payload.")));
}

Datum
hybrid_query(PG_FUNCTION_ARGS)
{
	struct varlena *vectorDatum = NULL;
	struct varlena *tsqueryDatum = NULL;
	int32		vectorBytes = 0;
	int32		tsqueryBytes = 0;
	uint16		flags = 0;
	uint16		fusion;
	float8		denseWeight;
	float8		bm25Weight;
	float8		alpha = 0.0;
	int32		rrfK;
	int32		denseK;
	int32		bm25K;
	int32		finalK = 0;
	bool		requireBm25Match;
	Size		totalSize;
	HybridQueryHeader *result;

	if (!PG_ARGISNULL(0))
	{
		vectorDatum = PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(0));
		vectorBytes = VARSIZE_ANY(vectorDatum);
		flags |= HYBRID_QUERY_FLAG_HAS_VECTOR;
	}

	if (!PG_ARGISNULL(1))
	{
		tsqueryDatum = PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(1));
		tsqueryBytes = VARSIZE_ANY(tsqueryDatum);
		flags |= HYBRID_QUERY_FLAG_HAS_TSQUERY;
	}

	if ((flags & (HYBRID_QUERY_FLAG_HAS_VECTOR | HYBRID_QUERY_FLAG_HAS_TSQUERY)) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("hybrid_query requires a vector_query or text_query")));

	fusion = HybridQueryParseFusion(PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));

	if (PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dense_weight cannot be null")));
	denseWeight = PG_GETARG_FLOAT8(3);
	if (denseWeight < 0 || isnan(denseWeight) || isinf(denseWeight))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dense_weight must be a finite non-negative value")));

	if (PG_ARGISNULL(4))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("bm25_weight cannot be null")));
	bm25Weight = PG_GETARG_FLOAT8(4);
	if (bm25Weight < 0 || isnan(bm25Weight) || isinf(bm25Weight))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("bm25_weight must be a finite non-negative value")));

	if (!PG_ARGISNULL(5))
	{
		alpha = PG_GETARG_FLOAT8(5);
		if (alpha < 0.0 || alpha > 1.0 || isnan(alpha) || isinf(alpha))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("alpha must be between 0 and 1")));
		flags |= HYBRID_QUERY_FLAG_ALPHA_IS_SET;
	}

	rrfK = PG_ARGISNULL(6) ? tqhybrid_default_rrf_k : PG_GETARG_INT32(6);
	HybridQueryCheckPositiveInt("rrf_k", rrfK);

	denseK = PG_ARGISNULL(7) ? tqhybrid_default_dense_k : PG_GETARG_INT32(7);
	HybridQueryCheckNonNegativeInt("dense_k", denseK);

	bm25K = PG_ARGISNULL(8) ? tqhybrid_default_bm25_k : PG_GETARG_INT32(8);
	HybridQueryCheckNonNegativeInt("bm25_k", bm25K);

	if (!PG_ARGISNULL(9))
	{
		finalK = PG_GETARG_INT32(9);
		HybridQueryCheckPositiveInt("final_k", finalK);
		flags |= HYBRID_QUERY_FLAG_FINAL_K_IS_SET;
	}

	if (PG_ARGISNULL(10))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("require_bm25_match cannot be null")));
	requireBm25Match = PG_GETARG_BOOL(10);
	if (requireBm25Match)
		flags |= HYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH;

	totalSize = HybridQueryVectorOffset() +
		MAXALIGN(vectorBytes) +
		MAXALIGN(tsqueryBytes);
	result = palloc0(totalSize);
	SET_VARSIZE(result, totalSize);
	result->version = HYBRID_QUERY_VERSION;
	result->flags = flags;
	result->fusion = fusion;
	result->denseWeight = denseWeight;
	result->bm25Weight = bm25Weight;
	result->alpha = alpha;
	result->rrfK = rrfK;
	result->denseK = denseK;
	result->bm25K = bm25K;
	result->finalK = finalK;
	result->vectorBytes = vectorBytes;
	result->tsqueryBytes = tsqueryBytes;

	if (vectorDatum != NULL)
	{
		memcpy((char *) result + HybridQueryVectorOffset(), vectorDatum, vectorBytes);
		pfree(vectorDatum);
	}
	if (tsqueryDatum != NULL)
	{
		memcpy((char *) result + HybridQueryTsQueryOffset(result), tsqueryDatum, tsqueryBytes);
		pfree(tsqueryDatum);
	}

	HybridQueryValidate(result);

	PG_RETURN_HYBRID_QUERY_P(result);
}

static double
HybridL2SquaredDistance(Vector *a, Vector *b)
{
	double		distance = 0.0;

	HybridCheckDims(a, b);

	for (int i = 0; i < a->dim; i++)
	{
		double		diff = (double) a->x[i] - (double) b->x[i];

		distance += diff * diff;
	}

	return distance;
}

static double
HybridNegativeInnerProduct(Vector *a, Vector *b)
{
	double		distance = 0.0;

	HybridCheckDims(a, b);

	for (int i = 0; i < a->dim; i++)
		distance += (double) a->x[i] * (double) b->x[i];

	return -distance;
}

static void
HybridCheckDims(Vector *a, Vector *b)
{
	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", a->dim, b->dim)));
}

static double
HybridCosineDistance(Vector *a, Vector *b)
{
	double		similarity = 0.0;
	double		norma = 0.0;
	double		normb = 0.0;

	HybridCheckDims(a, b);

	for (int i = 0; i < a->dim; i++)
	{
		similarity += (double) a->x[i] * (double) b->x[i];
		norma += (double) a->x[i] * (double) a->x[i];
		normb += (double) b->x[i] * (double) b->x[i];
	}

	similarity = similarity / sqrt(norma * normb);

	if (similarity > 1.0)
		similarity = 1.0;
	else if (similarity < -1.0)
		similarity = -1.0;

	return 1.0 - similarity;
}

Datum
hybrid_l2_distance(PG_FUNCTION_ARGS)
{
	Vector	   *value = PG_GETARG_VECTOR_P(0);
	HybridQueryHeader *query = PG_GETARG_HYBRID_QUERY_P(1);
	Vector	   *vectorQuery;

	HybridQueryValidate(query);
	if (HybridQueryGetTsQuery(query) != NULL)
	{
		if (!HybridQueryTextIndexOrderByContext(fcinfo))
			HybridQueryRejectTextFallback();
		PG_RETURN_FLOAT8(0.0);
	}
	vectorQuery = HybridQueryGetVector(query);

	if (vectorQuery == NULL)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(sqrt(HybridL2SquaredDistance(value, vectorQuery)));
}

Datum
hybrid_negative_inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *value = PG_GETARG_VECTOR_P(0);
	HybridQueryHeader *query = PG_GETARG_HYBRID_QUERY_P(1);
	Vector	   *vectorQuery;

	HybridQueryValidate(query);
	if (HybridQueryGetTsQuery(query) != NULL)
	{
		if (!HybridQueryTextIndexOrderByContext(fcinfo))
			HybridQueryRejectTextFallback();
		PG_RETURN_FLOAT8(0.0);
	}
	vectorQuery = HybridQueryGetVector(query);

	if (vectorQuery == NULL)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(HybridNegativeInnerProduct(value, vectorQuery));
}

Datum
hybrid_cosine_distance(PG_FUNCTION_ARGS)
{
	Vector	   *value = PG_GETARG_VECTOR_P(0);
	HybridQueryHeader *query = PG_GETARG_HYBRID_QUERY_P(1);
	Vector	   *vectorQuery;

	HybridQueryValidate(query);
	if (HybridQueryGetTsQuery(query) != NULL)
	{
		if (!HybridQueryTextIndexOrderByContext(fcinfo))
			HybridQueryRejectTextFallback();
		PG_RETURN_FLOAT8(0.0);
	}
	vectorQuery = HybridQueryGetVector(query);

	if (vectorQuery == NULL)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(HybridCosineDistance(value, vectorQuery));
}

Datum
hybrid_distance(PG_FUNCTION_ARGS)
{
	return hybrid_cosine_distance(fcinfo);
}
