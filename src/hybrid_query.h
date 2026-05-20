#ifndef HYBRID_QUERY_H
#define HYBRID_QUERY_H

#include "postgres.h"

#include "tsearch/ts_type.h"
#include "vector.h"

#define HYBRID_QUERY_VERSION 1

#define HYBRID_QUERY_FLAG_HAS_VECTOR			0x0001
#define HYBRID_QUERY_FLAG_HAS_TSQUERY			0x0002
#define HYBRID_QUERY_FLAG_ALPHA_IS_SET			0x0004
#define HYBRID_QUERY_FLAG_FINAL_K_IS_SET		0x0008
#define HYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH	0x0010

typedef enum HybridFusionMode
{
	HYBRID_FUSION_RRF = 1,
	HYBRID_FUSION_WEIGHTED = 2
} HybridFusionMode;

typedef struct HybridQueryHeader
{
	int32		vl_len_;
	uint16		version;
	uint16		flags;
	uint16		fusion;
	uint16		reserved;
	float8		denseWeight;
	float8		bm25Weight;
	float8		alpha;
	int32		rrfK;
	int32		denseK;
	int32		bm25K;
	int32		finalK;
	int32		vectorBytes;
	int32		tsqueryBytes;
	/* payload starts at MAXALIGN(sizeof(HybridQueryHeader)) */
} HybridQueryHeader;

#define DatumGetHybridQuery(x) ((HybridQueryHeader *) PG_DETOAST_DATUM(x))
#define PG_GETARG_HYBRID_QUERY_P(x) DatumGetHybridQuery(PG_GETARG_DATUM(x))
#define PG_RETURN_HYBRID_QUERY_P(x) PG_RETURN_POINTER(x)

Vector	   *HybridQueryGetVector(HybridQueryHeader *query);
TSQuery		HybridQueryGetTsQuery(HybridQueryHeader *query);
void		HybridQueryValidate(HybridQueryHeader *query);
const char *HybridQueryFusionName(uint16 fusion);

#endif
