#include "postgres.h"

#include <string.h>

#include "access/genam.h"
#include "catalog/pg_type_d.h"
#include "storage/bufmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/rel.h"

#include "tqgraph.h"
#include "tqgraph_psquare.h"
#include "tqgraph_score.h"

static bool
TqGraphReadFirstCodeTuple(Relation index, HnswMetaPageData *meta,
						  bool tqWeighted, uint8 *code, Size codeBytes,
						  uint32 *nodeId, float *ecCorrection)
{
	int			codeTuplesPerPage;
	int			codePageCount;
	BlockNumber *codeBlknos;
	bool		found = false;

	codeTuplesPerPage =
		TqGraphTuplesPerPage(TqGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  tqWeighted));
	codePageCount = TqGraphPageCount(meta->tqNodeCount, codeTuplesPerPage);
	codeBlknos = palloc(sizeof(BlockNumber) * codePageCount);
	TqGraphInitBlockMap(codeBlknos, codePageCount);

	for (int pageNo = 0; pageNo < codePageCount && !found; pageNo++)
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber blkno;

		if (!TqGraphResolveChainBlockNumber(index, meta->tqCodeStartBlkno,
											 pageNo, codePageCount,
											 HNSW_PAGE_KIND_TQ_CODE,
											 codeBlknos, &blkno))
			break;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);

		if ((opaque->pageKind & HNSW_PAGE_KIND_MASK) != HNSW_PAGE_KIND_TQ_CODE)
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			TqGraphCodeTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqGraphCodeTuple) PageGetItem(page, iid);
			if (tuple->type != TQ_GRAPH_CODE_TUPLE_TYPE)
				continue;

			memcpy(code, TqGraphTupleCode(tuple, meta->tqPayloadBytes, tqWeighted),
				   codeBytes);
			*ecCorrection = TqGraphTupleEcCorrection(tuple, tqWeighted);
			*nodeId = tuple->nodeId;
			found = true;
			break;
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(codeBlknos);
	return found;
}

/*
 * Diagnostic SQL hook: tq_debug_weighted_self_score(regclass) → jsonb
 *
 * Loads the first node in the given
 * index, computes:
 *   raw_self     — Σ X+[d]² · D'²[d]      (TqGraphCodeCodeWeightedRawScalar)
 *   mm_const     — Σ ecShift[d]²          (TqGraphMmConstScalar)
 *   ec_correction— ⟨X+, M⟩ stored on disk
 *   weighted_self— raw_self + 2 · ec_correction - mm_const
 *
 * For an index built with (tq_weighted = on), all four numbers should
 * be finite; weighted_self is meaningful as a self-similarity in the
 * TQ+ centered/weighted space.  For an index without tq_weighted the
 * function reports tq_weighted=false and returns zeros for the
 * derived quantities (the caller can detect the mode from the JSON).
 *
 * This is the parity-test entry point for weighted scoring: a SQL test
 * exercises it on a real index and asserts the values are sane.
 */
/*
 * P-square estimator diagnostic: tq_debug_psquare_estimate(observations float8[],
 * quantile float8) → float8.
 *
 * Constructs a P-square estimator at the given quantile, pushes every
 * observation in the array, returns the running estimate.  Used by the
 * regression test to confirm the streaming estimator matches the
 * in-memory quantile of a known distribution within ~5 % relative.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tq_debug_psquare_estimate);
Datum
tq_debug_psquare_estimate(PG_FUNCTION_ARGS)
{
	ArrayType  *observations = PG_GETARG_ARRAYTYPE_P(0);
	float8		quantile = PG_GETARG_FLOAT8(1);
	int			nelems;
	Datum	   *elems;
	bool	   *nulls;
	TqPSquareState p;

	if (quantile <= 0.0 || quantile >= 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("quantile must be in (0, 1), got %g", quantile)));

	deconstruct_array(observations, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL,
					  TYPALIGN_DOUBLE, &elems, &nulls, &nelems);

	TqPSquareInit(&p, quantile);
	for (int i = 0; i < nelems; i++)
	{
		if (nulls != NULL && nulls[i])
			continue;
		TqPSquarePush(&p, DatumGetFloat8(elems[i]));
	}

	PG_RETURN_FLOAT8(TqPSquareEstimate(&p));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tq_debug_weighted_self_score);
Datum
tq_debug_weighted_self_score(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Relation	index;
	HnswMetaPageData meta;
	float	   *ecShift = NULL;
	float	   *ecScale = NULL;
	bool		tqWeighted;
	uint8	   *code = NULL;
	float		ecCorrection = 0.0f;
	bool		found = false;
	uint32		nodeId = 0;
	double		rawSelf = 0.0;
	double		mmConst = 0.0;
	double		weightedSelf = 0.0;
	StringInfoData json;
	Size		codeBytes;

	index = index_open(indexOid, AccessShareLock);

	if (!TqGraphReadMeta(index, &meta) || meta.tqNodeCount == 0 ||
		!BlockNumberIsValid(meta.tqCodeStartBlkno))
	{
		index_close(index, AccessShareLock);
		PG_RETURN_NULL();
	}

	tqWeighted = (meta.tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0;
	codeBytes = TqGraphCodeBytesForBits(meta.dimensions, meta.tqBits);
	code = palloc(codeBytes);

	if ((meta.tqFlags & TQ_GRAPH_TQ_PLUS) != 0)
		(void) TqGraphLoadCorrection(index, meta.dimensions, &ecShift, &ecScale);

	found = TqGraphReadFirstCodeTuple(index, &meta, tqWeighted, code, codeBytes,
									  &nodeId, &ecCorrection);

	if (!found)
	{
		pfree(code);
		index_close(index, AccessShareLock);
		PG_RETURN_NULL();
	}

	if (ecScale != NULL)
		rawSelf = TqGraphCodeCodeWeightedRawScalar(code, code, meta.dimensions,
												   meta.tqBits, ecScale);
	if (ecShift != NULL)
		mmConst = TqGraphMmConstScalar(ecShift, meta.dimensions);

	weightedSelf = rawSelf + 2.0 * (double) ecCorrection - mmConst;

	/*
	 * Weighted SIMD parity hook: if the weighted SIMD kernels are available,
	 * recompute raw_self via the dispatched SIMD path on a freshly
	 * quantized i16 weight array.  The SQL test asserts that the SIMD
	 * raw matches the scalar raw within the i16 weight quantization
	 * tolerance — catches regressions in the SIMD path automatically.
	 */
	{
		double		simdRaw = 0.0;
		bool		simdRan = false;

		if (ecScale != NULL && tqWeighted)
			simdRan = TqGraphCodeCodeWeightedRawSimdSelf(code, meta.dimensions,
														 meta.tqBits, ecScale,
														 &simdRaw);

		pfree(code);
		if (ecShift != NULL)
			pfree(ecShift);
		if (ecScale != NULL)
			pfree(ecScale);
		index_close(index, AccessShareLock);

		initStringInfo(&json);
		appendStringInfo(&json,
						 "{\"node_id\":%u,"
						 "\"tq_weighted\":%s,"
						 "\"tq_plus\":%s,"
						 "\"dimensions\":%u,"
						 "\"tq_bits\":%u,"
						 "\"raw_self\":%.9g,"
						 "\"raw_self_simd\":%.9g,"
						 "\"simd_ran\":%s,"
						 "\"ec_correction\":%.9g,"
						 "\"mm_const\":%.9g,"
						 "\"weighted_self\":%.9g}",
						 nodeId,
						 tqWeighted ? "true" : "false",
						 (meta.tqFlags & TQ_GRAPH_TQ_PLUS) != 0 ? "true" : "false",
						 meta.dimensions,
						 meta.tqBits,
						 rawSelf,
						 simdRaw,
						 simdRan ? "true" : "false",
						 (double) ecCorrection,
						 mmConst,
						 weightedSelf);
	}

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}
