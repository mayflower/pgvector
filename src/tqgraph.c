#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "access/generic_xlog.h"
#include "access/genam.h"
#include "access/tableam.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "port/pg_bitutils.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/array.h"
#include "utils/datum.h"
#include "utils/fmgrprotos.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"


#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#endif


#include "tqgraph.h"
#include "tqgraph_psquare.h"
#include "tqgraph_score.h"

static int	TqGraphResultCompare(const void *a, const void *b);
static bool TqGraphEntryAlreadySelected(TqGraphFrontierItem *entries, int entryCount,
										uint32 nodeId);



static int
TqGraphIndexPayloadCount(Relation index)
{
	int			totalAttrs = IndexRelationGetNumberOfAttributes(index);
	int			keyAttrs = IndexRelationGetNumberOfKeyAttributes(index);
	int			payloadCount = totalAttrs - keyAttrs;

	if (payloadCount < 0)
		payloadCount = 0;
	if (payloadCount > TQ_GRAPH_MAX_PAYLOADS)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant native graph supports at most %d included payload columns",
						TQ_GRAPH_MAX_PAYLOADS)));

	for (int i = 0; i < payloadCount; i++)
	{
		Form_pg_attribute attr =
			TupleDescAttr(RelationGetDescr(index), keyAttrs + i);

		if (attr->atttypid != INT4OID)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("turboquant native graph INCLUDE payload columns must be integer")));
	}

	return payloadCount;
}

void
TqGraphCopyPayloadValues(TqGraphBuildState *state, int32 *payloads,
						 uint16 *payloadMask, Datum *values, bool *isnull)
{
	int			keyAttrs;

	*payloadMask = 0;
	if (state->payloadCount <= 0 || values == NULL || isnull == NULL)
		return;

	keyAttrs = state->indexInfo != NULL ? state->indexInfo->ii_NumIndexKeyAttrs :
		IndexRelationGetNumberOfKeyAttributes(state->index);

	for (int i = 0; i < state->payloadCount; i++)
	{
		int			attrIndex = keyAttrs + i;

		if (isnull[attrIndex])
			continue;

		payloads[i] = DatumGetInt32(values[attrIndex]);
		*payloadMask |= (uint16) (1U << i);
	}
}

static int
TqGraphPayloadSlotForHeapAttr(Relation index, AttrNumber heapAttno)
{
	int			totalAttrs = IndexRelationGetNumberOfAttributes(index);
	int			keyAttrs = IndexRelationGetNumberOfKeyAttributes(index);

	for (int i = keyAttrs; i < totalAttrs; i++)
	{
		if (index->rd_index->indkey.values[i] == heapAttno)
			return i - keyAttrs;
	}

	return -1;
}

static bool
TqGraphNodeMatchesPayload(TqGraphScanNode *node, int payloadSlot, int32 payloadValue)
{
	if (payloadSlot < 0)
		return true;
	if (payloadSlot >= TQ_GRAPH_MAX_PAYLOADS || node->payloads == NULL)
		return false;
	if ((node->payloadMask & (uint16) (1U << payloadSlot)) == 0)
		return false;

	return node->payloads[payloadSlot] == payloadValue;
}


static uint64
TqGraphMix64(uint64 x)
{
	x += UINT64CONST(0x9e3779b97f4a7c15);
	x = (x ^ (x >> 30)) * UINT64CONST(0xbf58476d1ce4e5b9);
	x = (x ^ (x >> 27)) * UINT64CONST(0x94d049bb133111eb);
	return x ^ (x >> 31);
}

int
TqGraphPickLevel(uint32 nodeId, int m)
{
	uint64		mixed = TqGraphMix64(nodeId);
	double		u = ((double) ((mixed >> 11) + 1)) * (1.0 / 9007199254740992.0);
	int			level = (int) floor(-log(u) * HnswGetMl(Max(m, 2)));

	return Min(level, Min(HnswGetMaxLevel(m), TQ_GRAPH_MAX_STORED_LEVEL));
}


static Size
TqGraphCorrectionTupleSize(int count)
{
	return MAXALIGN(offsetof(TqGraphCorrectionTupleData, values) +
					(sizeof(float) * count));
}





static bool
TqGraphBuildNodeHasLevel(TqGraphBuildState *state, uint32 nodeId, int level)
{
	return nodeId < state->nodeCount && level >= 0 && level <= state->nodes[nodeId].level;
}

static int
TqGraphFrontierCompare(const void *a, const void *b)
{
	const TqGraphFrontierItem *ia = (const TqGraphFrontierItem *) a;
	const TqGraphFrontierItem *ib = (const TqGraphFrontierItem *) b;

	if (ia->distance < ib->distance)
		return -1;
	if (ia->distance > ib->distance)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}

static int
TqGraphBuildOrderCompare(const void *a, const void *b)
{
	const TqGraphBuildOrderItem *ia = (const TqGraphBuildOrderItem *) a;
	const TqGraphBuildOrderItem *ib = (const TqGraphBuildOrderItem *) b;

	if (ia->key < ib->key)
		return -1;
	if (ia->key > ib->key)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}


static bool
TqGraphFrontierLess(TqGraphFrontierItem a, TqGraphFrontierItem b)
{
	if (a.distance != b.distance)
		return a.distance < b.distance;
	return a.nodeId < b.nodeId;
}

static bool
TqGraphFrontierGreater(TqGraphFrontierItem a, TqGraphFrontierItem b)
{
	if (a.distance != b.distance)
		return a.distance > b.distance;
	return a.nodeId > b.nodeId;
}

static void
TqGraphFrontierSwap(TqGraphFrontierItem *a, TqGraphFrontierItem *b)
{
	TqGraphFrontierItem tmp = *a;

	*a = *b;
	*b = tmp;
}

static void
TqGraphFrontierHeapSiftUp(TqGraphFrontierItem *heap, int idx, bool minHeap)
{
	while (idx > 0)
	{
		int			parent = (idx - 1) / 2;
		bool		before = minHeap ?
			TqGraphFrontierLess(heap[idx], heap[parent]) :
			TqGraphFrontierGreater(heap[idx], heap[parent]);

		if (!before)
			break;

		TqGraphFrontierSwap(&heap[idx], &heap[parent]);
		idx = parent;
	}
}

static void
TqGraphFrontierHeapSiftDown(TqGraphFrontierItem *heap, int count, int idx,
							bool minHeap)
{
	for (;;)
	{
		int			left = idx * 2 + 1;
		int			right = left + 1;
		int			best = idx;

		if (left < count)
		{
			bool		before = minHeap ?
				TqGraphFrontierLess(heap[left], heap[best]) :
				TqGraphFrontierGreater(heap[left], heap[best]);

			if (before)
				best = left;
		}

		if (right < count)
		{
			bool		before = minHeap ?
				TqGraphFrontierLess(heap[right], heap[best]) :
				TqGraphFrontierGreater(heap[right], heap[best]);

			if (before)
				best = right;
		}

		if (best == idx)
			break;

		TqGraphFrontierSwap(&heap[idx], &heap[best]);
		idx = best;
	}
}

static void
TqGraphFrontierHeapPush(TqGraphFrontierItem *heap, int *count,
						TqGraphFrontierItem item, bool minHeap)
{
	heap[*count] = item;
	(*count)++;
	TqGraphFrontierHeapSiftUp(heap, *count - 1, minHeap);
}

static int
TqGraphInitialFrontierCapacity(uint32 nodeCount, int searchEf, int entryCount,
							   int maxNeighbors)
{
	int			capacity;

	if (nodeCount == 0)
		return 0;

	capacity = Max(8, searchEf + entryCount + maxNeighbors);
	capacity = Max(capacity, (searchEf * 2) + entryCount);
	if ((uint32) capacity > nodeCount)
		capacity = (int) nodeCount;

	return capacity;
}

static void
TqGraphFrontierHeapPushGrowing(TqGraphFrontierItem **heap, int *count,
							   int *capacity, int maxCapacity,
							   TqGraphFrontierItem item, bool minHeap)
{
	if (*count >= *capacity)
	{
		int			newCapacity;

		if (*capacity >= maxCapacity)
			elog(ERROR, "turboquant graph frontier capacity exceeded");

		newCapacity = Max(8, *capacity * 2);
		if (newCapacity < *capacity || newCapacity > maxCapacity)
			newCapacity = maxCapacity;

		*heap = repalloc(*heap, sizeof(TqGraphFrontierItem) * newCapacity);
		*capacity = newCapacity;
	}

	TqGraphFrontierHeapPush(*heap, count, item, minHeap);
}

static TqGraphFrontierItem
TqGraphFrontierHeapPop(TqGraphFrontierItem *heap, int *count, bool minHeap)
{
	TqGraphFrontierItem item = heap[0];

	(*count)--;
	if (*count > 0)
	{
		heap[0] = heap[*count];
		TqGraphFrontierHeapSiftDown(heap, *count, 0, minHeap);
	}

	return item;
}

static void
TqGraphFrontierHeapReplaceRoot(TqGraphFrontierItem *heap, int count,
							   TqGraphFrontierItem item, bool minHeap)
{
	heap[0] = item;
	TqGraphFrontierHeapSiftDown(heap, count, 0, minHeap);
}

static bool
TqGraphOfferNearest(TqGraphFrontierItem *heap, int capacity, int *count,
					uint32 nodeId, double distance)
{
	TqGraphFrontierItem item;

	if (capacity <= 0)
		return false;

	item.nodeId = nodeId;
	item.distance = distance;

	if (*count < capacity)
	{
		TqGraphFrontierHeapPush(heap, count, item, false);
		return true;
	}

	if (TqGraphFrontierLess(item, heap[0]))
	{
		TqGraphFrontierHeapReplaceRoot(heap, *count, item, false);
		return true;
	}

	return false;
}

static double
TqGraphResultDistance(HnswScanOpaque so, Datum query, TqGraphScanNode *node,
					  double packedDistance, bool *exactScored)
{
	double		exactDistance;

	if (TqGraphCachedExactNodeDistance(so, query, node, &exactDistance))
	{
		*exactScored = true;
		return exactDistance;
	}

	*exactScored = false;
	return packedDistance;
}

static double
TqGraphEntryDistance(HnswScanOpaque so, Datum query, TqGraphScanNode *node)
{
	double		exactDistance;

	if (TqGraphCachedExactNodeDistance(so, query, node, &exactDistance))
		return exactDistance;

	return TqGraphScoreNode(so, node);
}

static void
TqGraphEnsureNodeCapacity(TqGraphBuildState *state)
{
	if (state->nodeCount < state->nodeCapacity)
		return;

	state->nodeCapacity = state->nodeCapacity == 0 ? 1024 : state->nodeCapacity * 2;
	state->nodes = repalloc(state->nodes, sizeof(TqGraphBuildNode) * state->nodeCapacity);
}

static void
TqGraphAppendBuildNode(TqGraphBuildState *state, ItemPointer tid, Datum value,
					   Datum *values, bool *isnull)
{
	Vector	   *vector = (Vector *) DatumGetPointer(value);
	TqGraphBuildNode *node;
	Size		vectorSize;
	uint32		nodeId = state->nodeCount;
	int			level;

	if (state->dimensions == 0)
		state->dimensions = vector->dim;
	else if (state->dimensions != vector->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions are not supported in the same turboquant graph")));

	TqGraphEnsureNodeCapacity(state);
	node = &state->nodes[state->nodeCount++];
	level = TqGraphPickLevel(nodeId, state->m);

	vectorSize = VECTOR_SIZE(vector->dim);
	node->vector = palloc(vectorSize);
	memcpy(node->vector, vector, vectorSize);
	node->code = palloc(TqGraphCodeBytesForBits(vector->dim, state->tqBits));
	if (state->payloadCount > 0)
	{
		node->payloads = palloc0(state->payloadBytes);
		TqGraphCopyPayloadValues(state, node->payloads, &node->payloadMask,
								 values, isnull);
	}
	node->level = level;
	node->norm = TqGraphVectorNorm(vector);
	node->flags = 0;
	node->heaptid = *tid;
	node->neighbors = palloc0(sizeof(uint32 *) * (level + 1));
	node->neighborCounts = palloc0(sizeof(int) * (level + 1));
	for (int i = 0; i <= level; i++)
		node->neighbors[i] = palloc0(sizeof(uint32) * (TqGraphLevelM(state->m, i) + 1));
	state->maxLevel = Max(state->maxLevel, level);
}

static void
TqGraphBuildCallback(Relation index, ItemPointer tid, Datum *values,
					 bool *isnull, bool tupleIsAlive, void *opaque)
{
	TqGraphBuildState *state = (TqGraphBuildState *) opaque;
	MemoryContext oldCtx;
	Datum		value;

	(void) index;
	(void) tupleIsAlive;

	if (isnull[0])
		return;

	oldCtx = MemoryContextSwitchTo(state->ctx);
	if (HnswFormIndexValue(&value, values, isnull, state->typeInfo, &state->support))
	{
		TqGraphAppendBuildNode(state, tid, value, values, isnull);
		pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, state->nodeCount);
	}
	MemoryContextSwitchTo(oldCtx);
}

/*
 * Extended P-square one-quantile estimator (Jain & Chlamtac
 * 1985, N=5 markers).  Streaming, fixed memory per estimator,
 * independent of input size.  Used to fit the
 * quantile-anchored ecShift / ecScale per coord.
 *
 * Reference: <https://www.cse.wustl.edu/~jain/papers/ftp/psqr.pdf>.
 *
 * State per estimator: 5 marker heights + 5 marker positions + a
 * count + the target quantile.  Once `count >= 5` the estimator is
 * fully initialized and `Estimate` returns the running quantile;
 * before that it falls back to the median of observed values.
 *
 * Only base C math — no SIMD or threads.  At dim=1536 with 2
 * estimators per coord (q_lo, q_hi) the per-push cost is ~50 ns × 2
 * × 1536 ≈ 154 µs per vector, dominated by the FMA in
 * update_desired_positions and the parabolic adjust in adjust_step.
 * For a 57k-vector FIQA build that's ~9 s of fit-time pre-pass —
 * one-time cost, no runtime impact.
 */

/*
 * c_outer for the bit-width's Lloyd-Max codebook — the outermost
 * centroid magnitude.  Already encoded in the file as the denominator
 * of the per-bit CODEBOOK_SCALE, but we need the raw float here for
 * the quantile fit math.
 */
static double
TqGraphCodebookOuter(int bits)
{
	if (bits == 2)
		return 1.510;
	if (bits == 1)
		return 1.000;
	return 2.733;					/* 4-bit default */
}

/*
 * Phi(x) — standard normal CDF.  Used to map the codebook outermost
 * centroid magnitude to the symmetric quantile probability that the
 * empirical fit anchors on.  For 4-bit, c_outer = 2.733 →
 * p_outer = 0.9968...; for 2-bit, c_outer = 1.510 → 0.9345.
 */
static double
TqGraphPhi(double x)
{
	return 0.5 * (1.0 + erf(x / 1.41421356237309504880));
}

/*
 * Quantile-anchored ecShift / ecScale fit.
 *
 * Streams every build vector through TqPreprocessVector, then pushes
 * each per-coord rotated value into a pair of P-square estimators
 * (q_lo, q_hi).  After all observations:
 *
 *    shift[d] = -(q_lo[d] + q_hi[d]) / 2
 *    scale[d] = (2 · c_outer) / (q_hi[d] - q_lo[d])     (width-floor)
 *
 * The shift/scale arrays are written into state->ecShift /
 * state->ecScale — same downstream consumers as the Welford path.
 */
static void
TqGraphFitCorrectionQuantile(TqGraphBuildState *state)
{
	const double MIN_QUANTILE_WIDTH = 1e-3;
	double		c_outer;
	double		p_outer;
	double		q_lo_target;
	double		q_hi_target;
	TqPSquareState *qLo;
	TqPSquareState *qHi;
	double	   *buffer;

	c_outer = TqGraphCodebookOuter(state->tqBits);
	p_outer = TqGraphPhi(c_outer);
	q_lo_target = 1.0 - p_outer;
	q_hi_target = p_outer;

	qLo = palloc0(sizeof(TqPSquareState) * state->dimensions);
	qHi = palloc0(sizeof(TqPSquareState) * state->dimensions);
	for (int dim = 0; dim < state->dimensions; dim++)
	{
		TqPSquareInit(&qLo[dim], q_lo_target);
		TqPSquareInit(&qHi[dim], q_hi_target);
	}

	buffer = palloc(sizeof(double) * state->dimensions);
	for (uint32 row = 0; row < state->nodeCount; row++)
	{
		TqPreprocessVector(state->nodes[row].vector, buffer);
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			TqPSquarePush(&qLo[dim], buffer[dim]);
			TqPSquarePush(&qHi[dim], buffer[dim]);
		}
	}
	pfree(buffer);

	state->ecShift = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	state->ecScale = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	for (int dim = 0; dim < state->dimensions; dim++)
	{
		double		q_lo = TqPSquareEstimate(&qLo[dim]);
		double		q_hi = TqPSquareEstimate(&qHi[dim]);
		double		denom = q_hi - q_lo;

		state->ecShift[dim] = (float) (-0.5 * (q_lo + q_hi));
		if (denom > MIN_QUANTILE_WIDTH)
			state->ecScale[dim] = (float) ((2.0 * c_outer) / denom);
		else
			state->ecScale[dim] = 1.0f;
	}

	pfree(qLo);
	pfree(qHi);
}

static void
TqGraphFitCorrection(TqGraphBuildState *state)
{
	double	   *mean;
	double	   *m2;
	double	   *buffer;

	if (state->nodeCount == 0 || state->dimensions <= 0 ||
		(state->scoreMode != TQ_SCORE_COSINE && state->scoreMode != TQ_SCORE_IP))
		return;

	if (state->tqQuantileFit)
	{
		TqGraphFitCorrectionQuantile(state);
		goto post_fit;
	}

	mean = palloc0(sizeof(double) * state->dimensions);
	m2 = palloc0(sizeof(double) * state->dimensions);
	buffer = palloc(sizeof(double) * state->dimensions);

	for (uint32 row = 0; row < state->nodeCount; row++)
	{
		double		n = (double) row + 1.0;

		TqPreprocessVector(state->nodes[row].vector, buffer);
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			double		value = buffer[dim];
			double		delta = value - mean[dim];

			mean[dim] += delta / n;
			m2[dim] += delta * (value - mean[dim]);
		}
	}

	state->ecShift = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	state->ecScale = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	for (int dim = 0; dim < state->dimensions; dim++)
	{
		double		variance = state->nodeCount > 1 ? m2[dim] / ((double) state->nodeCount - 1.0) : 0.0;
		double		stddev = variance > 0 ? sqrt(variance) : 0.0;

		state->ecShift[dim] = (float) -mean[dim];
		state->ecScale[dim] = stddev > FLT_EPSILON ? (float) (1.0 / stddev) : 1.0f;
	}

	pfree(buffer);
	pfree(m2);
	pfree(mean);

post_fit:

	/*
	 * cache mm_const = Σ ecShift² so build-time TQ+ scoring
	 * doesn't recompute it per neighbor-distance call.
	 */
	state->mmConst = TqGraphMmConstScalar(state->ecShift, state->dimensions);

	if (state->tqWeighted)
	{
		double		dPrimeSqMax = 0.0;
		double		weightScale;

		for (int dim = 0; dim < state->dimensions; dim++)
		{
			double		s = (double) state->ecScale[dim];

			if (fabs(s) > FLT_EPSILON)
			{
				double		w = 1.0 / (s * s);

				if (w > dPrimeSqMax)
					dPrimeSqMax = w;
			}
		}

		/*
		 * Quantize per-coord D'² to i16 so the AVX2 SIMD
		 * weighted-dot kernel can use _mm256_madd_epi16 directly.
		 * weight_scale = (INT16_MAX - 1) / max(D'²) keeps the largest
		 * weight at INT16_MAX-1; relative quantization error on the
		 * smallest non-zero weight is (min/max) · 1/32766 — well below
		 * the 4-bit code precision floor.
		 */
		weightScale = dPrimeSqMax > FLT_EPSILON
			? ((double) INT16_MAX - 1.0) / dPrimeSqMax
			: 1.0;
		state->weightScale = (float) weightScale;
		state->dPrimeSqI16 = MemoryContextAlloc(state->ctx,
												 sizeof(int16) * state->dimensions);
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			double		s = (double) state->ecScale[dim];
			double		w = (fabs(s) > FLT_EPSILON) ? 1.0 / (s * s) : 0.0;
			double		q = round(w * weightScale);

			if (q < 0.0)
				q = 0.0;
			if (q > (double) (INT16_MAX - 1))
				q = (double) (INT16_MAX - 1);
			state->dPrimeSqI16[dim] = (int16) q;
		}

		elog(DEBUG2, "turboquant TQ+ fit: dim=%d mm_const=%g max_dprime_sq=%g weight_scale=%g (fit=%s)",
			 state->dimensions, state->mmConst, dPrimeSqMax, weightScale,
			 state->tqQuantileFit ? "quantile" : "welford");
	}
}

static void
TqGraphEncodeBuildNodes(TqGraphBuildState *state)
{
	for (uint32 row = 0; row < state->nodeCount; row++)
	{
		TqGraphBuildNode *node = &state->nodes[row];

		if (state->tqWeighted)
		{
			float		xm = 0.0f;

			if (state->tqRenorm)
				node->scale = TqGraphEncodeVectorWithXmRenorm(state, node->vector,
															  node->code, &xm);
			else
				node->scale = TqGraphEncodeVectorWithXm(state, node->vector,
														 node->code, &xm);
			node->ecCorrection = xm;
		}
		else
		{
			node->scale = TqGraphEncodeVector(state, node->vector, node->code);
			node->ecCorrection = 0.0f;
		}

		node->correction = TqGraphCodeNorm(node->code, state->dimensions, state->tqBits);
	}
}

static bool
TqGraphHasNeighbor(TqGraphBuildState *state, uint32 src, uint32 dst, int level)
{
	TqGraphBuildNode *node = &state->nodes[src];

	if (!TqGraphBuildNodeHasLevel(state, src, level))
		return false;

	for (int i = 0; i < node->neighborCounts[level]; i++)
	{
		if (node->neighbors[level][i] == dst)
			return true;
	}

	return false;
}

static int
TqGraphSelectNeighbors(TqGraphBuildState *state, uint32 src,
					   TqGraphFrontierItem *candidates, int candidateCount,
					   int level, uint32 *selected)
{
	int			selectedCount = 0;
	int			maxNeighbors = TqGraphLevelM(state->m, level);

	qsort(candidates, candidateCount, sizeof(TqGraphFrontierItem), TqGraphFrontierCompare);

	for (int i = 0; i < candidateCount && selectedCount < maxNeighbors; i++)
	{
		uint32		candidate = candidates[i].nodeId;
		bool		good = true;

		if (candidate == src)
			continue;

		for (int j = 0; j < selectedCount; j++)
		{
			double		selectedDistance = TqGraphBuildDistance(state, candidate, selected[j]);

			if (selectedDistance < candidates[i].distance)
			{
				good = false;
				break;
			}
		}

		if (good)
			selected[selectedCount++] = candidate;
	}

	for (int i = 0; i < candidateCount && selectedCount < maxNeighbors; i++)
	{
		bool		seen = false;

		for (int j = 0; j < selectedCount; j++)
		{
			if (selected[j] == candidates[i].nodeId)
			{
				seen = true;
				break;
			}
		}

		if (!seen && candidates[i].nodeId != src)
			selected[selectedCount++] = candidates[i].nodeId;
	}

	return selectedCount;
}

static void
TqGraphPruneNeighbors(TqGraphBuildState *state, uint32 src, int level)
{
	TqGraphBuildNode *node = &state->nodes[src];
	int			count;
	TqGraphFrontierItem *candidates;
	uint32	   *selected;
	int			selectedCount;

	if (!TqGraphBuildNodeHasLevel(state, src, level))
		return;

	count = node->neighborCounts[level];
	if (count <= TqGraphLevelM(state->m, level))
		return;

	candidates = palloc(sizeof(TqGraphFrontierItem) * count);
	selected = palloc(sizeof(uint32) * TqGraphLevelM(state->m, level));

	for (int i = 0; i < count; i++)
	{
		candidates[i].nodeId = node->neighbors[level][i];
		candidates[i].distance = TqGraphBuildDistance(state, src, candidates[i].nodeId);
	}

	selectedCount = TqGraphSelectNeighbors(state, src, candidates, count, level, selected);
	memcpy(node->neighbors[level], selected, sizeof(uint32) * selectedCount);
	node->neighborCounts[level] = selectedCount;

	pfree(candidates);
	pfree(selected);
}

static void
TqGraphAddNeighbor(TqGraphBuildState *state, uint32 src, uint32 dst, int level)
{
	TqGraphBuildNode *node = &state->nodes[src];
	int			maxNeighbors = TqGraphLevelM(state->m, level);

	if (src == dst || !TqGraphBuildNodeHasLevel(state, src, level) ||
		!TqGraphBuildNodeHasLevel(state, dst, level) ||
		TqGraphHasNeighbor(state, src, dst, level))
		return;

	if (node->neighborCounts[level] < maxNeighbors)
	{
		node->neighbors[level][node->neighborCounts[level]++] = dst;
		return;
	}

	node->neighbors[level][node->neighborCounts[level]++] = dst;
	TqGraphPruneNeighbors(state, src, level);
}

static TqGraphFrontierItem
TqGraphBuildGreedySearch(TqGraphBuildState *state, uint32 queryNodeId,
						 uint32 entryNodeId, int level, bool *inserted)
{
	TqGraphFrontierItem current;
	bool		changed = true;

	current.nodeId = entryNodeId;
	current.distance = TqGraphBuildDistance(state, queryNodeId, entryNodeId);

	while (changed)
	{
		TqGraphBuildNode *node = &state->nodes[current.nodeId];

		changed = false;
		if (!TqGraphBuildNodeHasLevel(state, current.nodeId, level))
			break;

		for (int i = 0; i < node->neighborCounts[level]; i++)
		{
			uint32		neighbor = node->neighbors[level][i];
			double		distance;

			if (!inserted[neighbor] ||
				!TqGraphBuildNodeHasLevel(state, neighbor, level))
				continue;

			distance = TqGraphBuildDistance(state, queryNodeId, neighbor);
			if (distance < current.distance)
			{
				current.nodeId = neighbor;
				current.distance = distance;
				changed = true;
			}
		}
	}

	return current;
}

static int
TqGraphBuildSearchLayer(TqGraphBuildState *state, uint32 queryNodeId,
						TqGraphFrontierItem entry, int level, int ef,
						TqGraphFrontierItem *nearest, bool *inserted)
{
	uint32		visitGeneration;
	int			maxNeighbors = TqGraphLevelM(state->m, level);
	int			frontierCapacity = TqGraphInitialFrontierCapacity(state->nodeCount, ef, 1,
																 maxNeighbors);
	int			maxFrontierCapacity = (int) state->nodeCount;
	TqGraphFrontierItem *frontier = palloc(sizeof(TqGraphFrontierItem) * frontierCapacity);
	int			frontierCount = 0;
	int			nearestCount = 0;

	visitGeneration = ++state->buildVisitGeneration;
	if (visitGeneration == 0)
	{
		memset(state->buildVisitedGeneration, 0,
			   sizeof(uint32) * state->nodeCount);
		visitGeneration = ++state->buildVisitGeneration;
	}

	state->buildVisitedGeneration[entry.nodeId] = visitGeneration;
	(void) TqGraphOfferNearest(nearest, ef, &nearestCount, entry.nodeId, entry.distance);
	TqGraphFrontierHeapPushGrowing(&frontier, &frontierCount, &frontierCapacity,
								   maxFrontierCapacity, entry, true);

	while (frontierCount > 0)
	{
		TqGraphFrontierItem item = TqGraphFrontierHeapPop(frontier, &frontierCount, true);
		TqGraphBuildNode *node = &state->nodes[item.nodeId];

		if (nearestCount >= ef && TqGraphFrontierGreater(item, nearest[0]))
			break;

		if (!TqGraphBuildNodeHasLevel(state, item.nodeId, level))
			continue;

		for (int i = 0; i < node->neighborCounts[level]; i++)
		{
			uint32		neighbor = node->neighbors[level][i];
			double		distance;

			if (!inserted[neighbor] ||
				state->buildVisitedGeneration[neighbor] == visitGeneration ||
				!TqGraphBuildNodeHasLevel(state, neighbor, level))
				continue;

			state->buildVisitedGeneration[neighbor] = visitGeneration;
			distance = TqGraphBuildDistance(state, queryNodeId, neighbor);
			if (TqGraphOfferNearest(nearest, ef, &nearestCount, neighbor, distance))
			{
				TqGraphFrontierItem frontierItem;

				frontierItem.nodeId = neighbor;
				frontierItem.distance = distance;
				TqGraphFrontierHeapPushGrowing(&frontier, &frontierCount,
											   &frontierCapacity,
											   maxFrontierCapacity, frontierItem,
											   true);
			}
		}
	}

	pfree(frontier);
	return nearestCount;
}

static void
TqGraphBuildEdges(TqGraphBuildState *state)
{
	int			ef = Max(state->efConstruction, TqGraphLevelM(state->m, 0));
	uint32		entryNodeId;
	int			entryLevel;
	TqGraphFrontierItem *nearest;
	uint32	   *selected;
	TqGraphBuildOrderItem *order;
	bool	   *inserted;

	if (state->nodeCount == 0)
		return;

	nearest = palloc(sizeof(TqGraphFrontierItem) * ef);
	selected = palloc(sizeof(uint32) * TqGraphLevelM(state->m, 0));
	order = palloc(sizeof(TqGraphBuildOrderItem) * state->nodeCount);
	inserted = palloc0(sizeof(bool) * state->nodeCount);
	state->buildVisitedGeneration = palloc0(sizeof(uint32) * state->nodeCount);
	state->buildVisitGeneration = 0;
	state->buildQueryCtx = AllocSetContextCreate(state->ctx,
												 "TurboQuant graph build query context",
												 ALLOCSET_DEFAULT_SIZES);

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		order[i].nodeId = i;
		order[i].key = TqGraphMix64(i);
	}
	qsort(order, state->nodeCount, sizeof(TqGraphBuildOrderItem),
		  TqGraphBuildOrderCompare);

	entryNodeId = order[0].nodeId;
	entryLevel = state->nodes[entryNodeId].level;
	inserted[entryNodeId] = true;

	for (uint32 orderIdx = 1; orderIdx < state->nodeCount; orderIdx++)
	{
		uint32		i = order[orderIdx].nodeId;
		TqGraphFrontierItem levelEntry;
		int			nodeLevel = state->nodes[i].level;
		int			linkingLevel = Min(nodeLevel, entryLevel);

			CHECK_FOR_INTERRUPTS();
			TqGraphPrepareBuildQuery(state, i);

			levelEntry.nodeId = entryNodeId;
			levelEntry.distance = TqGraphBuildDistance(state, i, entryNodeId);

		for (int level = entryLevel; level > nodeLevel; level--)
		{
			if (TqGraphBuildNodeHasLevel(state, levelEntry.nodeId, level))
				levelEntry = TqGraphBuildGreedySearch(state, i,
													  levelEntry.nodeId,
													  level, inserted);
		}

		for (int level = linkingLevel; level >= 0; level--)
		{
			int			nearestCount;
			int			selectedCount;

			if (!TqGraphBuildNodeHasLevel(state, levelEntry.nodeId, level))
				continue;

			nearestCount = TqGraphBuildSearchLayer(state, i, levelEntry, level,
												   ef, nearest, inserted);
			selectedCount = TqGraphSelectNeighbors(state, i, nearest, nearestCount,
												  level, selected);

			memcpy(state->nodes[i].neighbors[level], selected,
				   sizeof(uint32) * selectedCount);
			state->nodes[i].neighborCounts[level] = selectedCount;
			for (int j = 0; j < selectedCount; j++)
				TqGraphAddNeighbor(state, selected[j], i, level);

			if (nearestCount > 0)
			{
				qsort(nearest, nearestCount, sizeof(TqGraphFrontierItem), TqGraphFrontierCompare);
				levelEntry = nearest[0];
			}
		}

		if (nodeLevel > entryLevel)
		{
			entryNodeId = i;
			entryLevel = nodeLevel;
		}
		inserted[i] = true;
	}

	state->entryNodeId = entryNodeId;
	state->maxLevel = entryLevel;
	pfree(nearest);
	pfree(selected);
	pfree(order);
	pfree(inserted);
	pfree(state->buildVisitedGeneration);
	state->buildVisitedGeneration = NULL;
	MemoryContextDelete(state->buildQueryCtx);
	state->buildQueryCtx = NULL;
	state->buildTqValid = false;
}

static void
TqGraphReorderBuildNodesForLocality(TqGraphBuildState *state)
{
	TqGraphBuildNode *reordered;
	uint32	   *oldToNew;
	uint32	   *newToOld;
	uint32	   *queue;
	uint32		head = 0;
	uint32		tail = 0;
	uint32		orderCount = 0;
	bool		identity = true;

	if (HnswGetGraphReorder(state->index) == TQ_GRAPH_REORDER_OFF)
		return;
	if (state->nodeCount < 2 || state->entryNodeId >= state->nodeCount)
		return;

	oldToNew = palloc(sizeof(uint32) * state->nodeCount);
	newToOld = palloc(sizeof(uint32) * state->nodeCount);
	queue = palloc(sizeof(uint32) * state->nodeCount);
	for (uint32 i = 0; i < state->nodeCount; i++)
		oldToNew[i] = UINT_MAX;

#define TQ_GRAPH_ENQUEUE_OLD_NODE(oldid) \
	do { \
		uint32 _oldid = (oldid); \
		if (_oldid < state->nodeCount && oldToNew[_oldid] == UINT_MAX) \
		{ \
			oldToNew[_oldid] = orderCount; \
			newToOld[orderCount++] = _oldid; \
			queue[tail++] = _oldid; \
		} \
	} while (0)

	TQ_GRAPH_ENQUEUE_OLD_NODE(state->entryNodeId);
	while (head < tail)
	{
		uint32		oldId = queue[head++];
		TqGraphBuildNode *node = &state->nodes[oldId];

		if (node->level < 0 || node->neighborCounts == NULL ||
			node->neighbors == NULL)
			continue;

		for (int i = 0; i < node->neighborCounts[0]; i++)
			TQ_GRAPH_ENQUEUE_OLD_NODE(node->neighbors[0][i]);
	}

	for (uint32 i = 0; i < state->nodeCount; i++)
		TQ_GRAPH_ENQUEUE_OLD_NODE(i);

#undef TQ_GRAPH_ENQUEUE_OLD_NODE

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		if (newToOld[i] != i)
		{
			identity = false;
			break;
		}
	}

	if (identity)
	{
		pfree(oldToNew);
		pfree(newToOld);
		pfree(queue);
		return;
	}

	reordered = MemoryContextAllocZero(state->ctx,
									   sizeof(TqGraphBuildNode) * state->nodeCount);
	for (uint32 newId = 0; newId < state->nodeCount; newId++)
		reordered[newId] = state->nodes[newToOld[newId]];

	for (uint32 newId = 0; newId < state->nodeCount; newId++)
	{
		TqGraphBuildNode *node = &reordered[newId];

		for (int level = 0; level <= node->level; level++)
		{
			for (int i = 0; i < node->neighborCounts[level]; i++)
			{
				uint32		oldNeighbor = node->neighbors[level][i];

				if (oldNeighbor < state->nodeCount &&
					oldToNew[oldNeighbor] != UINT_MAX)
					node->neighbors[level][i] = oldToNew[oldNeighbor];
			}
		}
	}

	state->entryNodeId = oldToNew[state->entryNodeId];
	state->nodes = reordered;

	pfree(oldToNew);
	pfree(newToOld);
	pfree(queue);
}




static void
TqGraphCreateMetaPage(Relation index, ForkNumber forkNum)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = HnswNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	HnswInitPageKind(buf, page, HNSW_PAGE_KIND_META);
	metap = HnswPageGetMeta(page);

	memset(metap, 0, sizeof(HnswMetaPageData));
	metap->magicNumber = HNSW_MAGIC_NUMBER;
	metap->version = HNSW_VERSION;
	metap->storageKind = HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE;
	metap->m = HnswGetM(index);
	metap->efConstruction = HnswGetEfConstruction(index);
	metap->graphEfSearch = HnswGetEfSearch(index);
	metap->graphOversampling = HnswGetGraphOversampling(index);
	metap->graphRescoreBand = HnswGetGraphRescoreBand(index);
	metap->tqBits = HnswGetTqBits(index);
	metap->graphMaxLevel = 0;
	metap->entryBlkno = InvalidBlockNumber;
	metap->entryOffno = InvalidOffsetNumber;
	metap->entryLevel = -1;
	metap->insertPage = InvalidBlockNumber;
	metap->tqEntryNodeId = UINT_MAX;
	metap->tqCodeStartBlkno = InvalidBlockNumber;
	metap->tqAdjStartBlkno = InvalidBlockNumber;
	metap->tqExactStartBlkno = InvalidBlockNumber;
	metap->tqCorrectionStartBlkno = InvalidBlockNumber;
	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(HnswMetaPageData)) - (char *) page;

	TqGraphFinishPage(buf);
}

void
TqGraphUpdateMetaPage(Relation index, TqGraphBuildState *state,
					  BlockNumber codeStart, BlockNumber adjStart,
					  BlockNumber exactStart, BlockNumber correctionStart)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;
	GenericXLogState *xlogState = NULL;

	buf = ReadBufferExtended(index, state->forkNum, HNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	if (!state->building && state->forkNum == MAIN_FORKNUM && RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}
	else
		page = BufferGetPage(buf);

	metap = HnswPageGetMeta(page);

	metap->dimensions = state->dimensions;
	metap->m = state->m;
	metap->efConstruction = state->efConstruction;
	metap->storageKind = HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE;
	metap->graphEfSearch = HnswGetEfSearch(index);
	metap->graphOversampling = HnswGetGraphOversampling(index);
	metap->graphRescoreBand = HnswGetGraphRescoreBand(index);
	metap->graphMaxLevel = state->maxLevel;
	metap->graphFlags = metap->graphFlags == 0 ? 1 : metap->graphFlags + 1;
	metap->entryBlkno = codeStart;
	metap->entryOffno = state->nodeCount > 0 ? FirstOffsetNumber : InvalidOffsetNumber;
	metap->entryLevel = state->nodeCount > 0 ? state->nodes[state->entryNodeId].level : -1;
	metap->tqNodeCount = state->nodeCount;
	metap->tqEntryNodeId = state->nodeCount > 0 ? state->entryNodeId : UINT_MAX;
	metap->tqCodeBytes = state->dimensions > 0 ? TqGraphCodeBytesForBits(state->dimensions, state->tqBits) : 0;
	metap->tqPayloadCount = state->payloadCount;
	metap->tqPayloadBytes = state->payloadBytes;
	metap->tqFlags = state->ecShift != NULL && state->ecScale != NULL ? TQ_GRAPH_TQ_PLUS : 0;
	if (state->tqWeighted)
		metap->tqFlags |= TQ_GRAPH_TQ_WEIGHTED;
	if (state->tqRenorm)
		metap->tqFlags |= TQ_GRAPH_TQ_RENORM;
	metap->tqBits = state->tqBits != 0 ? state->tqBits : TQ_DEFAULT_BITS;
	metap->tqCodeStartBlkno = codeStart;
	metap->tqAdjStartBlkno = adjStart;
	metap->tqExactStartBlkno = exactStart;
	metap->tqCorrectionStartBlkno = correctionStart;
	HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);

	HnswLogGraphWalRecord(index, state->forkNum, HNSW_METAPAGE_BLKNO, HNSW_GRAPH_OP_META_UPDATE);
}

static void
TqGraphBumpMetaGeneration(Relation index)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;
	GenericXLogState *xlogState = NULL;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	if (RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}
	else
		page = BufferGetPage(buf);

	metap = HnswPageGetMeta(page);
	metap->graphFlags = metap->graphFlags == 0 ? 1 : metap->graphFlags + 1;
	HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);

	UnlockReleaseBuffer(buf);
	HnswLogGraphWalRecord(index, MAIN_FORKNUM, HNSW_METAPAGE_BLKNO, HNSW_GRAPH_OP_META_UPDATE);
}

static BlockNumber
TqGraphWriteCodePages(TqGraphBuildState *state)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	Size		tupleSize = TqGraphCodeTupleSize(state->dimensions, state->payloadCount,
												 state->tqBits, state->tqWeighted);
	TqGraphCodeTuple tuple = palloc0(tupleSize);

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		TqGraphBuildNode *node = &state->nodes[i];

		if (!BufferIsValid(buf) || PageGetFreeSpace(page) < tupleSize)
		{
			TqGraphAppendPage(state->index, state->forkNum, &buf, &page, HNSW_PAGE_KIND_TQ_CODE);
			if (!BlockNumberIsValid(start))
				start = BufferGetBlockNumber(buf);
		}

		memset(tuple, 0, tupleSize);
		tuple->type = TQ_GRAPH_CODE_TUPLE_TYPE;
		tuple->level = node->level;
		tuple->flags = node->flags;
		tuple->nodeId = i;
		tuple->heaptid = node->heaptid;
		tuple->exactBlkno = node->exactBlkno;
		tuple->exactOffno = node->exactOffno;
		tuple->payloadMask = node->payloadMask;
		tuple->scale = node->scale;
		tuple->norm = node->norm;
		tuple->correction = node->correction;
		TqGraphTupleSetEcCorrection(tuple, state->tqWeighted, node->ecCorrection);
		if (state->payloadCount > 0 && node->payloads != NULL)
			memcpy(TqGraphTuplePayloads(tuple, state->tqWeighted), node->payloads, state->payloadBytes);
		memcpy(TqGraphTupleCode(tuple, state->payloadBytes, state->tqWeighted), node->code,
			   TqGraphCodeBytesForBits(state->dimensions, state->tqBits));

		if (PageAddItem(page, (Item) tuple, tupleSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
			elog(ERROR, "failed to add turboquant graph code item to \"%s\"", RelationGetRelationName(state->index));
		HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_ELEMENT_INSERT);
	}

	if (BufferIsValid(buf))
		TqGraphFinishPage(buf);

	pfree(tuple);
	return start;
}

static BlockNumber
TqGraphWriteAdjPages(TqGraphBuildState *state)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	Size		maxTupleSize = TqGraphAdjTupleSize(TqGraphLevelM(state->m, 0));
	TqGraphAdjTuple tuple = palloc0(maxTupleSize);

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		for (int level = 0; level < TqGraphLevelCapacity(state->m); level++)
		{
			TqGraphBuildNode *node = &state->nodes[i];
			int			count = level <= node->level ? node->neighborCounts[level] : 0;

			if (!BufferIsValid(buf) || PageGetFreeSpace(page) < maxTupleSize)
			{
				TqGraphAppendPage(state->index, state->forkNum, &buf, &page, HNSW_PAGE_KIND_TQ_ADJ);
				if (!BlockNumberIsValid(start))
					start = BufferGetBlockNumber(buf);
			}

			memset(tuple, 0, maxTupleSize);
			tuple->type = TQ_GRAPH_ADJ_TUPLE_TYPE;
			tuple->level = level;
			tuple->count = count;
			tuple->nodeId = i;
			for (int j = 0; j < count; j++)
				tuple->neighbors[j] = node->neighbors[level][j];

			if (PageAddItem(page, (Item) tuple, maxTupleSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
				elog(ERROR, "failed to add turboquant graph adjacency item to \"%s\"", RelationGetRelationName(state->index));
			HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_NEIGHBOR_INSERT);
		}
	}

	if (BufferIsValid(buf))
		TqGraphFinishPage(buf);

	pfree(tuple);
	return start;
}


static BlockNumber
TqGraphWriteCorrectionPages(TqGraphBuildState *state)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	Size		tupleSize;
	TqGraphCorrectionTuple tuple;

	if (state->nodeCount == 0 || state->dimensions <= 0 ||
		state->ecShift == NULL || state->ecScale == NULL)
		return InvalidBlockNumber;

	tupleSize = TqGraphCorrectionTupleSize(state->dimensions);
	tuple = palloc0(tupleSize);

	for (int field = 0; field < 2; field++)
	{
		const float *values = field == 0 ? state->ecShift : state->ecScale;

		if (!BufferIsValid(buf) || PageGetFreeSpace(page) < tupleSize)
		{
			TqGraphAppendPage(state->index, state->forkNum, &buf, &page,
							  HNSW_PAGE_KIND_TQ_CORRECTION);
			if (!BlockNumberIsValid(start))
				start = BufferGetBlockNumber(buf);
		}

		memset(tuple, 0, tupleSize);
		tuple->type = TQ_GRAPH_CORRECTION_TUPLE_TYPE;
		tuple->field = field;
		tuple->count = state->dimensions;
		tuple->startDim = 0;
		memcpy(tuple->values, values, sizeof(float) * state->dimensions);

		if (PageAddItem(page, (Item) tuple, tupleSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
			elog(ERROR, "failed to add turboquant graph correction item to \"%s\"", RelationGetRelationName(state->index));
		HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_ELEMENT_INSERT);
	}

	if (BufferIsValid(buf))
		TqGraphFinishPage(buf);

	pfree(tuple);
	return start;
}

static void
TqGraphWriteGraphDataPages(TqGraphBuildState *state, BlockNumber *codeStart,
						   BlockNumber *adjStart, BlockNumber *exactStart,
						   BlockNumber *correctionStart)
{
	if (state->nodeCount == 0)
	{
		*codeStart = InvalidBlockNumber;
		*adjStart = InvalidBlockNumber;
		*exactStart = InvalidBlockNumber;
		*correctionStart = InvalidBlockNumber;
		return;
	}

	*exactStart = TqGraphWriteExactPages(state);
	*correctionStart = TqGraphWriteCorrectionPages(state);
	*codeStart = TqGraphWriteCodePages(state);
	*adjStart = TqGraphWriteAdjPages(state);
}

static void
TqGraphWriteGraphPages(TqGraphBuildState *state)
{
	BlockNumber codeStart;
	BlockNumber adjStart;
	BlockNumber exactStart;
	BlockNumber correctionStart;

	TqGraphWriteGraphDataPages(state, &codeStart, &adjStart, &exactStart,
							   &correctionStart);
	TqGraphUpdateMetaPage(state->index, state, codeStart, adjStart, exactStart,
						  correctionStart);
}

static void
TqGraphWriteIndex(TqGraphBuildState *state)
{
	TqGraphCreateMetaPage(state->index, state->forkNum);
	TqGraphWriteGraphPages(state);
}

static int64
TqGraphElapsedUs(instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	return (int64) INSTR_TIME_GET_MICROSEC(duration);
}

static void
TqGraphDebugBuildPhaseStart(TqGraphBuildState *state, const char *phase)
{
	elog(DEBUG1, "turboquant native graph build phase start: relation=%s phase=%s nodes=%u dimensions=%d m=%d ef_construction=%d score_mode=%d",
		 RelationGetRelationName(state->index), phase, state->nodeCount,
		 state->dimensions, state->m, state->efConstruction, state->scoreMode);
}

static void
TqGraphDebugBuildPhaseDone(TqGraphBuildState *state, const char *phase,
						   instr_time phaseStart)
{
	elog(DEBUG1, "turboquant native graph build phase done: relation=%s phase=%s elapsed_ms=%.3f nodes=%u",
		 RelationGetRelationName(state->index), phase,
		 (double) TqGraphElapsedUs(phaseStart) / 1000.0,
		 state->nodeCount);
}

IndexBuildResult *
tqgraphbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	TqGraphBuildState state;
	IndexBuildResult *result;
	instr_time	totalStart;
	instr_time	phaseStart;
	int64		scanUs = 0;
	int64		correctionUs;
	int64		encodeUs;
	int64		edgesUs;
	int64		writeUs;
	int64		walUs = 0;

	memset(&state, 0, sizeof(state));
	INSTR_TIME_SET_CURRENT(totalStart);
	state.heap = heap;
	state.index = index;
	state.indexInfo = indexInfo;
	state.forkNum = MAIN_FORKNUM;
	state.building = true;
	state.typeInfo = HnswGetTypeInfo(index);
	state.m = HnswGetM(index);
	state.efConstruction = HnswGetEfConstruction(index);
	state.tqBits = HnswGetTqBits(index);
	state.tqWeighted = HnswGetTqWeightedOption(index);
	state.tqQuantileFit = HnswGetTqQuantileFitOption(index);
	state.tqRenorm = HnswGetTqRenormOption(index);
	state.buildExactDistances = hnsw_tq_build_exact_distances;
	if (state.tqRenorm && state.tqBits >= TQ_DEFAULT_BITS)
		ereport(NOTICE,
				(errmsg("tq_renorm has no measurable effect at tq_bits = %d",
						state.tqBits),
				 errdetail("At 4-bit and above the Lloyd-Max codebook is fine-grained enough that centroid_norm ≈ sqrt(d) and the renormalization factor sqrt(d) / centroid_norm ≈ 1.  Quality and latency are within noise; only the encoder pays ~1.4 %% extra cost per build."),
					 errhint("Use tq_renorm = on with tq_bits = 2 where the codebook undersamples Gaussian tails enough for the correction to restore lost magnitude.  At tq_bits ≥ 4, tq_quantile_fit already captures the bias correction renorm would target.")));
	else if (state.tqRenorm && state.tqBits == 1)
		ereport(NOTICE,
				(errmsg("tq_renorm is not recommended at tq_bits = 1"),
				 errdetail("At 1-bit the decoded vector carries only sign information (centroids ±0.7978846); centroid_norm no longer tracks decoded magnitude and the renormalization factor injects per-vector noise from EC asymmetry instead of correcting bias.  Measured on random-s-100-angular: recall@10 -6.87 pt vs baseline, nDCG@10 -7.39 pt."),
				 errhint("Use tq_renorm = on with tq_bits = 2 — the only bit width where renorm structurally improves quality.")));
	state.payloadCount = TqGraphIndexPayloadCount(index);
	state.payloadBytes = TqGraphPayloadBytes(state.payloadCount);
	HnswInitSupport(&state.support, index);
	state.scoreMode = TqGraphGetScoreMode(&state.support);
	state.ctx = AllocSetContextCreate(CurrentMemoryContext,
									  "TurboQuant native graph build context",
									  ALLOCSET_DEFAULT_SIZES);
	state.nodes = MemoryContextAllocZero(state.ctx, sizeof(TqGraphBuildNode) * 1024);
	state.nodeCapacity = 1024;

	if (heap != NULL)
	{
		TqGraphDebugBuildPhaseStart(&state, "scan");
		INSTR_TIME_SET_CURRENT(phaseStart);
		state.reltuples = table_index_build_scan(heap, index, indexInfo,
												 true, true, TqGraphBuildCallback, &state, NULL);
		scanUs = TqGraphElapsedUs(phaseStart);
		TqGraphDebugBuildPhaseDone(&state, "scan", phaseStart);
	}

	TqGraphDebugBuildPhaseStart(&state, "fit_correction");
	INSTR_TIME_SET_CURRENT(phaseStart);
	TqGraphFitCorrection(&state);
	correctionUs = TqGraphElapsedUs(phaseStart);
	TqGraphDebugBuildPhaseDone(&state, "fit_correction", phaseStart);

	TqGraphDebugBuildPhaseStart(&state, "encode");
	INSTR_TIME_SET_CURRENT(phaseStart);
	TqGraphEncodeBuildNodes(&state);
	encodeUs = TqGraphElapsedUs(phaseStart);
	TqGraphDebugBuildPhaseDone(&state, "encode", phaseStart);

	TqGraphDebugBuildPhaseStart(&state, "build_edges");
	INSTR_TIME_SET_CURRENT(phaseStart);
	TqGraphBuildEdges(&state);
	edgesUs = TqGraphElapsedUs(phaseStart);
	TqGraphDebugBuildPhaseDone(&state, "build_edges", phaseStart);

	TqGraphDebugBuildPhaseStart(&state, "reorder_nodes");
	INSTR_TIME_SET_CURRENT(phaseStart);
	TqGraphReorderBuildNodesForLocality(&state);
	TqGraphDebugBuildPhaseDone(&state, "reorder_nodes", phaseStart);

	TqGraphDebugBuildPhaseStart(&state, "write_pages");
	INSTR_TIME_SET_CURRENT(phaseStart);
	TqGraphWriteIndex(&state);
	writeUs = TqGraphElapsedUs(phaseStart);
	TqGraphDebugBuildPhaseDone(&state, "write_pages", phaseStart);

	if (RelationNeedsWAL(index))
	{
		TqGraphDebugBuildPhaseStart(&state, "wal_newpages");
		INSTR_TIME_SET_CURRENT(phaseStart);
		log_newpage_range(index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);
		walUs = TqGraphElapsedUs(phaseStart);
		TqGraphDebugBuildPhaseDone(&state, "wal_newpages", phaseStart);
	}

	elog(DEBUG1, "turboquant native graph build timings: relation=%s nodes=%u dimensions=%d scan_ms=%.3f fit_correction_ms=%.3f encode_ms=%.3f build_edges_ms=%.3f write_pages_ms=%.3f wal_ms=%.3f total_ms=%.3f",
		 RelationGetRelationName(index), state.nodeCount, state.dimensions,
		 (double) scanUs / 1000.0,
		 (double) correctionUs / 1000.0,
		 (double) encodeUs / 1000.0,
		 (double) edgesUs / 1000.0,
		 (double) writeUs / 1000.0,
		 (double) walUs / 1000.0,
		 (double) TqGraphElapsedUs(totalStart) / 1000.0);
	elog(DEBUG1, "turboquant native graph build distance paths: relation=%s calls=%llu query_split=%llu packed=%llu weighted=%llu code_code=%llu exact=%llu fallback=%llu",
		 RelationGetRelationName(index),
		 (unsigned long long) state.buildDistanceCalls,
		 (unsigned long long) state.buildDistanceQuerySplit,
		 (unsigned long long) state.buildDistancePacked,
		 (unsigned long long) state.buildDistanceWeighted,
		 (unsigned long long) state.buildDistanceCodeCode,
		 (unsigned long long) state.buildDistanceExact,
		 (unsigned long long) state.buildDistanceFallback);

	result = palloc0(sizeof(IndexBuildResult));
	result->heap_tuples = state.reltuples;
	result->index_tuples = state.nodeCount;

	MemoryContextDelete(state.ctx);

	return result;
}

void
tqgraphbuildempty(Relation index)
{
	TqGraphBuildState state;

	memset(&state, 0, sizeof(state));
	state.index = index;
	state.forkNum = INIT_FORKNUM;
	state.building = true;
	state.m = HnswGetM(index);
	state.efConstruction = HnswGetEfConstruction(index);
	state.tqBits = HnswGetTqBits(index);
	state.tqWeighted = HnswGetTqWeightedOption(index);
	state.tqQuantileFit = HnswGetTqQuantileFitOption(index);
	state.tqRenorm = HnswGetTqRenormOption(index);
	state.buildExactDistances = hnsw_tq_build_exact_distances;
	if (state.tqRenorm && state.tqBits >= TQ_DEFAULT_BITS)
		ereport(NOTICE,
				(errmsg("tq_renorm has no measurable effect at tq_bits = %d",
						state.tqBits),
				 errdetail("At 4-bit and above the Lloyd-Max codebook is fine-grained enough that centroid_norm ≈ sqrt(d) and the renormalization factor sqrt(d) / centroid_norm ≈ 1.  Quality and latency are within noise; only the encoder pays ~1.4 %% extra cost per build."),
					 errhint("Use tq_renorm = on with tq_bits = 2 where the codebook undersamples Gaussian tails enough for the correction to restore lost magnitude.  At tq_bits ≥ 4, tq_quantile_fit already captures the bias correction renorm would target.")));
	else if (state.tqRenorm && state.tqBits == 1)
		ereport(NOTICE,
				(errmsg("tq_renorm is not recommended at tq_bits = 1"),
				 errdetail("At 1-bit the decoded vector carries only sign information (centroids ±0.7978846); centroid_norm no longer tracks decoded magnitude and the renormalization factor injects per-vector noise from EC asymmetry instead of correcting bias.  Measured on random-s-100-angular: recall@10 -6.87 pt vs baseline, nDCG@10 -7.39 pt."),
				 errhint("Use tq_renorm = on with tq_bits = 2 — the only bit width where renorm structurally improves quality.")));
	state.payloadCount = TqGraphIndexPayloadCount(index);
	state.payloadBytes = TqGraphPayloadBytes(state.payloadCount);
	TqGraphWriteIndex(&state);
	log_newpage_range(index, INIT_FORKNUM, 0, RelationGetNumberOfBlocksInFork(index, INIT_FORKNUM), true);
}

static void
TqGraphAddElapsedUs(int64 *target, instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	*target += (int64) INSTR_TIME_GET_MICROSEC(duration);
}

static void
TqGraphResetScan(HnswScanOpaque so)
{
	so->first = true;
	so->returnedRows = 0;
	so->graphVisitedNodes = 0;
	so->graphScoredCodes = 0;
	so->graphCandidateCount = 0;
	so->graphRescoreCount = 0;
	so->graphRescorePages = 0;
	so->graphCodePagesRead = 0;
	so->graphAdjPagesRead = 0;
	so->graphEntryPointCount = 0;
	so->graphPrepareUs = 0;
	so->graphTraverseUs = 0;
	so->graphFillUs = 0;
	so->graphRescoreUs = 0;
	so->graphSortUs = 0;
	so->graphTotalUs = 0;
	so->tqGraphResults = NULL;
	so->tqGraphResultCount = 0;
	so->tqGraphResultIndex = 0;
	so->hasTupleTargetRows = false;
	so->hasEstimatedFilterSelectivity = false;
	so->hasInitialEffectiveEfSearch = false;
	so->returnedRows = 0;
	so->tupleTargetRows = -1;
	so->estimatedFilterSelectivity = -1.0;
	memset(&so->tq, 0, sizeof(HnswTqQuery));
	MemoryContextReset(so->tmpCtx);
}

IndexScanDesc
tqgraphbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	so = palloc0(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);
	HnswInitSupport(&so->support, index);
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "TurboQuant native graph scan context",
									   0, 8 * 1024, 256 * 1024);
	so->efSearch = HnswGetEfSearch(index);
	so->graphOversampling = HnswGetGraphOversampling(index);
	so->graphRescoreBand = HnswGetGraphRescoreBand(index);
	so->graphStorageKind = HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE;
	so->turboquantGraphScan = true;
	TqGraphResetScan(so);
	scan->opaque = so;

	return scan;
}

void
tqgraphrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));

	so->efSearch = HnswGetEfSearch(scan->indexRelation);
	so->graphOversampling = HnswGetGraphOversampling(scan->indexRelation);
	so->graphRescoreBand = HnswGetGraphRescoreBand(scan->indexRelation);
	so->graphStorageKind = HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE;
	so->turboquantGraphScan = true;
	TqGraphResetScan(so);
}

static Datum
TqGraphGetScanValue(IndexScanDesc scan, HnswScanOpaque so)
{
	Datum		value;

	if (scan->orderByData == NULL || scan->numberOfOrderBys < 1)
		elog(ERROR, "cannot scan turboquant graph index without order");

	value = scan->orderByData[0].sk_argument;
	if (DatumGetPointer(value) == NULL)
		return value;

	value = PointerGetDatum(PG_DETOAST_DATUM(value));

	if (so->support.normprocinfo != NULL)
	{
		if (!HnswCheckNorm(&so->support, value))
			value = PointerGetDatum(NULL);
		else
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

static int
TqGraphResultCompare(const void *a, const void *b)
{
	const TqGraphResult *ra = (const TqGraphResult *) a;
	const TqGraphResult *rb = (const TqGraphResult *) b;

	if (ra->distance < rb->distance)
		return -1;
	if (ra->distance > rb->distance)
		return 1;
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}

static int
TqGraphRescoreRefCompare(const void *a, const void *b)
{
	const TqGraphRescoreRef *ra = (const TqGraphRescoreRef *) a;
	const TqGraphRescoreRef *rb = (const TqGraphRescoreRef *) b;

	if (ra->blkno < rb->blkno)
		return -1;
	if (ra->blkno > rb->blkno)
		return 1;
	if (ra->offno < rb->offno)
		return -1;
	if (ra->offno > rb->offno)
		return 1;
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}

static bool
TqGraphResultLess(TqGraphResult a, TqGraphResult b)
{
	if (a.distance != b.distance)
		return a.distance < b.distance;
	return a.nodeId < b.nodeId;
}

static bool
TqGraphResultGreater(TqGraphResult a, TqGraphResult b)
{
	if (a.distance != b.distance)
		return a.distance > b.distance;
	return a.nodeId > b.nodeId;
}

static void
TqGraphResultSwap(TqGraphResult *a, TqGraphResult *b)
{
	TqGraphResult tmp = *a;

	*a = *b;
	*b = tmp;
}

static void
TqGraphResultHeapSiftUp(TqGraphResult *heap, int idx)
{
	while (idx > 0)
	{
		int			parent = (idx - 1) / 2;

		if (!TqGraphResultGreater(heap[idx], heap[parent]))
			break;

		TqGraphResultSwap(&heap[idx], &heap[parent]);
		idx = parent;
	}
}

static void
TqGraphResultHeapSiftDown(TqGraphResult *heap, int count, int idx)
{
	for (;;)
	{
		int			left = idx * 2 + 1;
		int			right = left + 1;
		int			best = idx;

		if (left < count && TqGraphResultGreater(heap[left], heap[best]))
			best = left;
		if (right < count && TqGraphResultGreater(heap[right], heap[best]))
			best = right;
		if (best == idx)
			break;

		TqGraphResultSwap(&heap[idx], &heap[best]);
		idx = best;
	}
}

static void
TqGraphResultHeapPush(TqGraphResult *heap, int *count, TqGraphResult item)
{
	heap[*count] = item;
	(*count)++;
	TqGraphResultHeapSiftUp(heap, *count - 1);
}

static void
TqGraphResultHeapReplaceRoot(TqGraphResult *heap, int count,
							 TqGraphResult item)
{
	heap[0] = item;
	TqGraphResultHeapSiftDown(heap, count, 0);
}

static void
TqGraphOfferCandidate(HnswScanOpaque so, TqGraphResult *results, int target,
					  int *count, uint32 nodeId, ItemPointer heaptid,
					  double distance, bool exactScored)
{
	TqGraphResult item;

	if (target <= 0)
		return;

	item.nodeId = nodeId;
	item.heaptid = *heaptid;
	item.distance = distance;
	item.exactScored = exactScored;

	if (*count < target)
	{
		TqGraphResultHeapPush(results, count, item);
		if (exactScored)
			so->graphRescoreCount++;
		return;
	}

	if (TqGraphResultLess(item, results[0]))
	{
		if (results[0].exactScored)
			so->graphRescoreCount--;
		TqGraphResultHeapReplaceRoot(results, *count, item);
		if (exactScored)
			so->graphRescoreCount++;
	}
}

bool
TqGraphReadMeta(Relation index, HnswMetaPageData *meta)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (metap->magicNumber != HNSW_MAGIC_NUMBER ||
		metap->storageKind != HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	memcpy(meta, metap, sizeof(HnswMetaPageData));
	if (meta->tqBits != 1 && meta->tqBits != 2 && meta->tqBits != TQ_DEFAULT_BITS)
		meta->tqBits = TQ_DEFAULT_BITS;
	UnlockReleaseBuffer(buf);
	return true;
}




static int
TqGraphScanAdjSlot(HnswMetaPageData *meta, uint32 nodeId, int level)
{
	return TqGraphAdjSlot(meta, nodeId, level);
}

static bool
TqGraphEntryAlreadySelected(TqGraphFrontierItem *entries, int entryCount,
							uint32 nodeId)
{
	for (int i = 0; i < entryCount; i++)
	{
		if (entries[i].nodeId == nodeId)
			return true;
	}

	return false;
}

static void
TqGraphOfferLevelEntry(TqGraphFrontierItem *entries, int *entryCount,
					   int *entryLevels, uint32 nodeId, int level)
{
	int			worst = 0;

	if (TqGraphEntryAlreadySelected(entries, *entryCount, nodeId))
		return;

	if (*entryCount < TQ_GRAPH_MAX_ENTRY_POINTS)
	{
		entries[*entryCount].nodeId = nodeId;
		entries[*entryCount].distance = DBL_MAX;
		entryLevels[*entryCount] = level;
		(*entryCount)++;
		return;
	}

	for (int i = 1; i < *entryCount; i++)
	{
		if (entryLevels[i] < entryLevels[worst] ||
			(entryLevels[i] == entryLevels[worst] &&
			 entries[i].nodeId > entries[worst].nodeId))
			worst = i;
	}

	if (level > entryLevels[worst] ||
		(level == entryLevels[worst] && nodeId < entries[worst].nodeId))
	{
		entries[worst].nodeId = nodeId;
		entries[worst].distance = DBL_MAX;
		entryLevels[worst] = level;
	}
}

static void
TqGraphOfferDistanceEntry(TqGraphFrontierItem *entries, int *entryCount,
						  TqGraphFrontierItem entry)
{
	int			worst = 0;

	for (int i = 0; i < *entryCount; i++)
	{
		if (entries[i].nodeId == entry.nodeId)
		{
			entries[i].distance = Min(entries[i].distance, entry.distance);
			return;
		}
	}

	if (*entryCount < TQ_GRAPH_MAX_ENTRY_POINTS)
	{
		entries[*entryCount] = entry;
		(*entryCount)++;
		return;
	}

	for (int i = 1; i < *entryCount; i++)
	{
		if (entries[i].distance > entries[worst].distance)
			worst = i;
	}

	if (entry.distance < entries[worst].distance)
		entries[worst] = entry;
}

static TqGraphFrontierItem
TqGraphScanGreedySearch(Relation index, HnswScanOpaque so, Datum query,
						HnswMetaPageData *meta,
						TqGraphScanStorage *storage, TqGraphFrontierItem entry,
						int level)
{
	TqGraphFrontierItem current = entry;
	bool		changed = true;
	int			maxNeighbors = TqGraphLevelM(meta->m, level);
	uint32		batchNodeIds[TQ_GRAPH_MAX_NEIGHBORS];
	double		batchDistances[TQ_GRAPH_MAX_NEIGHBORS];
	bool		lookaheadPrefetch;

	if (maxNeighbors > TQ_GRAPH_MAX_NEIGHBORS)
		elog(ERROR, "turboquant graph neighbor batch exceeds fixed capacity");

	/* Size-gated look-ahead prefetch (see TqGraphSearchBaseLayer). */
	if (hnsw_tq_graph_lookahead_prefetch == TQ_GRAPH_LOOKAHEAD_OFF)
		lookaheadPrefetch = false;
	else if (hnsw_tq_graph_lookahead_prefetch == TQ_GRAPH_LOOKAHEAD_ON)
		lookaheadPrefetch = true;
	else
	{
		Size		workingSetBytes = (Size) meta->tqNodeCount *
			(sizeof(TqGraphScanNode) + sizeof(uint32));

		lookaheadPrefetch = workingSetBytes >
			(Size) hnsw_tq_graph_lookahead_threshold_kb * 1024;
	}

	while (changed)
	{
		int			slot;
		int			batchCount = 0;

		changed = false;
		if (!TqGraphLoadAdjPage(index, so, meta, storage, current.nodeId, level))
			break;

		slot = TqGraphScanAdjSlot(meta, current.nodeId, level);
		for (int i = 0; i < storage->neighborCounts[slot]; i++)
		{
			uint32		neighbor = storage->neighbors[slot][i];

			if (lookaheadPrefetch && i + 1 < storage->neighborCounts[slot])
			{
				uint32		la = storage->neighbors[slot][i + 1];

				if (la < meta->tqNodeCount)
					TQ_GRAPH_PREFETCH_READ(&storage->nodes[la]);
			}

			if (neighbor >= meta->tqNodeCount ||
				!TqGraphLoadCodePage(index, so, meta, storage, neighbor) ||
				storage->nodes[neighbor].level < level)
				continue;

			batchNodeIds[batchCount++] = neighbor;
		}

		if (DatumGetPointer(query) != NULL)
		{
			for (int i = 0; i < batchCount; i++)
				batchDistances[i] =
					TqGraphEntryDistance(so, query,
										 &storage->nodes[batchNodeIds[i]]);
		}
		else
			TqGraphScoreNodeBatch(so, storage, batchNodeIds, batchCount,
								  batchDistances, query);

		for (int i = 0; i < batchCount; i++)
		{
			if (batchDistances[i] < current.distance)
			{
				current.nodeId = batchNodeIds[i];
				current.distance = batchDistances[i];
				changed = true;
			}
		}
	}

	return current;
}

static int
TqGraphSearchBaseLayer(Relation index, HnswScanOpaque so, Datum query,
					   HnswMetaPageData *meta,
					   TqGraphScanStorage *storage,
					   TqGraphFrontierItem *entries, int entryCount,
					   TqGraphResult *results, int resultTarget, int searchEf,
					   int payloadSlot, int32 payloadValue)
{
	bool	   *visited = NULL;
	uint32		visitGeneration = 0;
	bool		useVisitGeneration = storage->visitedGeneration != NULL &&
		storage->visitGeneration != NULL;
	int			frontierCount = 0;
	int			nearestCount = 0;
	int			resultCount = 0;
	int			maxNeighbors = TqGraphLevelM(meta->m, 0);
	int			frontierCapacity = TqGraphInitialFrontierCapacity(meta->tqNodeCount,
																 searchEf,
																 entryCount,
																 maxNeighbors);
	int			maxFrontierCapacity = (int) meta->tqNodeCount;
	TqGraphFrontierItem stackFrontier[TQ_GRAPH_STACK_FRONTIER_CAPACITY];
	TqGraphFrontierItem stackNearest[TQ_GRAPH_STACK_FRONTIER_CAPACITY];
	TqGraphFrontierItem *frontier =
		hnsw_tq_graph_stack_scratch &&
		maxFrontierCapacity <= TQ_GRAPH_STACK_FRONTIER_CAPACITY ?
		stackFrontier : palloc(sizeof(TqGraphFrontierItem) * frontierCapacity);
	TqGraphFrontierItem *nearest =
		hnsw_tq_graph_stack_scratch &&
		searchEf <= TQ_GRAPH_STACK_FRONTIER_CAPACITY ?
		stackNearest : palloc(sizeof(TqGraphFrontierItem) * searchEf);
	bool		frontierAllocated = frontier != stackFrontier;
	bool		nearestAllocated = nearest != stackNearest;
	uint32		batchNodeIds[TQ_GRAPH_MAX_NEIGHBORS];
	double		batchDistances[TQ_GRAPH_MAX_NEIGHBORS];
	bool		lookaheadPrefetch;

	if (maxNeighbors > TQ_GRAPH_MAX_NEIGHBORS)
		elog(ERROR, "turboquant graph neighbor batch exceeds fixed capacity");

	/*
	 * Size-gated adjacency-list look-ahead prefetch.  The
	 * original FAISS-style prefetch was reverted on FIQA-57k (commit
	 * 67f38bd) because the metadata working set fits in cache and
	 * explicit __builtin_prefetch was paid-for-nothing uops,
	 * regressing p50.  Auto mode here gates on a corpus-size
	 * threshold: when (storage->nodes + visitedGeneration) bytes
	 * exceeds hnsw.tq_graph_lookahead_threshold_kb the hint kicks in.
	 * On FIQA-scale (57k × ~64 B ≈ 3.6 MB << 24 MB default) auto
	 * stays off (no regression).  On 1M+ corpora (>> 64 MB) auto
	 * turns on.  off / on bypass the gate.
	 */
	if (hnsw_tq_graph_lookahead_prefetch == TQ_GRAPH_LOOKAHEAD_OFF)
		lookaheadPrefetch = false;
	else if (hnsw_tq_graph_lookahead_prefetch == TQ_GRAPH_LOOKAHEAD_ON)
		lookaheadPrefetch = true;
	else
	{
		Size		workingSetBytes = (Size) meta->tqNodeCount *
			(sizeof(TqGraphScanNode) + sizeof(uint32));

		lookaheadPrefetch = workingSetBytes >
			(Size) hnsw_tq_graph_lookahead_threshold_kb * 1024;
	}

	if (useVisitGeneration)
	{
		visitGeneration = ++(*storage->visitGeneration);
		if (visitGeneration == 0)
		{
			memset(storage->visitedGeneration, 0, sizeof(uint32) * meta->tqNodeCount);
			visitGeneration = ++(*storage->visitGeneration);
		}
	}
	else
		visited = palloc0(sizeof(bool) * meta->tqNodeCount);

	for (int i = 0; i < entryCount; i++)
	{
		TqGraphFrontierItem entry = entries[i];

		if (entry.nodeId >= meta->tqNodeCount ||
			(useVisitGeneration ?
			 storage->visitedGeneration[entry.nodeId] == visitGeneration :
			 visited[entry.nodeId]))
			continue;

		if (useVisitGeneration)
			storage->visitedGeneration[entry.nodeId] = visitGeneration;
		else
			visited[entry.nodeId] = true;

		TqGraphFrontierHeapPushGrowing(&frontier, &frontierCount, &frontierCapacity,
									   maxFrontierCapacity, entry, true);
		(void) TqGraphOfferNearest(nearest, searchEf, &nearestCount, entry.nodeId, entry.distance);
	}

	while (frontierCount > 0)
	{
		TqGraphFrontierItem item = TqGraphFrontierHeapPop(frontier, &frontierCount, true);
		uint32		nodeId = item.nodeId;
		int			slot;

		if (!TqGraphLoadCodePage(index, so, meta, storage, nodeId))
			continue;

		if (nearestCount >= searchEf && TqGraphFrontierGreater(item, nearest[0]))
			break;

		so->graphVisitedNodes++;

		if (!TqGraphLoadAdjPage(index, so, meta, storage, nodeId, 0))
			continue;

		slot = TqGraphScanAdjSlot(meta, nodeId, 0);
		{
			int			batchCount = 0;

			for (int i = 0; i < storage->neighborCounts[slot]; i++)
			{
				uint32		neighbor = storage->neighbors[slot][i];

				/*
				 * Look-ahead prefetch — issue a HW prefetch
				 * for the *next* iteration's storage->nodes[] and
				 * visitedGeneration[] entries while the current
				 * iteration's load + visit-test stalls.  Gated on
				 * lookaheadPrefetch (size-aware) so it stays off on
				 * small corpora where the HW prefetcher already
				 * covers the access.
				 */
				if (lookaheadPrefetch && i + 1 < storage->neighborCounts[slot])
				{
					uint32		la = storage->neighbors[slot][i + 1];

					if (la < meta->tqNodeCount)
					{
						TQ_GRAPH_PREFETCH_READ(&storage->nodes[la]);
						if (useVisitGeneration)
							TQ_GRAPH_PREFETCH_READ(&storage->visitedGeneration[la]);
					}
				}

				if (neighbor >= meta->tqNodeCount ||
					(useVisitGeneration ?
					 storage->visitedGeneration[neighbor] == visitGeneration :
					 visited[neighbor]))
					continue;

				if (!TqGraphLoadCodePage(index, so, meta, storage, neighbor))
					continue;

				if (useVisitGeneration)
					storage->visitedGeneration[neighbor] = visitGeneration;
				else
					visited[neighbor] = true;
				batchNodeIds[batchCount++] = neighbor;
			}

			TqGraphScoreNodeBatch(so, storage, batchNodeIds, batchCount,
								  batchDistances, query);
			for (int i = 0; i < batchCount; i++)
			{
				uint32		neighbor = batchNodeIds[i];
				double		neighborDistance = batchDistances[i];
				bool		accepted;

				accepted = TqGraphOfferNearest(nearest, searchEf, &nearestCount,
											   neighbor, neighborDistance);
				if (accepted)
				{
					TqGraphFrontierItem frontierItem;

					frontierItem.nodeId = neighbor;
					frontierItem.distance = neighborDistance;
					TqGraphFrontierHeapPushGrowing(&frontier, &frontierCount,
												   &frontierCapacity,
												   maxFrontierCapacity,
												   frontierItem, true);
				}
			}
		}
	}

	for (int i = 0; i < nearestCount; i++)
	{
		uint32		nodeId = nearest[i].nodeId;
		TqGraphScanNode *node;
		bool		exactScored;
		double		resultDistance;

		if (nodeId >= meta->tqNodeCount ||
			!TqGraphLoadCodePage(index, so, meta, storage, nodeId))
			continue;

		node = &storage->nodes[nodeId];
		if (node->flags & TQ_GRAPH_NODE_DEAD ||
			!TqGraphNodeMatchesPayload(node, payloadSlot, payloadValue))
			continue;

		resultDistance = TqGraphResultDistance(so, query, node,
											  nearest[i].distance,
											  &exactScored);
		TqGraphOfferCandidate(so, results, resultTarget, &resultCount,
							  nodeId, &node->heaptid, resultDistance,
							  exactScored);
	}

	if (visited != NULL)
		pfree(visited);
	if (frontierAllocated)
		pfree(frontier);
	if (nearestAllocated)
		pfree(nearest);
	return resultCount;
}

int
TqGraphTraverse(Relation index, HnswScanOpaque so, HnswMetaPageData *meta,
				TqGraphScanStorage *storage, TqGraphResult *results,
				int resultTarget, int searchEf, Datum query,
				int payloadSlot, int32 payloadValue)
{
	uint32		entryNodeId = meta->tqEntryNodeId < meta->tqNodeCount ? meta->tqEntryNodeId : 0;
	TqGraphFrontierItem entry;
	TqGraphFrontierItem entries[TQ_GRAPH_MAX_ENTRY_POINTS];
	int			entryLevels[TQ_GRAPH_MAX_ENTRY_POINTS];
	TqGraphFrontierItem sampled[TQ_GRAPH_ENTRY_SAMPLE_COUNT];
	uint32		sampledNodeIds[TQ_GRAPH_ENTRY_SAMPLE_COUNT];
	double		sampledDistances[TQ_GRAPH_ENTRY_SAMPLE_COUNT];
	int			entryCount = 0;
	int			sampledCount = 0;

	if (meta->tqNodeCount == 0 ||
		!TqGraphLoadCodePage(index, so, meta, storage, entryNodeId))
		return 0;

	entry.nodeId = entryNodeId;
	entry.distance = TqGraphEntryDistance(so, query, &storage->nodes[entryNodeId]);

	for (int level = meta->graphMaxLevel; level > 0; level--)
	{
		if (storage->nodes[entry.nodeId].level >= level)
			entry = TqGraphScanGreedySearch(index, so, query, meta, storage, entry,
											level);
	}

	entries[entryCount] = entry;
	entryLevels[entryCount] = storage->nodes[entry.nodeId].level;
	entryCount++;

	/*
	 * Keep alternative high-level entry points instead of relying on one
	 * global entry. Build that shape from the compact native graph: score the
	 * best level-bearing nodes, then let deterministic sampled candidates
	 * replace weak high-level entries when the graph topology is not yet
	 * strong enough for pure hierarchical routing.
	 */
	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeId];

		if (nodeId == entry.nodeId || node->flags & TQ_GRAPH_NODE_DEAD ||
			node->level <= 0)
			continue;

		TqGraphOfferLevelEntry(entries, &entryCount, entryLevels, nodeId,
							   node->level);
	}

	if (entryCount > 1)
	{
		uint32		entryNodeIds[TQ_GRAPH_MAX_ENTRY_POINTS];
		double		entryDistances[TQ_GRAPH_MAX_ENTRY_POINTS];
		int			scoreCount = 0;

		for (int i = 1; i < entryCount; i++)
		{
			if (!TqGraphLoadCodePage(index, so, meta, storage, entries[i].nodeId))
				continue;

			entryNodeIds[scoreCount++] = entries[i].nodeId;
		}

		TqGraphScoreNodeBatch(so, storage, entryNodeIds, scoreCount,
							  entryDistances, query);
		for (int i = 1; i < entryCount; i++)
		{
			for (int j = 0; j < scoreCount; j++)
			{
				if (entries[i].nodeId == entryNodeIds[j])
				{
					entries[i].distance = entryDistances[j];
					break;
				}
			}
		}
	}

	if (meta->tqNodeCount > 1)
	{
		int			sampleTarget = searchEf;
		int			sampleCount;

		if (so->tq.bits == TQ_DEFAULT_BITS &&
			(TqScoreMode) so->tq.scoreMode == TQ_SCORE_L2 &&
			so->tq.dimensions >= 1024)
			sampleTarget = Max(TQ_GRAPH_MAX_ENTRY_POINTS,
							   searchEf / TQ_GRAPH_HIGHDIM_ENTRY_SAMPLE_DIVISOR);

		sampleCount = Min((int) meta->tqNodeCount,
						  Min(TQ_GRAPH_ENTRY_SAMPLE_COUNT,
							  Max(sampleTarget,
								  TQ_GRAPH_MAX_ENTRY_POINTS)));

		for (int i = 0; i < sampleCount; i++)
		{
			uint32		nodeId = sampleCount == 1 ? 0 :
				(uint32) (((uint64) i * (meta->tqNodeCount - 1)) / (sampleCount - 1));
			bool		seen = TqGraphEntryAlreadySelected(entries, entryCount,
															nodeId);

			for (int j = 0; j < sampledCount; j++)
			{
				if (sampled[j].nodeId == nodeId)
				{
					seen = true;
					break;
				}
			}

			if (seen || !TqGraphLoadCodePage(index, so, meta, storage, nodeId))
				continue;

			sampled[sampledCount].nodeId = nodeId;
			sampledNodeIds[sampledCount] = nodeId;
			sampledCount++;
		}

		TqGraphScoreNodeBatch(so, storage, sampledNodeIds, sampledCount,
							  sampledDistances, query);
		for (int i = 0; i < sampledCount; i++)
		{
			double		exactDistance;

			if (TqGraphExactHighdimEntryDistance(so, query,
												 &storage->nodes[sampledNodeIds[i]],
												 &exactDistance))
				sampledDistances[i] = exactDistance;
			sampled[i].distance = sampledDistances[i];
		}

		qsort(sampled, sampledCount, sizeof(TqGraphFrontierItem),
			  TqGraphFrontierCompare);
		for (int i = 0; i < sampledCount; i++)
			TqGraphOfferDistanceEntry(entries, &entryCount, sampled[i]);
	}

	so->graphEntryPointCount = entryCount;

	return TqGraphSearchBaseLayer(index, so, query, meta, storage, entries, entryCount,
								  results, resultTarget, searchEf,
								  payloadSlot, payloadValue);
}

static int
TqGraphFillCandidateBand(Relation index, HnswScanOpaque so,
						 HnswMetaPageData *meta,
						 TqGraphScanStorage *storage,
						 TqGraphResult *results, int resultTarget, int count,
						 int payloadSlot, int32 payloadValue, Datum query)
{
	bool	   *selected;
	uint32		batchNodeIds[TQ_GRAPH_MAX_NEIGHBORS];
	double		batchDistances[TQ_GRAPH_MAX_NEIGHBORS];
	int			batchCount = 0;
	uint32		payloadFirst = 0;
	uint32		payloadCount = 0;
	bool		usePayloadRefs;

	if (count >= resultTarget || resultTarget <= 0)
		return count;

	selected = palloc0(sizeof(bool) * meta->tqNodeCount);
	for (int i = 0; i < count; i++)
	{
		if (results[i].nodeId < meta->tqNodeCount)
			selected[results[i].nodeId] = true;
	}

	usePayloadRefs = TqGraphPayloadRefRange(storage, payloadSlot, payloadValue,
											&payloadFirst, &payloadCount);
	if (payloadSlot >= 0 && storage->payloadRefs != NULL && !usePayloadRefs)
	{
		pfree(selected);
		return count;
	}

	for (uint32 i = 0; i < (usePayloadRefs ? payloadCount : meta->tqNodeCount); i++)
	{
		uint32		nodeId = usePayloadRefs ?
			storage->payloadRefs[payloadFirst + i].nodeId : i;
		TqGraphScanNode *node;

		if (selected[nodeId])
			continue;
		if (!TqGraphLoadCodePage(index, so, meta, storage, nodeId))
			continue;

		node = &storage->nodes[nodeId];
		if (node->flags & TQ_GRAPH_NODE_DEAD)
			continue;
		if (!TqGraphNodeMatchesPayload(node, payloadSlot, payloadValue))
			continue;

		batchNodeIds[batchCount++] = nodeId;
		if (batchCount == TQ_GRAPH_MAX_NEIGHBORS)
		{
			TqGraphScoreNodeBatch(so, storage, batchNodeIds, batchCount,
								  batchDistances, query);
			for (int i = 0; i < batchCount; i++)
			{
				node = &storage->nodes[batchNodeIds[i]];
				TqGraphOfferCandidate(so, results, resultTarget, &count,
									  batchNodeIds[i], &node->heaptid,
									  batchDistances[i], false);
			}
			batchCount = 0;
		}
	}

	if (batchCount > 0)
	{
		TqGraphScoreNodeBatch(so, storage, batchNodeIds, batchCount,
							  batchDistances, query);
		for (int i = 0; i < batchCount; i++)
		{
			TqGraphScanNode *node = &storage->nodes[batchNodeIds[i]];

			TqGraphOfferCandidate(so, results, resultTarget, &count,
								  batchNodeIds[i], &node->heaptid,
								  batchDistances[i], false);
		}
	}

	pfree(selected);
	return count;
}

static bool
TqGraphCollectPayloadExactBand(HnswScanOpaque so, HnswMetaPageData *meta,
							   TqGraphScanStorage *storage, Datum query,
							   int payloadSlot, int32 payloadValue,
							   TqGraphResult *results, int resultTarget,
							   int *count)
{
	uint32		payloadFirst;
	uint32		payloadCount;

	if (payloadSlot < 0 || resultTarget <= 0 ||
		!TqGraphPayloadRefRange(storage, payloadSlot, payloadValue,
								&payloadFirst, &payloadCount))
		return false;

	if (payloadCount > TQ_GRAPH_PAYLOAD_EXACT_MAX)
		return false;

	*count = 0;
	for (uint32 i = 0; i < payloadCount; i++)
	{
		uint32		nodeId = storage->payloadRefs[payloadFirst + i].nodeId;
		TqGraphScanNode *node;
		double		distance;
		bool		exactScored = false;

		if (nodeId >= meta->tqNodeCount)
			continue;

		node = &storage->nodes[nodeId];
		if (node->flags & TQ_GRAPH_NODE_DEAD ||
			!TqGraphNodeMatchesPayload(node, payloadSlot, payloadValue))
			continue;

		if (node->exactVector != NULL)
		{
			distance = TqGraphExactVectorDistance(so, query, node->exactVector);
			exactScored = true;
		}
		else
			distance = TqGraphScoreNode(so, node);

		TqGraphOfferCandidate(so, results, resultTarget, count, nodeId,
							  &node->heaptid, distance, exactScored);
	}

	so->graphEntryPointCount = 0;
	return true;
}

static int
TqGraphFinalRescoreCount(HnswScanOpaque so, TqGraphResult *results, int count,
						 int effectiveEf)
{
	(void) results;
	(void) effectiveEf;

	if (count <= 0 || so->graphRescoreBand == TQ_GRAPH_RESCORE_BAND_NONE)
		return 0;

	return count;
}

void
TqGraphExactRescore(Relation index, HnswScanOpaque so, Datum query,
					HnswMetaPageData *meta, TqGraphScanNode *nodes,
					TqGraphResult *results, int count)
{
	TqGraphRescoreRef *refs;
	TqGraphRescoreRef stackRefs[TQ_GRAPH_STACK_RESCORE_CAPACITY];
	bool		refsAllocated;
	int			refCount = 0;
	Size		vectorSize = VECTOR_SIZE(meta->dimensions);
	char	   *vectorScratch;

	if (DatumGetPointer(query) == NULL || count == 0 ||
		so->graphRescoreBand == TQ_GRAPH_RESCORE_BAND_NONE)
		return;

	vectorScratch = palloc(vectorSize);

	if (hnsw_tq_graph_stack_scratch && count <= TQ_GRAPH_STACK_RESCORE_CAPACITY)
	{
		refs = stackRefs;
		refsAllocated = false;
	}
	else
	{
		refs = palloc(sizeof(TqGraphRescoreRef) * count);
		refsAllocated = true;
	}
	for (int i = 0; i < count; i++)
	{
		TqGraphScanNode *node;

		if (results[i].nodeId >= meta->tqNodeCount)
			continue;
		if (results[i].exactScored)
			continue;

		node = &nodes[results[i].nodeId];
		if (node->exactVector != NULL)
		{
			results[i].distance =
				TqGraphExactVectorDistance(so, query, node->exactVector);
			so->graphRescoreCount++;
			results[i].exactScored = true;
			continue;
		}

		if (!BlockNumberIsValid(node->exactBlkno) ||
			!TqGraphExactByteOffsetIsValid(node->exactOffno))
			continue;

		refs[refCount].resultIndex = i;
		refs[refCount].nodeId = results[i].nodeId;
		refs[refCount].blkno = node->exactBlkno;
		refs[refCount].offno = node->exactOffno;
		refCount++;
	}

	qsort(refs, refCount, sizeof(TqGraphRescoreRef), TqGraphRescoreRefCompare);

	for (int i = 0; i < refCount; i++)
	{
		TqGraphScanNode *node;

		CHECK_FOR_INTERRUPTS();

		if (refs[i].nodeId >= meta->tqNodeCount)
			continue;

		node = &nodes[refs[i].nodeId];
		if (TqGraphReadExactVectorInto(index, node, meta->dimensions,
									   vectorScratch, so))
		{
			results[refs[i].resultIndex].distance =
				TqGraphExactVectorDistance(so, query, vectorScratch);
			so->graphRescoreCount++;
			results[refs[i].resultIndex].exactScored = true;
		}
	}

	pfree(vectorScratch);
	if (refsAllocated)
		pfree(refs);
}



static void
TqGraphCollectResults(IndexScanDesc scan, HnswScanOpaque so)
{
	HnswMetaPageData meta;
	instr_time	totalStart;
	instr_time	phaseStart;
	Datum		query = TqGraphGetScanValue(scan, so);
	int			resultTarget;
	int			searchEf;
	int			effectiveEf;
	int			count = 0;
	TqGraphResult *results;
	TqGraphScanStorage storage;
	int64		activeTarget;
	double		estimatedSelectivity;
	int			rescoreCount;
	int			finalCount;
	AttrNumber	payloadHeapAttno = InvalidAttrNumber;
	int32		payloadValue = 0;
	int			payloadSlot = -1;
	int			candidateOversampling;
	bool		hasPayloadFilter = false;

	INSTR_TIME_SET_CURRENT(totalStart);
	INSTR_TIME_SET_CURRENT(phaseStart);

	if (!TqGraphReadMeta(scan->indexRelation, &meta) ||
		meta.tqNodeCount == 0 ||
		!BlockNumberIsValid(meta.tqCodeStartBlkno) ||
		!BlockNumberIsValid(meta.tqAdjStartBlkno))
	{
		so->tqGraphResults = NULL;
		so->tqGraphResultCount = 0;
		return;
	}

	HnswPrepareTqQuery(scan->indexRelation, &so->support, query, &so->tq);
	activeTarget = TqGraphGetActiveLimitTupleTarget();
	estimatedSelectivity = TqGraphGetActiveEstimatedFilterSelectivity();
	if (TqGraphGetActivePayloadInt4Filter(&payloadHeapAttno, &payloadValue))
	{
		payloadSlot = TqGraphPayloadSlotForHeapAttr(scan->indexRelation,
												   payloadHeapAttno);
		hasPayloadFilter = payloadSlot >= 0 &&
			payloadSlot < meta.tqPayloadCount;
		if (!hasPayloadFilter)
			payloadSlot = -1;
	}
	if (!so->hasTupleTargetRows && activeTarget >= 0)
	{
		so->hasTupleTargetRows = true;
		so->tupleTargetRows = activeTarget;
	}
	TqGraphSeedScanContext(so, activeTarget, estimatedSelectivity);
	effectiveEf = so->hasInitialEffectiveEfSearch ?
		so->initialEffectiveEfSearch : so->efSearch;

	candidateOversampling = Max(so->graphOversampling, 1);
	if (hasPayloadFilter && (TqScoreMode) so->tq.scoreMode != TQ_SCORE_L2)
		candidateOversampling = Min(candidateOversampling, 2);

	if (so->hasTupleTargetRows)
	{
			resultTarget = (int) Min((int64) INT_MAX,
									 Max(Max((int64) 1, so->tupleTargetRows) *
										 candidateOversampling,
										 (int64) effectiveEf));
		if (estimatedSelectivity > 0 && estimatedSelectivity < 1)
		{
			int64		filteredTarget =
				(int64) ceil((double) Max((int64) 1, so->tupleTargetRows) /
							 estimatedSelectivity) *
				candidateOversampling;

			resultTarget = (int) Min((int64) INT_MAX,
									 Max((int64) resultTarget, filteredTarget));
		}
	}
	else
		resultTarget = effectiveEf;

	if (so->graphRescoreBand == TQ_GRAPH_RESCORE_BAND_AUTO &&
		(TqScoreMode) so->tq.scoreMode == TQ_SCORE_L2)
	{
			int64		l2Target = (int64) effectiveEf *
				Max(so->graphOversampling, 1);

		if (meta.dimensions >= 1024)
		{
				if (so->tq.bits < TQ_DEFAULT_BITS)
					l2Target *= TQ_GRAPH_LOWBIT_HIGHDIM_L2_TARGET_MULT;
				else
					l2Target = (int64) effectiveEf *
						TQ_GRAPH_HIGHDIM_L2_TARGET_EF_MULT;
		}

		resultTarget = (int) Min((int64) INT_MAX,
								 Max((int64) resultTarget, l2Target));
	}

	if (!hasPayloadFilter && estimatedSelectivity > 0 && estimatedSelectivity < 1)
	{
		int64		conservativeTarget =
			Min((int64) meta.tqNodeCount, (int64) hnsw_max_scan_tuples);

		/*
		 * Native graph scans currently receive planner selectivity but not
		 * the actual heap predicate. When clustered data makes the nearest
		 * global neighborhood mostly miss the filter, a k/selectivity band
		 * can return too few post-filter rows. Widen to the configured scan
		 * budget only when the predicate cannot be mapped to graph-owned
		 * payload columns.
		 */
		resultTarget = (int) Min((int64) INT_MAX,
								 Max((int64) resultTarget, conservativeTarget));
	}

	resultTarget = Max(resultTarget, 1);
	resultTarget = Min(resultTarget, (int) meta.tqNodeCount);
	searchEf = Min(Max(effectiveEf, resultTarget), (int) meta.tqNodeCount);
	results = palloc(sizeof(TqGraphResult) * resultTarget);
	TqGraphInitScanStorage(scan->indexRelation, &meta, &storage);
	TqGraphAddElapsedUs(&so->graphPrepareUs, phaseStart);

	if (hasPayloadFilter &&
		TqGraphCollectPayloadExactBand(so, &meta, &storage, query,
									   payloadSlot, payloadValue, results,
									   resultTarget, &count))
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		qsort(results, count, sizeof(TqGraphResult), TqGraphResultCompare);
		TqGraphAddElapsedUs(&so->graphSortUs, phaseStart);
		so->graphCandidateCount = count;
		so->tqGraphResults = results;
		so->tqGraphResultCount = count;
		so->tqGraphResultIndex = 0;
		TqGraphAddElapsedUs(&so->graphTotalUs, totalStart);
		HnswRecordGraphScanStats(so);
		return;
	}

	INSTR_TIME_SET_CURRENT(phaseStart);
	count = TqGraphTraverse(scan->indexRelation, so, &meta, &storage, results,
							resultTarget, searchEf, query, payloadSlot,
							payloadValue);
	TqGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
	if (estimatedSelectivity > 0 && estimatedSelectivity < 1 && count < resultTarget)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		count = TqGraphFillCandidateBand(scan->indexRelation, so, &meta,
										 &storage, results, resultTarget, count,
										 payloadSlot, payloadValue, query);
		TqGraphAddElapsedUs(&so->graphFillUs, phaseStart);
	}
	INSTR_TIME_SET_CURRENT(phaseStart);
	qsort(results, count, sizeof(TqGraphResult), TqGraphResultCompare);
	TqGraphAddElapsedUs(&so->graphSortUs, phaseStart);

	if (so->graphRescoreBand == TQ_GRAPH_RESCORE_BAND_AUTO &&
		(TqScoreMode) so->tq.scoreMode == TQ_SCORE_L2 &&
		count > 0 && count < resultTarget && results[0].distance < 1.0)
	{
		int			wideTarget = (int) Min((int64) meta.tqNodeCount,
											   Max((int64) resultTarget,
												   (int64) effectiveEf *
												   Max(so->graphOversampling, 1) *
												   TQ_GRAPH_TIGHT_L2_FILL_MULT));

		if (wideTarget > resultTarget)
		{
			pfree(results);
			resultTarget = wideTarget;
				searchEf = Min(Max(effectiveEf, resultTarget),
							   (int) meta.tqNodeCount);
			results = palloc(sizeof(TqGraphResult) * resultTarget);
			INSTR_TIME_SET_CURRENT(phaseStart);
			count = TqGraphTraverse(scan->indexRelation, so, &meta, &storage,
									results, resultTarget, searchEf, query,
									payloadSlot, payloadValue);
			TqGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
			if (count < resultTarget)
			{
				INSTR_TIME_SET_CURRENT(phaseStart);
				count = TqGraphFillCandidateBand(scan->indexRelation, so, &meta,
												 &storage, results,
												 resultTarget, count,
												 payloadSlot, payloadValue,
												 query);
				TqGraphAddElapsedUs(&so->graphFillUs, phaseStart);
			}
			INSTR_TIME_SET_CURRENT(phaseStart);
			qsort(results, count, sizeof(TqGraphResult), TqGraphResultCompare);
			TqGraphAddElapsedUs(&so->graphSortUs, phaseStart);
		}
	}

	so->graphCandidateCount = count;
	rescoreCount = TqGraphFinalRescoreCount(so, results, count, effectiveEf);
	finalCount = so->graphRescoreBand == TQ_GRAPH_RESCORE_BAND_AUTO ?
		rescoreCount : count;
	INSTR_TIME_SET_CURRENT(phaseStart);
	TqGraphExactRescore(scan->indexRelation, so, query, &meta, storage.nodes,
						results, rescoreCount);
	TqGraphAddElapsedUs(&so->graphRescoreUs, phaseStart);
	INSTR_TIME_SET_CURRENT(phaseStart);
	qsort(results, finalCount, sizeof(TqGraphResult), TqGraphResultCompare);
	TqGraphAddElapsedUs(&so->graphSortUs, phaseStart);
	so->tqGraphResults = results;
	so->tqGraphResultCount = finalCount;
	so->tqGraphResultIndex = 0;
	TqGraphAddElapsedUs(&so->graphTotalUs, totalStart);
	HnswRecordGraphScanStats(so);
}

bool
tqgraphgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	TqGraphResult *results;

	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with turboquant graph");

		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);
		TqGraphCollectResults(scan, so);
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);
		so->first = false;
	}

	if (so->tqGraphResultIndex >= so->tqGraphResultCount)
	{
		MemoryContextSwitchTo(oldCtx);
		return false;
	}

	results = (TqGraphResult *) so->tqGraphResults;
	scan->xs_heaptid = results[so->tqGraphResultIndex++].heaptid;
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	so->returnedRows++;

	MemoryContextSwitchTo(oldCtx);
	return true;
}

void
tqgraphendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);
	pfree(so);
	scan->opaque = NULL;
}

bool
tqgraphinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			  Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
			  ,bool indexUnchanged
#endif
			  ,IndexInfo *indexInfo
)
{
	Datum		value;
	const HnswTypeInfo *typeInfo = HnswGetTypeInfo(index);
	HnswSupport support;

	(void) heap;
	(void) checkUnique;
#if PG_VERSION_NUM >= 140000
	(void) indexUnchanged;
#endif
	if (isnull[0])
		return false;

	HnswInitSupport(&support, index);
	if (!HnswFormIndexValue(&value, values, isnull, typeInfo, &support))
		return false;

	LockPage(index, HNSW_UPDATE_LOCK, ExclusiveLock);
	PG_TRY();
	{
		TqGraphInsertValueInPlace(index, indexInfo, heap_tid, value,
								  values, isnull);
	}
	PG_CATCH();
	{
		UnlockPage(index, HNSW_UPDATE_LOCK, ExclusiveLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(index, HNSW_UPDATE_LOCK, ExclusiveLock);

	return true;
}

static bool
TqGraphRepairAdjacencyForDeadNodes(Relation index, HnswMetaPageData *meta,
								   bool *deadNodes)
{
	int			adjTuplesPerPage;
	int			adjPageCount;
	int			levelCapacity;
	bool		changedAny = false;
	BlockNumber blkno;
	BlockNumber nblocks;

	if (!BlockNumberIsValid(meta->tqAdjStartBlkno) || deadNodes == NULL)
		return false;

	levelCapacity = TqGraphLevelCapacity(meta->m);
	adjTuplesPerPage =
		TqGraphTuplesPerPage(TqGraphAdjTupleSize(TqGraphLevelM(meta->m, 0)));
	adjPageCount = TqGraphPageCount(TqGraphAdjRecordCount(meta), adjTuplesPerPage);
	blkno = meta->tqAdjStartBlkno;
	nblocks = RelationGetNumberOfBlocks(index);

	for (int pageNo = 0;
		 pageNo < adjPageCount && BlockNumberIsValid(blkno) && blkno < nblocks;
		 pageNo++)
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		BlockNumber nextblkno;
		OffsetNumber maxoff;
		bool		changed = false;
		GenericXLogState *xlogState = NULL;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		if (RelationNeedsWAL(index))
		{
			xlogState = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(xlogState, buf, 0);
		}
		else
			page = BufferGetPage(buf);

		opaque = HnswPageGetOpaque(page);
		if ((opaque->pageKind & HNSW_PAGE_KIND_MASK) != HNSW_PAGE_KIND_TQ_ADJ)
		{
			if (xlogState != NULL)
				GenericXLogAbort(xlogState);
			UnlockReleaseBuffer(buf);
			break;
		}
		nextblkno = opaque->nextblkno;

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			TqGraphAdjTuple tuple;
			uint16		maxNeighbors;
			uint16		oldCount;
			uint16		scanCount;
			uint16		newCount = 0;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != TQ_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount ||
				tuple->level >= levelCapacity)
				continue;

			maxNeighbors = TqGraphLevelM(meta->m, tuple->level);
			oldCount = tuple->count;
			scanCount = Min(oldCount, maxNeighbors);

			if (!deadNodes[tuple->nodeId])
			{
				for (int i = 0; i < scanCount; i++)
				{
					uint32		neighbor = tuple->neighbors[i];

					if (neighbor < meta->tqNodeCount && !deadNodes[neighbor])
						tuple->neighbors[newCount++] = neighbor;
				}
			}

			if (newCount != oldCount)
			{
				if (newCount < scanCount)
					memset(&tuple->neighbors[newCount], 0,
						   sizeof(uint32) * (scanCount - newCount));
				tuple->count = newCount;
				changed = true;
			}
		}

		if (changed)
			HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_VACUUM_REPAIR);

		if (xlogState != NULL)
		{
			if (changed)
				GenericXLogFinish(xlogState);
			else
				GenericXLogAbort(xlogState);
		}
		else if (changed)
			MarkBufferDirty(buf);

		UnlockReleaseBuffer(buf);

		if (changed)
		{
			changedAny = true;
			HnswLogGraphWalRecord(index, MAIN_FORKNUM, blkno, HNSW_GRAPH_OP_VACUUM_REPAIR);
		}

		blkno = nextblkno;
	}

	return changedAny;
}

void
TqGraphCollectVacuumStats(Relation index, HnswMetaPageData *meta,
						  int64 *liveNodes, int64 *deadNodes,
						  int64 *adjacencyRefs, int64 *deadNeighborRefs)
{
	int			codeTuplesPerPage;
	int			codePageCount;
	int			tqBits = meta->tqBits != 0 ? meta->tqBits : TQ_DEFAULT_BITS;
	int			adjTuplesPerPage;
	int			adjPageCount;
	int			levelCapacity;
	bool	   *deadBitmap;
	BlockNumber nblocks;
	BlockNumber blkno;

	*liveNodes = 0;
	*deadNodes = 0;
	*adjacencyRefs = 0;
	*deadNeighborRefs = 0;

	if (meta->tqNodeCount == 0 ||
		!BlockNumberIsValid(meta->tqCodeStartBlkno))
		return;

	deadBitmap = palloc0(sizeof(bool) * meta->tqNodeCount);
	nblocks = RelationGetNumberOfBlocks(index);
	codeTuplesPerPage =
		TqGraphTuplesPerPage(TqGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  tqBits,
												  (meta->tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0));
	codePageCount = TqGraphPageCount(meta->tqNodeCount, codeTuplesPerPage);
	blkno = meta->tqCodeStartBlkno;

	for (int pageNo = 0;
		 pageNo < codePageCount && BlockNumberIsValid(blkno) && blkno < nblocks;
		 pageNo++)
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();

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
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			TqGraphCodeTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqGraphCodeTuple) PageGetItem(page, iid);
			if (tuple->type != TQ_GRAPH_CODE_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount)
				continue;

			if (tuple->flags & TQ_GRAPH_NODE_DEAD)
			{
				deadBitmap[tuple->nodeId] = true;
				(*deadNodes)++;
			}
			else
				(*liveNodes)++;
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}

	if (!BlockNumberIsValid(meta->tqAdjStartBlkno))
	{
		pfree(deadBitmap);
		return;
	}

	levelCapacity = TqGraphLevelCapacity(meta->m);
	adjTuplesPerPage =
		TqGraphTuplesPerPage(TqGraphAdjTupleSize(TqGraphLevelM(meta->m, 0)));
	adjPageCount = TqGraphPageCount(TqGraphAdjRecordCount(meta), adjTuplesPerPage);
	blkno = meta->tqAdjStartBlkno;

	for (int pageNo = 0;
		 pageNo < adjPageCount && BlockNumberIsValid(blkno) && blkno < nblocks;
		 pageNo++)
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);
		if ((opaque->pageKind & HNSW_PAGE_KIND_MASK) != HNSW_PAGE_KIND_TQ_ADJ)
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			TqGraphAdjTuple tuple;
			uint16		maxNeighbors;
			bool		deadSource;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != TQ_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount ||
				tuple->level >= levelCapacity)
				continue;

			maxNeighbors = TqGraphLevelM(meta->m, tuple->level);
			deadSource = deadBitmap[tuple->nodeId];

			for (int i = 0; i < Min(tuple->count, maxNeighbors); i++)
			{
				uint32		neighbor = tuple->neighbors[i];

				(*adjacencyRefs)++;
				if (deadSource || neighbor >= meta->tqNodeCount ||
					deadBitmap[neighbor])
					(*deadNeighborRefs)++;
			}
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}

	pfree(deadBitmap);
}

IndexBulkDeleteResult *
tqgraphbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				  IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation	index = info->index;
	HnswMetaPageData meta;
	int			codeTuplesPerPage;
	int			codePageCount;
	int			tqBits;
	double		liveTuples = 0;
	bool		changedAny = false;
	bool		repairAny = false;
	bool		hasDeadNodes = false;
	bool	   *deadNodes = NULL;

	if (stats == NULL)
		stats = palloc0(sizeof(IndexBulkDeleteResult));

	if (callback == NULL || !TqGraphReadMeta(index, &meta) ||
		meta.tqNodeCount == 0 ||
		!BlockNumberIsValid(meta.tqCodeStartBlkno))
		return stats;

	tqBits = meta.tqBits != 0 ? meta.tqBits : TQ_DEFAULT_BITS;
	codeTuplesPerPage =
		TqGraphTuplesPerPage(TqGraphCodeTupleSize(meta.dimensions,
												  meta.tqPayloadCount,
												  tqBits,
												  (meta.tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0));
	codePageCount = TqGraphPageCount(meta.tqNodeCount, codeTuplesPerPage);
	deadNodes = palloc0(sizeof(bool) * meta.tqNodeCount);

	LockPage(index, HNSW_SCAN_LOCK, ExclusiveLock);
	PG_TRY();
	{
		BlockNumber blkno = meta.tqCodeStartBlkno;
		BlockNumber nblocks = RelationGetNumberOfBlocks(index);

		for (int pageNo = 0;
			 pageNo < codePageCount && BlockNumberIsValid(blkno) && blkno < nblocks;
			 pageNo++)
		{
			Buffer		buf;
			Page		page;
			HnswPageOpaque opaque;
			BlockNumber nextblkno;
			OffsetNumber maxoff;
			bool		changed = false;
			GenericXLogState *xlogState = NULL;

			CHECK_FOR_INTERRUPTS();

			buf = ReadBuffer(index, blkno);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

			if (RelationNeedsWAL(index))
			{
				xlogState = GenericXLogStart(index);
				page = GenericXLogRegisterBuffer(xlogState, buf, 0);
			}
			else
				page = BufferGetPage(buf);

			opaque = HnswPageGetOpaque(page);
			if ((opaque->pageKind & HNSW_PAGE_KIND_MASK) != HNSW_PAGE_KIND_TQ_CODE)
			{
				if (xlogState != NULL)
					GenericXLogAbort(xlogState);
				UnlockReleaseBuffer(buf);
				break;
			}
			nextblkno = opaque->nextblkno;

			maxoff = PageGetMaxOffsetNumber(page);
			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
			{
				ItemId		iid = PageGetItemId(page, offno);
				TqGraphCodeTuple tuple;

				if (!ItemIdIsUsed(iid))
					continue;

				tuple = (TqGraphCodeTuple) PageGetItem(page, iid);
				if (tuple->type != TQ_GRAPH_CODE_TUPLE_TYPE ||
					tuple->nodeId >= meta.tqNodeCount)
					continue;

				if (tuple->flags & TQ_GRAPH_NODE_DEAD)
				{
					deadNodes[tuple->nodeId] = true;
					hasDeadNodes = true;
					continue;
				}

				if (callback(&tuple->heaptid, callback_state))
				{
					tuple->flags |= TQ_GRAPH_NODE_DEAD;
					deadNodes[tuple->nodeId] = true;
					hasDeadNodes = true;
					stats->tuples_removed += 1;
					changed = true;
				}
				else
					liveTuples += 1;
			}

			if (changed)
				HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_VACUUM_DELETE);

			if (xlogState != NULL)
				GenericXLogFinish(xlogState);
			else if (changed)
				MarkBufferDirty(buf);

			UnlockReleaseBuffer(buf);

			if (changed)
			{
				changedAny = true;
				HnswLogGraphWalRecord(index, MAIN_FORKNUM, blkno, HNSW_GRAPH_OP_VACUUM_DELETE);
			}

			blkno = nextblkno;
		}

		if (hasDeadNodes)
			repairAny = TqGraphRepairAdjacencyForDeadNodes(index, &meta, deadNodes);

		if (changedAny || repairAny)
			TqGraphBumpMetaGeneration(index);
	}
	PG_CATCH();
	{
		UnlockPage(index, HNSW_SCAN_LOCK, ExclusiveLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(index, HNSW_SCAN_LOCK, ExclusiveLock);
	pfree(deadNodes);

	stats->num_index_tuples = liveTuples;
	stats->estimated_count = false;

	return stats;
}

IndexBulkDeleteResult *
tqgraphvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	(void) info;

	if (stats == NULL)
		stats = palloc0(sizeof(IndexBulkDeleteResult));

	return stats;
}
