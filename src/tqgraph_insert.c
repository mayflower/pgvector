#include "postgres.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "access/generic_xlog.h"
#include "catalog/pg_type_d.h"
#include "storage/bufmgr.h"
#include "utils/datum.h"
#include "utils/rel.h"

#include "tqgraph.h"
#include "tqgraph_score.h"

static int
TqGraphInsertResultCompare(const void *a, const void *b)
{
	const TqGraphResult *ia = (const TqGraphResult *) a;
	const TqGraphResult *ib = (const TqGraphResult *) b;

	if (ia->distance < ib->distance)
		return -1;
	if (ia->distance > ib->distance)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}

static int
TqGraphInsertFrontierCompare(const void *a, const void *b)
{
	const TqGraphFrontierItem *ia = (const TqGraphFrontierItem *) a;
	const TqGraphFrontierItem *ib = (const TqGraphFrontierItem *) b;

	if (ia->distance < ib->distance)
		return -1;
	if (ia->distance > ib->distance)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}

static double
TqGraphExactDistanceToNode(Relation index, HnswMetaPageData *meta,
						   TqGraphScanStorage *storage, HnswSupport *support,
						   Vector *query, uint32 nodeId)
{
	Vector	   *value;
	double		distance;

	if (nodeId >= meta->tqNodeCount ||
		!TqGraphLoadCodePage(index, NULL, meta, storage, nodeId) ||
		(storage->nodes[nodeId].flags & TQ_GRAPH_NODE_DEAD))
		return DBL_MAX;

	value = TqGraphReadExactVector(index, &storage->nodes[nodeId], meta->dimensions);
	if (value == NULL)
		return DBL_MAX;

	distance = TqGraphExactDistance(support, PointerGetDatum(query),
									PointerGetDatum(value));
	pfree(value);

	return distance;
}

static bool
TqGraphLoadAdjTuple(Relation index, HnswMetaPageData *meta, uint32 nodeId,
					int level, uint32 *neighbors, int *count)
{
	int			slot;
	int			pageNo;
	int			adjTuplesPerPage;
	int			adjPageCount;
	BlockNumber blkno;
	Buffer		buf;
	Page		page;
	OffsetNumber maxoff;
	bool		found = false;

	*count = 0;
	if (nodeId >= meta->tqNodeCount || level < 0 ||
		level >= TqGraphLevelCapacity(meta->m))
		return false;

	adjTuplesPerPage = TqGraphTuplesPerPage(TqGraphAdjTupleSize(TqGraphLevelM(meta->m, 0)));
	adjPageCount = TqGraphPageCount(TqGraphAdjRecordCount(meta), adjTuplesPerPage);
	slot = TqGraphAdjSlot(meta, nodeId, level);
	pageNo = slot / adjTuplesPerPage;
	blkno = TqGraphGetChainBlockNumber(index, meta->tqAdjStartBlkno, pageNo,
										adjPageCount, HNSW_PAGE_KIND_TQ_ADJ);
	if (!BlockNumberIsValid(blkno))
		return false;

	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	maxoff = PageGetMaxOffsetNumber(page);
	for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
	{
		ItemId		iid = PageGetItemId(page, offno);
		TqGraphAdjTuple tuple = (TqGraphAdjTuple) PageGetItem(page, iid);

		if (tuple->type == TQ_GRAPH_ADJ_TUPLE_TYPE &&
			tuple->nodeId == nodeId &&
			tuple->level == level)
		{
			*count = tuple->count;
			memcpy(neighbors, tuple->neighbors, sizeof(uint32) * tuple->count);
			found = true;
			break;
		}
	}
	UnlockReleaseBuffer(buf);

	return found;
}

static void
TqGraphUpdateAdjTuple(Relation index, HnswMetaPageData *meta, uint32 nodeId,
					  int level, uint32 *neighbors, int count)
{
	int			slot;
	int			pageNo;
	int			adjTuplesPerPage;
	int			adjPageCount;
	BlockNumber blkno;
	Buffer		buf;
	Page		page;
	OffsetNumber maxoff;
	GenericXLogState *xlogState = NULL;
	bool		found = false;

	if (nodeId >= meta->tqNodeCount || level < 0 ||
		level >= TqGraphLevelCapacity(meta->m) ||
		count > TqGraphLevelM(meta->m, level))
		elog(ERROR, "invalid turboquant graph adjacency update");

	adjTuplesPerPage = TqGraphTuplesPerPage(TqGraphAdjTupleSize(TqGraphLevelM(meta->m, 0)));
	adjPageCount = TqGraphPageCount(TqGraphAdjRecordCount(meta), adjTuplesPerPage);
	slot = TqGraphAdjSlot(meta, nodeId, level);
	pageNo = slot / adjTuplesPerPage;
	blkno = TqGraphGetChainBlockNumber(index, meta->tqAdjStartBlkno, pageNo,
										adjPageCount, HNSW_PAGE_KIND_TQ_ADJ);
	if (!BlockNumberIsValid(blkno))
		elog(ERROR, "missing turboquant graph adjacency page");

	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}
	else
		page = BufferGetPage(buf);

	maxoff = PageGetMaxOffsetNumber(page);
	for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
	{
		ItemId		iid = PageGetItemId(page, offno);
		TqGraphAdjTuple tuple = (TqGraphAdjTuple) PageGetItem(page, iid);

		if (tuple->type == TQ_GRAPH_ADJ_TUPLE_TYPE &&
			tuple->nodeId == nodeId &&
			tuple->level == level)
		{
			tuple->count = count;
			memcpy(tuple->neighbors, neighbors, sizeof(uint32) * count);
			HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_NEIGHBOR_UPDATE);
			found = true;
			break;
		}
	}

	if (!found)
		elog(ERROR, "missing turboquant graph adjacency tuple");

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);

	HnswLogGraphWalRecord(index, MAIN_FORKNUM, blkno, HNSW_GRAPH_OP_NEIGHBOR_UPDATE);
}

static bool
TqGraphSelectedContains(uint32 *selected, int selectedCount, uint32 nodeId)
{
	for (int i = 0; i < selectedCount; i++)
	{
		if (selected[i] == nodeId)
			return true;
	}

	return false;
}

static bool
TqGraphInsertCandidateDiverse(Relation index, HnswMetaPageData *meta,
							  TqGraphScanStorage *storage,
							  HnswSupport *support, uint32 candidate,
							  double candidateDistance, uint32 *selected,
							  int selectedCount)
{
	Vector	   *candidateVector;
	bool		good = true;

	if (selectedCount == 0)
		return true;

	candidateVector = TqGraphReadExactVector(index, &storage->nodes[candidate],
											 meta->dimensions);
	if (candidateVector == NULL)
		return true;

	for (int i = 0; i < selectedCount; i++)
	{
		double		selectedDistance;

		selectedDistance = TqGraphExactDistanceToNode(index, meta, storage,
													  support, candidateVector,
													  selected[i]);
		if (selectedDistance < candidateDistance)
		{
			good = false;
			break;
		}
	}

	pfree(candidateVector);
	return good;
}

static void
TqGraphSelectInsertNeighbors(Relation index, HnswMetaPageData *meta,
							 TqGraphScanStorage *storage,
							 HnswSupport *support,
							 TqGraphResult *candidates, int candidateCount,
							 int nodeLevel, uint32 **selected,
							 int *selectedCounts)
{
	qsort(candidates, candidateCount, sizeof(TqGraphResult), TqGraphInsertResultCompare);

	for (int level = 0; level <= nodeLevel; level++)
	{
		int			maxNeighbors = TqGraphLevelM(meta->m, level);

		for (int i = 0; i < candidateCount &&
			 selectedCounts[level] < maxNeighbors; i++)
		{
			uint32		nodeId = candidates[i].nodeId;

			if (nodeId >= meta->tqNodeCount ||
				storage->nodes[nodeId].level < level ||
				(storage->nodes[nodeId].flags & TQ_GRAPH_NODE_DEAD))
				continue;

			if (!TqGraphInsertCandidateDiverse(index, meta, storage, support,
											   nodeId, candidates[i].distance,
											   selected[level],
											   selectedCounts[level]))
				continue;

			selected[level][selectedCounts[level]++] = nodeId;
		}

		for (int i = 0; i < candidateCount &&
			 selectedCounts[level] < maxNeighbors; i++)
		{
			uint32		nodeId = candidates[i].nodeId;

			if (nodeId >= meta->tqNodeCount ||
				storage->nodes[nodeId].level < level ||
				(storage->nodes[nodeId].flags & TQ_GRAPH_NODE_DEAD) ||
				TqGraphSelectedContains(selected[level],
										selectedCounts[level], nodeId))
				continue;

			selected[level][selectedCounts[level]++] = nodeId;
		}
	}
}

static void
TqGraphUpdateReciprocalNeighbor(Relation index, HnswMetaPageData *meta,
								TqGraphScanStorage *storage,
								HnswSupport *support, Vector *newVector,
								uint32 newNodeId, uint32 src, int level)
{
	int			maxNeighbors = TqGraphLevelM(meta->m, level);
	uint32	   *neighbors;
	uint32	   *pruned;
	int			count;
	bool		found = false;
	Vector	   *sourceVector = NULL;

	if (src >= meta->tqNodeCount ||
		!TqGraphLoadCodePage(index, NULL, meta, storage, src) ||
		(storage->nodes[src].flags & TQ_GRAPH_NODE_DEAD) ||
		storage->nodes[src].level < level)
		return;

	neighbors = palloc0(sizeof(uint32) * (maxNeighbors + 1));
	pruned = palloc0(sizeof(uint32) * maxNeighbors);
	if (!TqGraphLoadAdjTuple(index, meta, src, level, neighbors, &count))
	{
		pfree(neighbors);
		pfree(pruned);
		return;
	}

	for (int i = 0; i < count; i++)
	{
		if (neighbors[i] == newNodeId)
		{
			found = true;
			break;
		}
	}

	if (!found)
		neighbors[count++] = newNodeId;

	if (count > maxNeighbors)
	{
		TqGraphFrontierItem *ranked = palloc(sizeof(TqGraphFrontierItem) * count);
		int			prunedCount = 0;

		sourceVector = TqGraphReadExactVector(index, &storage->nodes[src],
											  meta->dimensions);
		if (sourceVector == NULL)
		{
			pfree(ranked);
			pfree(neighbors);
			pfree(pruned);
			return;
		}

		for (int i = 0; i < count; i++)
		{
			ranked[i].nodeId = neighbors[i];
			if (neighbors[i] == newNodeId)
				ranked[i].distance = TqGraphExactDistance(support,
														  PointerGetDatum(sourceVector),
														  PointerGetDatum(newVector));
			else
				ranked[i].distance = TqGraphExactDistanceToNode(index, meta, storage,
																support, sourceVector,
																neighbors[i]);
		}

		qsort(ranked, count, sizeof(TqGraphFrontierItem), TqGraphInsertFrontierCompare);
		for (int i = 0; i < count && prunedCount < maxNeighbors; i++)
		{
			if (ranked[i].distance < DBL_MAX)
				pruned[prunedCount++] = ranked[i].nodeId;
		}

		memcpy(neighbors, pruned, sizeof(uint32) * prunedCount);
		count = prunedCount;
		pfree(ranked);
		pfree(sourceVector);
	}

	TqGraphUpdateAdjTuple(index, meta, src, level, neighbors, count);
	pfree(neighbors);
	pfree(pruned);
}

static void
TqGraphUpdateReciprocalNeighbors(Relation index, HnswMetaPageData *meta,
								 TqGraphScanStorage *storage,
								 HnswSupport *support, Vector *newVector,
								 uint32 newNodeId, int nodeLevel,
								 uint32 **selected, int *selectedCounts)
{
	for (int level = 0; level <= nodeLevel; level++)
	{
		for (int i = 0; i < selectedCounts[level]; i++)
			TqGraphUpdateReciprocalNeighbor(index, meta, storage, support,
											newVector, newNodeId,
											selected[level][i], level);
	}
}



static void
TqGraphAppendInsertedCode(Relation index, BlockNumber *codeStart,
						  uint32 nodeId, ItemPointer heapTid, int nodeLevel,
						  Vector *vector, uint8 *code, float scale,
						  int32 *payloads, uint16 payloadMask, int payloadCount,
						  int bits, BlockNumber exactBlkno, OffsetNumber exactOffno,
						  bool tqWeighted, float ecCorrection)
{
	Size		payloadBytes = TqGraphPayloadBytes(payloadCount);
	Size		tupleSize = TqGraphCodeTupleSize(vector->dim, payloadCount,
												 bits, tqWeighted);
	TqGraphCodeTuple tuple = palloc0(tupleSize);
	BlockNumber codeBlkno;

	tuple->type = TQ_GRAPH_CODE_TUPLE_TYPE;
	tuple->level = nodeLevel;
	tuple->flags = 0;
	tuple->nodeId = nodeId;
	tuple->heaptid = *heapTid;
	tuple->exactBlkno = exactBlkno;
	tuple->exactOffno = exactOffno;
	tuple->payloadMask = payloadMask;
	tuple->scale = scale;
	tuple->norm = TqGraphVectorNorm(vector);
	tuple->correction = TqGraphCodeNorm(code, vector->dim, bits);
	TqGraphTupleSetEcCorrection(tuple, tqWeighted, ecCorrection);
	if (payloadBytes > 0 && payloads != NULL)
		memcpy(TqGraphTuplePayloads(tuple, tqWeighted), payloads, payloadBytes);
	memcpy(TqGraphTupleCode(tuple, payloadBytes, tqWeighted), code,
		   TqGraphCodeBytesForBits(vector->dim, bits));

	(void) TqGraphAppendTuple(index, MAIN_FORKNUM, codeStart,
							  HNSW_PAGE_KIND_TQ_CODE, (Item) tuple,
							  tupleSize, HNSW_GRAPH_OP_ELEMENT_INSERT,
							  &codeBlkno);
	pfree(tuple);
}

static void
TqGraphAppendInsertedAdj(Relation index, BlockNumber *adjStart, int m,
						 uint32 nodeId, int nodeLevel, uint32 **selected,
						 int *selectedCounts)
{
	Size		tupleSize = TqGraphAdjTupleSize(TqGraphLevelM(m, 0));
	TqGraphAdjTuple tuple = palloc0(tupleSize);

	for (int level = 0; level < TqGraphLevelCapacity(m); level++)
	{
		BlockNumber adjBlkno;
		int			count = level <= nodeLevel ? selectedCounts[level] : 0;

		memset(tuple, 0, tupleSize);
		tuple->type = TQ_GRAPH_ADJ_TUPLE_TYPE;
		tuple->level = level;
		tuple->count = count;
		tuple->nodeId = nodeId;
		for (int i = 0; i < count; i++)
			tuple->neighbors[i] = selected[level][i];

		(void) TqGraphAppendTuple(index, MAIN_FORKNUM, adjStart,
								  HNSW_PAGE_KIND_TQ_ADJ, (Item) tuple,
								  tupleSize, HNSW_GRAPH_OP_NEIGHBOR_INSERT,
								  &adjBlkno);
	}

	pfree(tuple);
}

bool
TqGraphInsertValueInPlace(Relation index, IndexInfo *indexInfo,
						  ItemPointer heap_tid, Datum value,
						  Datum *values, bool *isnull)
{
	HnswMetaPageData meta;
	TqGraphScanStorage storage;
	HnswScanOpaqueData insertSo;
	HnswSupport support;
	TqGraphBuildState metaState;
	TqGraphBuildNode *metaNodes;
	Vector	   *vector = (Vector *) DatumGetPointer(value);
	uint8	   *code;
	float		scale;
	uint32		newNodeId;
	int			nodeLevel;
	int			levelCapacity;
	uint32	  **selected;
	int		   *selectedCounts;
	BlockNumber codeStart;
	BlockNumber adjStart;
	BlockNumber exactStart;
	BlockNumber exactBlkno;
	OffsetNumber exactOffno;
	uint32		entryNodeId;
	int			entryLevel;
	int			payloadCount;
	Size		payloadBytes;
	int32	   *payloads = NULL;
	uint16		payloadMask = 0;
	float	   *ecShift = NULL;
	float	   *ecScale = NULL;
	bool		insertTqWeighted;
	bool		insertTqRenorm;
	bool		insertExactStorage;
	float		insertXm;

	if (!TqGraphReadMeta(index, &meta))
		elog(ERROR, "turboquant native graph metapage is missing or invalid");

	if (meta.dimensions != 0 && meta.dimensions != vector->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions are not supported in the same turboquant graph")));

	newNodeId = meta.tqNodeCount;
	nodeLevel = TqGraphPickLevel(newNodeId, meta.m);
	levelCapacity = TqGraphLevelCapacity(meta.m);
	payloadCount = meta.tqPayloadCount;
	payloadBytes = TqGraphPayloadBytes(payloadCount);
	if (payloadCount > 0)
	{
		TqGraphBuildState payloadState;

		memset(&payloadState, 0, sizeof(payloadState));
		payloadState.index = index;
		payloadState.indexInfo = indexInfo;
		payloadState.payloadCount = payloadCount;
		payloadState.payloadBytes = payloadBytes;
		payloads = palloc0(payloadBytes);
		TqGraphCopyPayloadValues(&payloadState, payloads, &payloadMask,
								 values, isnull);
	}
	(void) TqGraphLoadCorrection(index, vector->dim, &ecShift, &ecScale);
	code = palloc(TqGraphCodeBytesForBits(vector->dim, meta.tqBits));
	insertTqWeighted = (meta.tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0;
	insertTqRenorm = (meta.tqFlags & TQ_GRAPH_TQ_RENORM) != 0;
	insertExactStorage = (meta.tqFlags & TQ_GRAPH_EXACT_FREE) == 0;
	insertXm = 0.0f;
	if (ecShift != NULL && ecScale != NULL)
	{
		if (insertTqWeighted && insertTqRenorm)
		{
			float		centroidNorm;
			double		dimSqrt;

			scale = TqEncodeVectorWithCorrectionXmRenormBits(vector, code, meta.tqBits,
															 ecShift, ecScale,
															 &insertXm, &centroidNorm);
			if (scale != 0.0f && centroidNorm > 0.0f)
			{
				dimSqrt = sqrt((double) vector->dim);
				scale = (float) ((double) scale * dimSqrt / (double) centroidNorm);
			}
		}
		else if (insertTqWeighted)
			scale = TqEncodeVectorWithCorrectionAndXmBits(vector, code, meta.tqBits,
														   ecShift, ecScale, &insertXm);
		else
			scale = TqEncodeVectorWithCorrectionBits(vector, code, meta.tqBits,
													 ecShift, ecScale);
	}
	else
		scale = TqEncodeVectorBits(vector, code, meta.tqBits);
	selected = palloc0(sizeof(uint32 *) * levelCapacity);
	selectedCounts = palloc0(sizeof(int) * levelCapacity);
	for (int level = 0; level < levelCapacity; level++)
		selected[level] = palloc0(sizeof(uint32) * TqGraphLevelM(meta.m, level));

	HnswInitSupport(&support, index);

	if (meta.tqNodeCount > 0)
	{
		TqGraphResult *candidates;
		int			resultTarget;
		int			searchEf;
		int			candidateCount;
		Datum		query = PointerGetDatum(vector);

		memset(&insertSo, 0, sizeof(insertSo));
		insertSo.support = support;
		insertSo.efSearch = Max(meta.efConstruction, TqGraphLevelM(meta.m, 0));
		insertSo.graphOversampling = 1;
		insertSo.graphRescoreBand = insertExactStorage ?
			TQ_GRAPH_RESCORE_BAND_EXACT : TQ_GRAPH_RESCORE_BAND_NONE;
		HnswPrepareTqQuery(index, &support, query, &insertSo.tq);

		TqGraphInitScanStorage(index, &meta, &storage);
		resultTarget = Min(Max(meta.efConstruction, TqGraphLevelM(meta.m, 0)),
						   (int) meta.tqNodeCount);
		searchEf = resultTarget;
		candidates = palloc0(sizeof(TqGraphResult) * resultTarget);
		candidateCount = TqGraphTraverse(index, &insertSo, &meta, &storage,
										  candidates, resultTarget, searchEf,
										  query, -1, 0);
		if (insertExactStorage)
			TqGraphExactRescore(index, &insertSo, query, &meta, storage.nodes,
								candidates, candidateCount);
		TqGraphSelectInsertNeighbors(index, &meta, &storage, &support,
									 candidates, candidateCount, nodeLevel,
									 selected, selectedCounts);
		TqGraphUpdateReciprocalNeighbors(index, &meta, &storage, &support,
										 vector, newNodeId, nodeLevel,
										 selected, selectedCounts);
		pfree(candidates);
	}

	codeStart = meta.tqCodeStartBlkno;
	adjStart = meta.tqAdjStartBlkno;
	exactStart = meta.tqExactStartBlkno;
	if (insertExactStorage)
		TqGraphAppendInsertedExact(index, &exactStart, newNodeId, vector,
								   vector->dim, &exactBlkno, &exactOffno);
	else
	{
		exactBlkno = InvalidBlockNumber;
		exactOffno = InvalidOffsetNumber;
	}
	TqGraphAppendInsertedCode(index, &codeStart, newNodeId, heap_tid, nodeLevel,
							  vector, code, scale, payloads, payloadMask,
							  payloadCount, meta.tqBits, exactBlkno, exactOffno,
							  insertTqWeighted, insertXm);
	TqGraphAppendInsertedAdj(index, &adjStart, meta.m, newNodeId, nodeLevel,
							 selected, selectedCounts);

	entryNodeId = meta.tqNodeCount == 0 || nodeLevel > meta.graphMaxLevel ?
		newNodeId : meta.tqEntryNodeId;
	entryLevel = entryNodeId == newNodeId ? nodeLevel : meta.entryLevel;

	memset(&metaState, 0, sizeof(metaState));
	metaState.index = index;
	metaState.forkNum = MAIN_FORKNUM;
	metaState.building = false;
	metaState.dimensions = vector->dim;
	metaState.m = meta.m;
	metaState.efConstruction = meta.efConstruction;
	metaState.tqBits = meta.tqBits;
	metaState.tqWeighted = (meta.tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0;
	metaState.tqRenorm = (meta.tqFlags & TQ_GRAPH_TQ_RENORM) != 0;
	metaState.tqExactStorage = insertExactStorage;
	metaState.payloadCount = payloadCount;
	metaState.payloadBytes = payloadBytes;
	metaState.ecShift = ecShift;
	metaState.ecScale = ecScale;
	metaState.nodeCount = meta.tqNodeCount + 1;
	metaState.maxLevel = Max(meta.graphMaxLevel, nodeLevel);
	metaState.entryNodeId = entryNodeId;
	metaNodes = palloc0(sizeof(TqGraphBuildNode) * metaState.nodeCount);
	metaNodes[entryNodeId].level = entryLevel;
	metaState.nodes = metaNodes;
	TqGraphUpdateMetaPage(index, &metaState, codeStart, adjStart, exactStart,
						  meta.tqCorrectionStartBlkno);
	TqGraphInvalidateCaches(index);
	pfree(metaNodes);

	for (int level = 0; level < levelCapacity; level++)
		pfree(selected[level]);
	pfree(selected);
	pfree(selectedCounts);
	if (payloads != NULL)
		pfree(payloads);
	if (ecShift != NULL)
		pfree(ecShift);
	if (ecScale != NULL)
		pfree(ecScale);
	pfree(code);

	return true;
}
