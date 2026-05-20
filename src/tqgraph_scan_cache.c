#include "postgres.h"

#include <string.h>

#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "tqgraph.h"

static TqGraphNativeCache *tqGraphCacheList = NULL;

static int
TqGraphPayloadRefCompare(const void *a, const void *b)
{
	const TqGraphPayloadRef *ra = (const TqGraphPayloadRef *) a;
	const TqGraphPayloadRef *rb = (const TqGraphPayloadRef *) b;

	if (ra->payloadSlot < rb->payloadSlot)
		return -1;
	if (ra->payloadSlot > rb->payloadSlot)
		return 1;
	if (ra->payloadValue < rb->payloadValue)
		return -1;
	if (ra->payloadValue > rb->payloadValue)
		return 1;
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}


static bool
TqGraphShouldCacheExactVectors(Relation index, HnswMetaPageData *meta)
{
	int			policy = HnswGetGraphExactCache(index);
	Size		exactBytes;
	Size		totalBytes;

	if (policy == TQ_GRAPH_EXACT_CACHE_OFF ||
		meta->tqNodeCount == 0 || meta->dimensions == 0 ||
		!BlockNumberIsValid(meta->tqExactStartBlkno))
		return false;
	if (policy == TQ_GRAPH_EXACT_CACHE_ON)
		return true;

	exactBytes = VECTOR_SIZE(meta->dimensions);
	if (meta->tqNodeCount > SIZE_MAX / exactBytes)
		return false;
	totalBytes = exactBytes * (Size) meta->tqNodeCount;

	return totalBytes <= TQ_GRAPH_EXACT_CACHE_AUTO_MAX_BYTES;
}

static void
TqGraphInitScanStorageUncached(HnswMetaPageData *meta, TqGraphScanStorage *storage,
							   bool cacheExactVectors)
{
	memset(storage, 0, sizeof(TqGraphScanStorage));
	storage->ctx = CurrentMemoryContext;
	storage->nodes = palloc0(sizeof(TqGraphScanNode) * meta->tqNodeCount);
	if (meta->tqNodeCount > 0 && meta->tqCodeBytes > 0)
		storage->codeArena = palloc0((Size) meta->tqNodeCount * meta->tqCodeBytes);
	if (meta->tqNodeCount > 0 && meta->tqPayloadBytes > 0)
		storage->payloadArena = palloc0((Size) meta->tqNodeCount * meta->tqPayloadBytes);
	if (cacheExactVectors && meta->tqNodeCount > 0 && meta->dimensions > 0)
	{
		storage->exactBytes = VECTOR_SIZE(meta->dimensions);
		storage->exactArena = palloc0((Size) meta->tqNodeCount * storage->exactBytes);
	}
	storage->levelCount = TqGraphLevelCapacity(meta->m);
	storage->neighbors = palloc0(sizeof(uint32 *) * TqGraphAdjRecordCount(meta));
	storage->neighborCounts = palloc0(sizeof(uint16) * TqGraphAdjRecordCount(meta));
	storage->codeTuplesPerPage =
		TqGraphTuplesPerPage(TqGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  (meta->tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0));
	storage->codePageCount = TqGraphPageCount(meta->tqNodeCount, storage->codeTuplesPerPage);
	storage->adjPageCount = BlockNumberIsValid(meta->tqAdjStartBlkno) ? 1 : 0;
	storage->codePagesLoaded = palloc0(sizeof(bool) * storage->codePageCount);
	if (storage->adjPageCount > 0)
		storage->adjPagesLoaded = palloc0(sizeof(bool) * storage->adjPageCount);
	storage->codeBlknos = palloc(sizeof(BlockNumber) * storage->codePageCount);
	TqGraphInitBlockMap(storage->codeBlknos, storage->codePageCount);
	storage->adjBlknos = palloc(sizeof(BlockNumber) * TqGraphAdjRecordCount(meta));
	storage->adjOffnos = palloc(sizeof(OffsetNumber) * TqGraphAdjRecordCount(meta));
	for (int i = 0; i < TqGraphAdjRecordCount(meta); i++)
	{
		storage->adjBlknos[i] = InvalidBlockNumber;
		storage->adjOffnos[i] = InvalidOffsetNumber;
	}
}

bool
TqGraphLoadCodePage(Relation index, HnswScanOpaque so, HnswMetaPageData *meta,
					TqGraphScanStorage *storage, uint32 nodeId)
{
	int			pageNo;
	BlockNumber blkno;

	if (nodeId >= meta->tqNodeCount || !BlockNumberIsValid(meta->tqCodeStartBlkno))
		return false;

	if (storage->cached)
		return storage->nodes[nodeId].loaded;

	pageNo = nodeId / storage->codeTuplesPerPage;
	if (pageNo < 0 || pageNo >= storage->codePageCount)
		return false;

	if (storage->codePagesLoaded[pageNo])
		return storage->nodes[nodeId].loaded;

	blkno = TqGraphGetMappedBlockNumber(meta->tqCodeStartBlkno, pageNo,
										 storage->codeBlknos);

	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;
		bool		fallbackTried = false;

retry:

		CHECK_FOR_INTERRUPTS();
		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);
		if ((opaque->pageKind & HNSW_PAGE_KIND_MASK) != HNSW_PAGE_KIND_TQ_CODE)
		{
			UnlockReleaseBuffer(buf);
			if (!fallbackTried)
			{
				fallbackTried = true;
				if (!TqGraphResolveChainBlockNumber(index, meta->tqCodeStartBlkno,
													 pageNo, storage->codePageCount,
													 HNSW_PAGE_KIND_TQ_CODE,
													 storage->codeBlknos, &blkno))
					return false;
				goto retry;
			}
			return false;
		}

		if (so != NULL)
			so->graphCodePagesRead++;
		maxoff = PageGetMaxOffsetNumber(page);
		{
			bool		tqWeighted = (meta->tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0;

			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
			{
				ItemId		iid = PageGetItemId(page, offno);
				TqGraphCodeTuple tuple = (TqGraphCodeTuple) PageGetItem(page, iid);
				TqGraphScanNode *node;

				if (tuple->type != TQ_GRAPH_CODE_TUPLE_TYPE ||
					tuple->nodeId >= meta->tqNodeCount)
					continue;

				node = &storage->nodes[tuple->nodeId];
				node->heaptid = tuple->heaptid;
				node->level = tuple->level;
				node->exactBlkno = tuple->exactBlkno;
				node->exactOffno = tuple->exactOffno;
				node->payloadMask = tuple->payloadMask;
				node->scale = tuple->scale;
				node->norm = tuple->norm;
				node->codeNorm = tuple->correction;
				node->ecCorrection = TqGraphTupleEcCorrection(tuple, tqWeighted);
				node->flags = tuple->flags;
				if (meta->tqPayloadBytes > 0 && storage->payloadArena != NULL)
				{
					node->payloads = (int32 *) (storage->payloadArena +
												((Size) tuple->nodeId * meta->tqPayloadBytes));
					memcpy(node->payloads, TqGraphTuplePayloads(tuple, tqWeighted),
						   meta->tqPayloadBytes);
				}
				node->code = storage->codeArena + ((Size) tuple->nodeId * meta->tqCodeBytes);
				memcpy(node->code, TqGraphTupleCode(tuple, meta->tqPayloadBytes, tqWeighted),
					   meta->tqCodeBytes);
				node->loaded = true;
			}
		}

		storage->codePagesLoaded[pageNo] = true;
		UnlockReleaseBuffer(buf);
	}

	return storage->nodes[nodeId].loaded;
}

static void
TqGraphLoadAllAdjPages(Relation index, HnswScanOpaque so, HnswMetaPageData *meta,
					   TqGraphScanStorage *storage)
{
	BlockNumber blkno = meta->tqAdjStartBlkno;
	BlockNumber nblocks;

	if (!BlockNumberIsValid(blkno) || storage->adjPageCount <= 0 ||
		storage->adjPagesLoaded == NULL || storage->adjPagesLoaded[0])
		return;

	nblocks = RelationGetNumberOfBlocks(index);
	while (BlockNumberIsValid(blkno) && blkno < nblocks)
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

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

		if (so != NULL)
			so->graphAdjPagesRead++;
		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			TqGraphAdjTuple tuple;
			int			tupleSlot;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != TQ_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount ||
				tuple->level > meta->graphMaxLevel ||
				tuple->level >= storage->levelCount ||
				tuple->count > TqGraphLevelM(meta->m, tuple->level))
				continue;

			tupleSlot = TqGraphAdjSlot(meta, tuple->nodeId, tuple->level);
			storage->adjBlknos[tupleSlot] = blkno;
			storage->adjOffnos[tupleSlot] = offno;
			storage->neighborCounts[tupleSlot] = tuple->count;
			if (storage->neighbors[tupleSlot] != NULL)
			{
				pfree(storage->neighbors[tupleSlot]);
				storage->neighbors[tupleSlot] = NULL;
			}
			if (tuple->count > 0)
			{
				storage->neighbors[tupleSlot] =
					MemoryContextAlloc(storage->ctx, sizeof(uint32) * tuple->count);
				memcpy(storage->neighbors[tupleSlot], tuple->neighbors, sizeof(uint32) * tuple->count);
			}
		}

		UnlockReleaseBuffer(buf);
		if (nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	storage->adjPagesLoaded[0] = true;
}

bool
TqGraphLoadAdjPage(Relation index, HnswScanOpaque so, HnswMetaPageData *meta,
				   TqGraphScanStorage *storage, uint32 nodeId, int level)
{
	if (nodeId >= meta->tqNodeCount || level < 0 || level > meta->graphMaxLevel ||
		!BlockNumberIsValid(meta->tqAdjStartBlkno))
		return false;

	if (storage->cached)
		return true;

	TqGraphLoadAllAdjPages(index, so, meta, storage);
	return true;
}

static bool
TqGraphLoadExactVectors(Relation index, HnswMetaPageData *meta,
						TqGraphScanStorage *storage)
{
	if (storage->exactArena == NULL ||
		!BlockNumberIsValid(meta->tqExactStartBlkno))
		return false;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeId];
		char	   *exactVector;

		CHECK_FOR_INTERRUPTS();
		if (!node->loaded)
			continue;

		exactVector = storage->exactArena +
			((Size) nodeId * storage->exactBytes);
		if (!TqGraphReadExactVectorInto(index, node, meta->dimensions,
										exactVector, NULL))
			return false;
		node->exactVector = exactVector;
	}

	return true;
}

static void
TqGraphBuildPayloadRefs(HnswMetaPageData *meta, TqGraphScanStorage *storage)
{
	uint32		refCount = 0;
	uint32		refIndex = 0;

	if (meta->tqPayloadCount == 0 || meta->tqPayloadBytes == 0)
		return;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeId];

		if (!node->loaded || node->payloads == NULL ||
			node->flags & TQ_GRAPH_NODE_DEAD)
			continue;

		for (int slot = 0; slot < meta->tqPayloadCount; slot++)
		{
			if (node->payloadMask & (uint16) (1U << slot))
				refCount++;
		}
	}

	if (refCount == 0)
		return;

	storage->payloadRefs = palloc(sizeof(TqGraphPayloadRef) * refCount);
	storage->payloadRefCount = refCount;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeId];

		if (!node->loaded || node->payloads == NULL ||
			node->flags & TQ_GRAPH_NODE_DEAD)
			continue;

		for (int slot = 0; slot < meta->tqPayloadCount; slot++)
		{
			if ((node->payloadMask & (uint16) (1U << slot)) == 0)
				continue;

			storage->payloadRefs[refIndex].payloadSlot = (int16) slot;
			storage->payloadRefs[refIndex].payloadValue = node->payloads[slot];
			storage->payloadRefs[refIndex].nodeId = nodeId;
			refIndex++;
		}
	}

	qsort(storage->payloadRefs, storage->payloadRefCount,
		  sizeof(TqGraphPayloadRef), TqGraphPayloadRefCompare);
}

bool
TqGraphPayloadRefRange(TqGraphScanStorage *storage, int payloadSlot,
					   int32 payloadValue, uint32 *firstIndex,
					   uint32 *refCount)
{
	uint32		lo = 0;
	uint32		hi = storage->payloadRefCount;
	uint32		first;

	if (payloadSlot < 0 || storage->payloadRefs == NULL ||
		storage->payloadRefCount == 0)
		return false;

	while (lo < hi)
	{
		uint32		mid = lo + (hi - lo) / 2;
		TqGraphPayloadRef *ref = &storage->payloadRefs[mid];

		if (ref->payloadSlot < payloadSlot ||
			(ref->payloadSlot == payloadSlot &&
			 ref->payloadValue < payloadValue))
			lo = mid + 1;
		else
			hi = mid;
	}

	first = lo;
	hi = storage->payloadRefCount;
	while (lo < hi)
	{
		uint32		mid = lo + (hi - lo) / 2;
		TqGraphPayloadRef *ref = &storage->payloadRefs[mid];

		if (ref->payloadSlot > payloadSlot ||
			(ref->payloadSlot == payloadSlot &&
			 ref->payloadValue > payloadValue))
			hi = mid;
		else
			lo = mid + 1;
	}

	if (first >= lo)
		return false;

	*firstIndex = first;
	*refCount = lo - first;
	return true;
}

static bool
TqGraphCacheMatches(TqGraphNativeCache *cache, Relation index,
					HnswMetaPageData *meta)
{
	return cache->relid == RelationGetRelid(index) &&
		cache->relfilenumber == TqGraphRelFileNumber(index) &&
		cache->dimensions == meta->dimensions &&
		cache->m == meta->m &&
		cache->graphMaxLevel == meta->graphMaxLevel &&
		cache->graphFlags == meta->graphFlags &&
		cache->tqNodeCount == meta->tqNodeCount &&
		cache->tqEntryNodeId == meta->tqEntryNodeId &&
		cache->tqCodeBytes == meta->tqCodeBytes &&
		cache->tqBits == meta->tqBits &&
		cache->tqPayloadCount == meta->tqPayloadCount &&
		cache->tqPayloadBytes == meta->tqPayloadBytes &&
		cache->tqCodeStartBlkno == meta->tqCodeStartBlkno &&
		cache->tqAdjStartBlkno == meta->tqAdjStartBlkno &&
		cache->tqExactStartBlkno == meta->tqExactStartBlkno &&
		cache->tqCorrectionStartBlkno == meta->tqCorrectionStartBlkno;
}

static void
TqGraphDropStaleCaches(Relation index, HnswMetaPageData *meta)
{
	TqGraphNativeCache **link = &tqGraphCacheList;

	while (*link != NULL)
	{
		TqGraphNativeCache *cache = *link;

		if (cache->relid == RelationGetRelid(index) &&
			!TqGraphCacheMatches(cache, index, meta))
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}

		link = &cache->next;
	}
}

void
TqGraphInvalidateCaches(Relation index)
{
	TqGraphNativeCache **link = &tqGraphCacheList;
	Oid			relid = RelationGetRelid(index);

	while (*link != NULL)
	{
		TqGraphNativeCache *cache = *link;

		if (cache->relid == relid)
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}

		link = &cache->next;
	}
}

static TqGraphNativeCache *
TqGraphFindCache(Relation index, HnswMetaPageData *meta)
{
	TqGraphDropStaleCaches(index, meta);

	for (TqGraphNativeCache *cache = tqGraphCacheList; cache != NULL; cache = cache->next)
	{
		if (TqGraphCacheMatches(cache, index, meta))
			return cache;
	}

	return NULL;
}

static TqGraphNativeCache *
TqGraphBuildCache(Relation index, HnswMetaPageData *meta)
{
	MemoryContext cacheCtx;
	MemoryContext oldCtx;
	TqGraphNativeCache *cache;

	cacheCtx = AllocSetContextCreate(CacheMemoryContext,
									 "turboquant native graph cache",
									 ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(cacheCtx);

	cache = palloc0(sizeof(TqGraphNativeCache));
	cache->relid = RelationGetRelid(index);
	cache->relfilenumber = TqGraphRelFileNumber(index);
	cache->dimensions = meta->dimensions;
	cache->m = meta->m;
	cache->graphMaxLevel = meta->graphMaxLevel;
	cache->graphFlags = meta->graphFlags;
	cache->tqNodeCount = meta->tqNodeCount;
	cache->tqEntryNodeId = meta->tqEntryNodeId;
	cache->tqCodeBytes = meta->tqCodeBytes;
	cache->tqBits = meta->tqBits;
	cache->tqPayloadCount = meta->tqPayloadCount;
	cache->tqPayloadBytes = meta->tqPayloadBytes;
	cache->tqCodeStartBlkno = meta->tqCodeStartBlkno;
	cache->tqAdjStartBlkno = meta->tqAdjStartBlkno;
	cache->tqExactStartBlkno = meta->tqExactStartBlkno;
	cache->tqCorrectionStartBlkno = meta->tqCorrectionStartBlkno;
	cache->ctx = cacheCtx;

	TqGraphInitScanStorageUncached(meta, &cache->storage,
								   TqGraphShouldCacheExactVectors(index, meta));
	cache->storage.ctx = cacheCtx;
	if (meta->tqNodeCount > 0)
	{
		cache->storage.visitedGeneration = palloc0(sizeof(uint32) * meta->tqNodeCount);
		cache->storage.visitGeneration = palloc0(sizeof(uint32));
	}

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount;
		 nodeId += cache->storage.codeTuplesPerPage)
		(void) TqGraphLoadCodePage(index, NULL, meta, &cache->storage, nodeId);

	TqGraphBuildPayloadRefs(meta, &cache->storage);

	if (meta->tqNodeCount > 0)
		TqGraphLoadAllAdjPages(index, NULL, meta, &cache->storage);

	if (cache->storage.exactArena != NULL)
		(void) TqGraphLoadExactVectors(index, meta, &cache->storage);
	cache->storage.cached = true;

	cache->next = tqGraphCacheList;
	tqGraphCacheList = cache;

	MemoryContextSwitchTo(oldCtx);
	return cache;
}

void
TqGraphInitScanStorage(Relation index, HnswMetaPageData *meta,
					   TqGraphScanStorage *storage)
{
	TqGraphNativeCache *cache;

	cache = TqGraphFindCache(index, meta);
	if (cache == NULL)
		cache = TqGraphBuildCache(index, meta);

	memcpy(storage, &cache->storage, sizeof(TqGraphScanStorage));
	storage->cached = true;
}

TqGraphNativeCache *
TqGraphInitInsertStorage(Relation index, HnswMetaPageData *meta,
						 TqGraphScanStorage *storage)
{
	TqGraphNativeCache *cache;

	cache = TqGraphFindCache(index, meta);
	if (cache == NULL)
		cache = TqGraphBuildCache(index, meta);

	memcpy(storage, &cache->storage, sizeof(TqGraphScanStorage));
	storage->cached = true;
	return cache;
}

static uint32 *
TqGraphCacheCopyNeighbors(TqGraphScanStorage *storage, uint32 *neighbors,
						  int count)
{
	uint32	   *copy;

	if (count <= 0)
		return NULL;

	copy = MemoryContextAlloc(storage->ctx, sizeof(uint32) * count);
	memcpy(copy, neighbors, sizeof(uint32) * count);
	return copy;
}

void
TqGraphAppendInsertCacheNode(TqGraphNativeCache *cache, HnswMetaPageData *meta,
							 uint32 nodeId, ItemPointer heapTid,
							 int nodeLevel, Vector *vector, uint8 *code,
							 float scale, float norm, float codeNorm,
							 float ecCorrection, int32 *payloads,
							 uint16 payloadMask, BlockNumber exactBlkno,
							 OffsetNumber exactOffno, uint32 **selected,
							 int *selectedCounts, BlockNumber *adjBlknos,
							 OffsetNumber *adjOffnos, BlockNumber codeStart,
							 BlockNumber adjStart, BlockNumber exactStart,
							 uint32 entryNodeId, uint16 graphMaxLevel)
{
	TqGraphScanStorage *storage;
	uint32		oldNodeCount;
	uint32		newNodeCount;
	int			oldAdjRecordCount;
	int			newAdjRecordCount;
	int			oldCodePageCount;
	int			newCodePageCount;
	Size		codeBytes;
	Size		payloadBytes;
	MemoryContext oldCtx;

	if (cache == NULL || nodeId != cache->tqNodeCount)
		return;

	storage = &cache->storage;
	oldNodeCount = cache->tqNodeCount;
	newNodeCount = oldNodeCount + 1;
	oldAdjRecordCount = oldNodeCount * TqGraphLevelCapacity(meta->m);
	newAdjRecordCount = newNodeCount * TqGraphLevelCapacity(meta->m);
	oldCodePageCount = storage->codePageCount;
	newCodePageCount = TqGraphPageCount(newNodeCount, storage->codeTuplesPerPage);
	codeBytes = meta->tqCodeBytes;
	payloadBytes = meta->tqPayloadBytes;

	oldCtx = MemoryContextSwitchTo(cache->ctx);

	storage->nodes = repalloc(storage->nodes,
							  sizeof(TqGraphScanNode) * newNodeCount);
	memset(&storage->nodes[nodeId], 0, sizeof(TqGraphScanNode));
	if (codeBytes > 0)
	{
		storage->codeArena = repalloc(storage->codeArena,
									  (Size) newNodeCount * codeBytes);
		memcpy(storage->codeArena + ((Size) nodeId * codeBytes),
			   code, codeBytes);
	}
	if (payloadBytes > 0)
	{
		storage->payloadArena = repalloc(storage->payloadArena,
										 (Size) newNodeCount * payloadBytes);
		if (payloads != NULL)
			memcpy(storage->payloadArena + ((Size) nodeId * payloadBytes),
				   payloads, payloadBytes);
		else
			memset(storage->payloadArena + ((Size) nodeId * payloadBytes),
				   0, payloadBytes);
	}
	if (storage->exactArena != NULL)
	{
		storage->exactArena = repalloc(storage->exactArena,
									   (Size) newNodeCount * storage->exactBytes);
		memcpy(storage->exactArena + ((Size) nodeId * storage->exactBytes),
			   vector, storage->exactBytes);
	}
	if (storage->visitedGeneration != NULL)
	{
		storage->visitedGeneration = repalloc(storage->visitedGeneration,
											  sizeof(uint32) * newNodeCount);
		storage->visitedGeneration[nodeId] = 0;
	}
	if (newCodePageCount != oldCodePageCount)
	{
		storage->codePagesLoaded = repalloc(storage->codePagesLoaded,
											sizeof(bool) * newCodePageCount);
		storage->codeBlknos = repalloc(storage->codeBlknos,
									   sizeof(BlockNumber) * newCodePageCount);
		for (int i = oldCodePageCount; i < newCodePageCount; i++)
		{
			storage->codePagesLoaded[i] = true;
			storage->codeBlknos[i] = InvalidBlockNumber;
		}
		storage->codePageCount = newCodePageCount;
	}

	storage->neighbors = repalloc(storage->neighbors,
								  sizeof(uint32 *) * newAdjRecordCount);
	storage->neighborCounts = repalloc(storage->neighborCounts,
									   sizeof(uint16) * newAdjRecordCount);
	storage->adjBlknos = repalloc(storage->adjBlknos,
								  sizeof(BlockNumber) * newAdjRecordCount);
	storage->adjOffnos = repalloc(storage->adjOffnos,
								  sizeof(OffsetNumber) * newAdjRecordCount);
	memset(&storage->neighbors[oldAdjRecordCount], 0,
		   sizeof(uint32 *) * (newAdjRecordCount - oldAdjRecordCount));
	memset(&storage->neighborCounts[oldAdjRecordCount], 0,
		   sizeof(uint16) * (newAdjRecordCount - oldAdjRecordCount));
	for (int i = oldAdjRecordCount; i < newAdjRecordCount; i++)
	{
		storage->adjBlknos[i] = InvalidBlockNumber;
		storage->adjOffnos[i] = InvalidOffsetNumber;
	}

	for (int level = 0; level < TqGraphLevelCapacity(meta->m); level++)
	{
		int			slot = TqGraphAdjSlot(meta, nodeId, level);
		int			count = level <= nodeLevel ? selectedCounts[level] : 0;

		storage->neighborCounts[slot] = count;
		if (level <= nodeLevel)
		{
			storage->adjBlknos[slot] = adjBlknos[level];
			storage->adjOffnos[slot] = adjOffnos[level];
		}
		storage->neighbors[slot] =
			TqGraphCacheCopyNeighbors(storage,
									  level <= nodeLevel ? selected[level] : NULL,
									  count);
	}

	storage->nodes[nodeId].heaptid = *heapTid;
	storage->nodes[nodeId].level = nodeLevel;
	storage->nodes[nodeId].exactBlkno = exactBlkno;
	storage->nodes[nodeId].exactOffno = exactOffno;
	storage->nodes[nodeId].payloadMask = payloadMask;
	storage->nodes[nodeId].scale = scale;
	storage->nodes[nodeId].norm = norm;
	storage->nodes[nodeId].codeNorm = codeNorm;
	storage->nodes[nodeId].ecCorrection = ecCorrection;
	storage->nodes[nodeId].flags = 0;
	storage->nodes[nodeId].loaded = true;
	if (codeBytes > 0)
		storage->nodes[nodeId].code = storage->codeArena + ((Size) nodeId * codeBytes);
	if (payloadBytes > 0)
		storage->nodes[nodeId].payloads =
			(int32 *) (storage->payloadArena + ((Size) nodeId * payloadBytes));
	if (storage->exactArena != NULL)
		storage->nodes[nodeId].exactVector =
			storage->exactArena + ((Size) nodeId * storage->exactBytes);

	if (payloadBytes > 0 && payloads != NULL && payloadMask != 0)
	{
		uint32		addedRefs = 0;

		for (int slot = 0; slot < meta->tqPayloadCount; slot++)
		{
			if (payloadMask & (uint16) (1U << slot))
				addedRefs++;
		}
		if (addedRefs > 0)
		{
			uint32		oldRefCount = storage->payloadRefCount;
			uint32		refIndex = oldRefCount;

			if (storage->payloadRefs == NULL)
				storage->payloadRefs =
					palloc(sizeof(TqGraphPayloadRef) * addedRefs);
			else
				storage->payloadRefs = repalloc(storage->payloadRefs,
												sizeof(TqGraphPayloadRef) *
												(oldRefCount + addedRefs));
			for (int slot = 0; slot < meta->tqPayloadCount; slot++)
			{
				if ((payloadMask & (uint16) (1U << slot)) == 0)
					continue;
				storage->payloadRefs[refIndex].payloadSlot = (int16) slot;
				storage->payloadRefs[refIndex].payloadValue = payloads[slot];
				storage->payloadRefs[refIndex].nodeId = nodeId;
				refIndex++;
			}
			storage->payloadRefCount = oldRefCount + addedRefs;
			qsort(storage->payloadRefs, storage->payloadRefCount,
				  sizeof(TqGraphPayloadRef), TqGraphPayloadRefCompare);
		}
	}

	cache->tqNodeCount = newNodeCount;
	cache->tqEntryNodeId = entryNodeId;
	cache->graphMaxLevel = graphMaxLevel;
	cache->tqCodeStartBlkno = codeStart;
	cache->tqAdjStartBlkno = adjStart;
	cache->tqExactStartBlkno = exactStart;

	MemoryContextSwitchTo(oldCtx);
}
