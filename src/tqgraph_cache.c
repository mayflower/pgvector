#include "postgres.h"

#include <float.h>
#include <string.h>

#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "tqgraph.h"

static TqGraphCorrectionCache *tqGraphCorrectionCacheList = NULL;

bool
TqGraphLoadCorrection(Relation index, int dimensions, float **ecShift, float **ecScale)
{
	HnswMetaPageData meta;
	BlockNumber blkno;
	BlockNumber nblocks;
	bool		haveShift = false;
	bool		haveScale = false;
	int			hops = 0;
	TqGraphCorrectionCache *cache;

	*ecShift = NULL;
	*ecScale = NULL;

	if (!TqGraphReadMeta(index, &meta) ||
		(meta.tqFlags & TQ_GRAPH_TQ_PLUS) == 0 ||
		meta.dimensions != (uint32) dimensions ||
		!BlockNumberIsValid(meta.tqCorrectionStartBlkno))
		return false;

	for (cache = tqGraphCorrectionCacheList; cache != NULL; cache = cache->next)
	{
		if (cache->relid == RelationGetRelid(index) &&
			cache->relfilenumber == TqGraphRelFileNumber(index) &&
			cache->dimensions == meta.dimensions &&
			cache->tqFlags == meta.tqFlags &&
			cache->tqCorrectionStartBlkno == meta.tqCorrectionStartBlkno)
		{
			*ecShift = palloc(sizeof(float) * dimensions);
			*ecScale = palloc(sizeof(float) * dimensions);
			memcpy(*ecShift, cache->ecShift, sizeof(float) * dimensions);
			memcpy(*ecScale, cache->ecScale, sizeof(float) * dimensions);
			return true;
		}
	}

	*ecShift = palloc(sizeof(float) * dimensions);
	*ecScale = palloc(sizeof(float) * dimensions);
	blkno = meta.tqCorrectionStartBlkno;
	nblocks = RelationGetNumberOfBlocks(index);

	while (BlockNumberIsValid(blkno) && blkno < nblocks && hops++ < (int) nblocks)
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);

		if ((opaque->pageKind & HNSW_PAGE_KIND_MASK) != HNSW_PAGE_KIND_TQ_CORRECTION)
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			TqGraphCorrectionTuple tuple = (TqGraphCorrectionTuple) PageGetItem(page, iid);

			if (tuple->type != TQ_GRAPH_CORRECTION_TUPLE_TYPE ||
				tuple->startDim != 0 ||
				tuple->count != dimensions)
				continue;

			if (tuple->field == 0)
			{
				memcpy(*ecShift, tuple->values, sizeof(float) * dimensions);
				haveShift = true;
			}
			else if (tuple->field == 1)
			{
				memcpy(*ecScale, tuple->values, sizeof(float) * dimensions);
				haveScale = true;
			}
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);

		if (haveShift && haveScale)
		{
			MemoryContext oldCtx;

			cache = MemoryContextAllocZero(CacheMemoryContext,
										   sizeof(TqGraphCorrectionCache));
			cache->ctx = AllocSetContextCreate(CacheMemoryContext,
											   "TurboQuant graph correction cache",
											   ALLOCSET_SMALL_SIZES);
			cache->relid = RelationGetRelid(index);
			cache->relfilenumber = TqGraphRelFileNumber(index);
			cache->dimensions = meta.dimensions;
			cache->tqFlags = meta.tqFlags;
			cache->tqCorrectionStartBlkno = meta.tqCorrectionStartBlkno;
			oldCtx = MemoryContextSwitchTo(cache->ctx);
			cache->ecShift = palloc(sizeof(float) * dimensions);
			cache->ecScale = palloc(sizeof(float) * dimensions);
			memcpy(cache->ecShift, *ecShift, sizeof(float) * dimensions);
			memcpy(cache->ecScale, *ecScale, sizeof(float) * dimensions);
			MemoryContextSwitchTo(oldCtx);
			cache->next = tqGraphCorrectionCacheList;
			tqGraphCorrectionCacheList = cache;
			return true;
		}
	}

	pfree(*ecShift);
	pfree(*ecScale);
	*ecShift = NULL;
	*ecScale = NULL;
	return false;
}
