#include "postgres.h"

#include <string.h>

#include "access/generic_xlog.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/rel.h"

#include "tqgraph.h"

static TqGraphExactSlabPageHeader
TqGraphExactSlabHeader(Page page)
{
	ItemId		iid;

	if (PageGetMaxOffsetNumber(page) < FirstOffsetNumber)
		return NULL;

	iid = PageGetItemId(page, FirstOffsetNumber);
	if (!ItemIdIsValid(iid) || !ItemIdHasStorage(iid))
		return NULL;

	return (TqGraphExactSlabPageHeader) PageGetItem(page, iid);
}

static Size
TqGraphExactSlabCapacity(Page page)
{
	TqGraphExactSlabPageHeader header = TqGraphExactSlabHeader(page);

	if (header == NULL)
		return 0;

	return header->capacity;
}

static bool
TqGraphExactPageIsSlab(Page page)
{
	TqGraphExactSlabPageHeader header = TqGraphExactSlabHeader(page);

	return header != NULL && header->magic == TQ_GRAPH_EXACT_SLAB_MAGIC;
}

bool
TqGraphExactByteOffsetIsValid(OffsetNumber offno)
{
	return offno != InvalidOffsetNumber;
}

static void
TqGraphInitExactSlabPage(Page page)
{
	Size		tupleSize = PageGetFreeSpace(page);
	TqGraphExactSlabPageHeader tuple;

	tupleSize -= tupleSize % MAXIMUM_ALIGNOF;
	if (tupleSize <= offsetof(TqGraphExactSlabPageHeaderData, data))
		elog(ERROR, "turboquant graph exact slab page has no tuple capacity");

	tuple = palloc0(tupleSize);
	tuple->magic = TQ_GRAPH_EXACT_SLAB_MAGIC;
	tuple->capacity = tupleSize - offsetof(TqGraphExactSlabPageHeaderData, data);

	if (PageAddItem(page, (Item) tuple, tupleSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
		elog(ERROR, "failed to add turboquant graph exact slab item");

	pfree(tuple);
}


BlockNumber
TqGraphWriteExactPages(TqGraphBuildState *state)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	Size		vectorSize = VECTOR_SIZE(state->dimensions);

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		TqGraphBuildNode *node = &state->nodes[i];
		char	   *src = (char *) node->vector;
		Size		remaining = vectorSize;

		while (remaining > 0)
		{
			TqGraphExactSlabPageHeader header;
			Size		capacity;
			Size		available;
			Size		chunk;

			if (!BufferIsValid(buf))
			{
				TqGraphAppendPage(state->index, state->forkNum, &buf, &page, HNSW_PAGE_KIND_TQ_EXACT);
				TqGraphInitExactSlabPage(page);
				if (!BlockNumberIsValid(start))
					start = BufferGetBlockNumber(buf);
			}

			header = TqGraphExactSlabHeader(page);
			capacity = TqGraphExactSlabCapacity(page);
			if (header->used >= capacity)
			{
				TqGraphAppendPage(state->index, state->forkNum, &buf, &page, HNSW_PAGE_KIND_TQ_EXACT);
				TqGraphInitExactSlabPage(page);
				header = TqGraphExactSlabHeader(page);
				capacity = TqGraphExactSlabCapacity(page);
			}

			available = capacity - header->used;
			chunk = Min(remaining, available);
			if (chunk == 0)
				elog(ERROR, "turboquant graph exact slab page has no capacity");

			if (remaining == vectorSize)
			{
				node->exactBlkno = BufferGetBlockNumber(buf);
				node->exactOffno = (OffsetNumber) (header->used + 1);
			}

			memcpy(header->data + header->used, src, chunk);
			header->used += chunk;
			src += chunk;
			remaining -= chunk;
			HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_ELEMENT_INSERT);
		}
	}

	if (BufferIsValid(buf))
		TqGraphFinishPage(buf);

	return start;
}


bool
TqGraphReadExactVectorInto(Relation index, TqGraphScanNode *node, int dimensions,
						   char *dest, HnswScanOpaque so)
{
	Buffer		buf;
	Page		page;
	Size		vectorSize = VECTOR_SIZE(dimensions);
	Size		copied = 0;
	BlockNumber blkno;
	OffsetNumber offno;

	if (!BlockNumberIsValid(node->exactBlkno) ||
		!TqGraphExactByteOffsetIsValid(node->exactOffno))
		return false;

	if (node->exactVector != NULL)
	{
		memcpy(dest, node->exactVector, vectorSize);
		return true;
	}

	blkno = node->exactBlkno;
	offno = node->exactOffno;
	while (copied < vectorSize)
	{
		HnswPageOpaque opaque;

		if (!BlockNumberIsValid(blkno))
			return false;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);

		if ((opaque->pageKind & HNSW_PAGE_KIND_MASK) != HNSW_PAGE_KIND_TQ_EXACT)
		{
			UnlockReleaseBuffer(buf);
			return false;
		}

		if (TqGraphExactPageIsSlab(page))
		{
			TqGraphExactSlabPageHeader header = TqGraphExactSlabHeader(page);
			Size		offset = (Size) offno - 1;
			Size		available;
			Size		chunk;

			if (offset >= header->used)
			{
				UnlockReleaseBuffer(buf);
				return false;
			}

			available = header->used - offset;
			chunk = Min(vectorSize - copied, available);
			memcpy(dest + copied, header->data + offset, chunk);
			copied += chunk;
			blkno = opaque->nextblkno;
			offno = FirstOffsetNumber;
			if (so != NULL)
				so->graphRescorePages++;
			UnlockReleaseBuffer(buf);
			continue;
		}
		else
		{
			ItemId		iid;
			TqGraphExactTuple tuple;

			if (node->exactOffno > PageGetMaxOffsetNumber(page))
			{
				UnlockReleaseBuffer(buf);
				return false;
			}

			iid = PageGetItemId(page, node->exactOffno);
			tuple = (TqGraphExactTuple) PageGetItem(page, iid);
			if (tuple->type == TQ_GRAPH_EXACT_TUPLE_TYPE)
			{
				memcpy(dest, tuple->vector, vectorSize);
				if (so != NULL)
					so->graphRescorePages++;
				UnlockReleaseBuffer(buf);
				return true;
			}
		}

		UnlockReleaseBuffer(buf);
		return false;
	}

	return true;
}

Vector *
TqGraphReadExactVector(Relation index, TqGraphScanNode *node, int dimensions)
{
	Vector	   *vector;
	Size		vectorSize = VECTOR_SIZE(dimensions);

	vector = palloc(vectorSize);
	if (!TqGraphReadExactVectorInto(index, node, dimensions, (char *) vector, NULL))
	{
		pfree(vector);
		return NULL;
	}

	return vector;
}


void
TqGraphAppendInsertedExact(Relation index, BlockNumber *exactStart,
						   uint32 nodeId, Vector *vector, int dimensions,
						   BlockNumber *exactBlkno, OffsetNumber *exactOffno)
{
	char	   *src = (char *) vector;
	Size		remaining = VECTOR_SIZE(dimensions);
	BlockNumber blkno = *exactStart;

	(void) nodeId;

	*exactBlkno = InvalidBlockNumber;
	*exactOffno = InvalidOffsetNumber;

	while (remaining > 0)
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		TqGraphExactSlabPageHeader header;
		Size		capacity;
		Size		available;
		Size		chunk;
		GenericXLogState *xlogState = NULL;
		bool		createdStart = false;
		BlockNumber initBlkno = InvalidBlockNumber;
		BlockNumber linkBlkno = InvalidBlockNumber;

		if (!BlockNumberIsValid(blkno))
		{
			LockRelationForExtension(index, ExclusiveLock);
			buf = HnswNewBuffer(index, MAIN_FORKNUM);
			UnlockRelationForExtension(index, ExclusiveLock);
			blkno = BufferGetBlockNumber(buf);
			*exactStart = blkno;
			createdStart = true;
		}
		else
		{
			for (;;)
			{
				BlockNumber nextblkno;

				buf = ReadBuffer(index, blkno);
				LockBuffer(buf, BUFFER_LOCK_SHARE);
				page = BufferGetPage(buf);
				nextblkno = HnswPageGetOpaque(page)->nextblkno;
				UnlockReleaseBuffer(buf);
				if (!BlockNumberIsValid(nextblkno))
					break;
				blkno = nextblkno;
			}

			buf = ReadBuffer(index, blkno);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		}

		if (RelationNeedsWAL(index))
		{
			xlogState = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(xlogState, buf,
											 createdStart ? GENERIC_XLOG_FULL_IMAGE : 0);
		}
		else
			page = BufferGetPage(buf);

		if (createdStart)
		{
			HnswInitPageKind(buf, page, HNSW_PAGE_KIND_TQ_EXACT);
			TqGraphInitExactSlabPage(page);
			initBlkno = blkno;
		}

		opaque = HnswPageGetOpaque(page);
		if ((opaque->pageKind & HNSW_PAGE_KIND_MASK) != HNSW_PAGE_KIND_TQ_EXACT)
			elog(ERROR, "unexpected turboquant graph exact page kind while appending");
		if (!TqGraphExactPageIsSlab(page))
			elog(ERROR, "cannot append to legacy turboquant graph exact tuple page");

		header = TqGraphExactSlabHeader(page);
		capacity = TqGraphExactSlabCapacity(page);
		available = header->used < capacity ? capacity - header->used : 0;

		if (available == 0)
		{
			Buffer		newbuf;
			Page		newpage;
			BlockNumber newblkno;

			LockRelationForExtension(index, ExclusiveLock);
			newbuf = HnswNewBuffer(index, MAIN_FORKNUM);
			UnlockRelationForExtension(index, ExclusiveLock);
			newblkno = BufferGetBlockNumber(newbuf);

			if (xlogState != NULL)
				newpage = GenericXLogRegisterBuffer(xlogState, newbuf,
													GENERIC_XLOG_FULL_IMAGE);
			else
				newpage = BufferGetPage(newbuf);

			HnswInitPageKind(newbuf, newpage, HNSW_PAGE_KIND_TQ_EXACT);
			TqGraphInitExactSlabPage(newpage);
			opaque->nextblkno = newblkno;
			HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_PAGE_LINK);
			linkBlkno = blkno;
			initBlkno = newblkno;

			if (xlogState != NULL)
				GenericXLogFinish(xlogState);
			else
			{
				MarkBufferDirty(buf);
				MarkBufferDirty(newbuf);
			}

			UnlockReleaseBuffer(buf);
			UnlockReleaseBuffer(newbuf);
			HnswLogGraphWalRecord(index, MAIN_FORKNUM, linkBlkno,
								   HNSW_GRAPH_OP_PAGE_LINK);
			HnswLogGraphWalRecord(index, MAIN_FORKNUM, initBlkno,
								   HNSW_GRAPH_OP_PAGE_INIT);
			blkno = newblkno;
			continue;
		}

		chunk = Min(remaining, available);
		if (!BlockNumberIsValid(*exactBlkno))
		{
			*exactBlkno = blkno;
			*exactOffno = (OffsetNumber) (header->used + 1);
		}

		memcpy(header->data + header->used, src, chunk);
		header->used += chunk;
		src += chunk;
		remaining -= chunk;
		HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_ELEMENT_INSERT);

		if (xlogState != NULL)
			GenericXLogFinish(xlogState);
		else
			MarkBufferDirty(buf);

		UnlockReleaseBuffer(buf);
		if (BlockNumberIsValid(initBlkno))
			HnswLogGraphWalRecord(index, MAIN_FORKNUM, initBlkno,
								   HNSW_GRAPH_OP_PAGE_INIT);
		HnswLogGraphWalRecord(index, MAIN_FORKNUM, blkno,
							   HNSW_GRAPH_OP_ELEMENT_INSERT);
	}
}
