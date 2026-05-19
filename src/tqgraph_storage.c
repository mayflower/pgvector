#include "postgres.h"

#include "access/generic_xlog.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/rel.h"

#include "tqgraph.h"

Oid
TqGraphRelFileNumber(Relation index)
{
#if PG_VERSION_NUM >= 150000
	return index->rd_locator.relNumber;
#else
	return index->rd_node.relNode;
#endif
}

void
TqGraphInitBlockMap(BlockNumber *blknos, int count)
{
	for (int i = 0; i < count; i++)
		blknos[i] = InvalidBlockNumber;
}

bool
TqGraphEnsureBlockMap(Relation index, BlockNumber startBlkno, int pageCount,
					  uint16 pageKind, BlockNumber *blknos)
{
	BlockNumber blkno = startBlkno;

	if (pageCount <= 0)
		return true;

	if (BlockNumberIsValid(blknos[pageCount - 1]))
		return true;

	for (int pageNo = 0; pageNo < pageCount; pageNo++)
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;

		if (BlockNumberIsValid(blknos[pageNo]))
		{
			blkno = blknos[pageNo];
			continue;
		}

		if (!BlockNumberIsValid(blkno))
			return false;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);
		if ((opaque->pageKind & HNSW_PAGE_KIND_MASK) != pageKind)
		{
			UnlockReleaseBuffer(buf);
			return false;
		}

		blknos[pageNo] = blkno;
		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}

	return true;
}

BlockNumber
TqGraphGetChainBlockNumber(Relation index, BlockNumber startBlkno, int pageNo,
						   int pageCount, uint16 pageKind)
{
	BlockNumber *blknos;
	BlockNumber blkno;

	if (pageNo < 0 || pageNo >= pageCount)
		return InvalidBlockNumber;

	blknos = palloc(sizeof(BlockNumber) * pageCount);
	TqGraphInitBlockMap(blknos, pageCount);
	if (!TqGraphEnsureBlockMap(index, startBlkno, pageCount, pageKind, blknos))
		blkno = InvalidBlockNumber;
	else
		blkno = blknos[pageNo];
	pfree(blknos);

	return blkno;
}

BlockNumber
TqGraphGetMappedBlockNumber(BlockNumber startBlkno, int pageNo, BlockNumber *blknos)
{
	if (pageNo < 0)
		return InvalidBlockNumber;
	if (blknos != NULL && BlockNumberIsValid(blknos[pageNo]))
		return blknos[pageNo];
	if (!BlockNumberIsValid(startBlkno))
		return InvalidBlockNumber;

	return startBlkno + pageNo;
}

bool
TqGraphResolveChainBlockNumber(Relation index, BlockNumber startBlkno,
							   int pageNo, int pageCount, uint16 pageKind,
							   BlockNumber *blknos, BlockNumber *blkno)
{
	if (blkno == NULL || pageNo < 0 || pageNo >= pageCount)
		return false;

	if (blknos != NULL)
	{
		if (!BlockNumberIsValid(blknos[pageNo]) &&
			!TqGraphEnsureBlockMap(index, startBlkno, pageCount, pageKind, blknos))
			return false;

		*blkno = blknos[pageNo];
		return BlockNumberIsValid(*blkno);
	}

	*blkno = TqGraphGetChainBlockNumber(index, startBlkno, pageNo,
									 pageCount, pageKind);
	return BlockNumberIsValid(*blkno);
}
void
TqGraphFinishPage(Buffer buf)
{
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

void
TqGraphAppendPage(Relation index, ForkNumber forkNum, Buffer *buf, Page *page, uint16 pageKind)
{
	Buffer		newbuf;

	LockRelationForExtension(index, ExclusiveLock);
	newbuf = HnswNewBuffer(index, forkNum);
	UnlockRelationForExtension(index, ExclusiveLock);

	if (BufferIsValid(*buf))
	{
		HnswPageGetOpaque(*page)->nextblkno = BufferGetBlockNumber(newbuf);
		HnswMarkPageGraphOp(*page, HNSW_GRAPH_OP_PAGE_LINK);
		TqGraphFinishPage(*buf);
	}

	*buf = newbuf;
	*page = BufferGetPage(newbuf);
	HnswInitPageKind(newbuf, *page, pageKind);
}

OffsetNumber
TqGraphAppendTuple(Relation index, ForkNumber forkNum, BlockNumber *startBlkno,
				   uint16 pageKind, Item tuple, Size tupleSize,
				   uint16 graphOpKind, BlockNumber *insertBlkno)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber blkno = *startBlkno;
	GenericXLogState *xlogState = NULL;
	OffsetNumber offno;
	bool		createdStart = false;
	Buffer		linkbuf = InvalidBuffer;
	BlockNumber linkBlkno = InvalidBlockNumber;
	BlockNumber initBlkno = InvalidBlockNumber;

	if (!BlockNumberIsValid(blkno))
	{
		LockRelationForExtension(index, ExclusiveLock);
		buf = HnswNewBuffer(index, forkNum);
		UnlockRelationForExtension(index, ExclusiveLock);
		blkno = BufferGetBlockNumber(buf);
		*startBlkno = blkno;
		createdStart = true;
	}
	else
	{
		for (;;)
		{
			BlockNumber nextblkno;

			buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			nextblkno = HnswPageGetOpaque(page)->nextblkno;
			UnlockReleaseBuffer(buf);

			if (!BlockNumberIsValid(nextblkno))
				break;
			blkno = nextblkno;
		}

		buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
	}

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (RelationNeedsWAL(index) && forkNum == MAIN_FORKNUM)
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf,
										 createdStart ? GENERIC_XLOG_FULL_IMAGE : 0);
	}
	else
		page = BufferGetPage(buf);

	if (createdStart)
	{
		HnswInitPageKind(buf, page, pageKind);
		initBlkno = blkno;
	}
	else if ((HnswPageGetOpaque(page)->pageKind & HNSW_PAGE_KIND_MASK) != pageKind)
		elog(ERROR, "unexpected turboquant graph page kind while appending");

	if (PageGetFreeSpace(page) < tupleSize)
	{
		Buffer		newbuf;
		Page		newpage;

		LockRelationForExtension(index, ExclusiveLock);
		newbuf = HnswNewBuffer(index, forkNum);
		UnlockRelationForExtension(index, ExclusiveLock);

		if (xlogState != NULL)
			newpage = GenericXLogRegisterBuffer(xlogState, newbuf,
												GENERIC_XLOG_FULL_IMAGE);
		else
			newpage = BufferGetPage(newbuf);

		HnswInitPageKind(newbuf, newpage, pageKind);
		HnswPageGetOpaque(page)->nextblkno = BufferGetBlockNumber(newbuf);
		HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_PAGE_LINK);
		linkbuf = buf;
		linkBlkno = blkno;
		initBlkno = BufferGetBlockNumber(newbuf);
		blkno = initBlkno;
		buf = newbuf;
		page = newpage;
	}

	offno = PageAddItem(page, tuple, tupleSize, InvalidOffsetNumber, false, false);
	if (offno == InvalidOffsetNumber)
		elog(ERROR, "failed to append turboquant graph tuple");

	HnswMarkPageGraphOp(page, graphOpKind);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
	{
		if (BufferIsValid(linkbuf))
			MarkBufferDirty(linkbuf);
		MarkBufferDirty(buf);
	}

	UnlockReleaseBuffer(buf);
	if (BufferIsValid(linkbuf))
		UnlockReleaseBuffer(linkbuf);

	if (BlockNumberIsValid(linkBlkno))
		HnswLogGraphWalRecord(index, forkNum, linkBlkno, HNSW_GRAPH_OP_PAGE_LINK);
	if (BlockNumberIsValid(initBlkno))
		HnswLogGraphWalRecord(index, forkNum, initBlkno, HNSW_GRAPH_OP_PAGE_INIT);
	HnswLogGraphWalRecord(index, forkNum, blkno, graphOpKind);

	*insertBlkno = blkno;
	return offno;
}
