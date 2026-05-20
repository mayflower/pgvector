#include "postgres.h"

#include <math.h>
#include <string.h>

#include "access/generic_xlog.h"
#include "access/genam.h"
#include "access/tableam.h"
#include "catalog/pg_type_d.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/buffile.h"
#include "storage/lmgr.h"
#include "tsearch/ts_type.h"
#include "utils/builtins.h"
#include "utils/hsearch.h"
#include "utils/jsonb.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "hnsw.h"
#include "tqgraph.h"
#include "tqhybrid.h"
#include "tqhybrid_bm25.h"

typedef struct TqHybridTidNode
{
	ItemPointerData tid;
	uint32		nodeId;
} TqHybridTidNode;

typedef struct TqHybridNodeState
{
	ItemPointerData tid;
	bool		live;
} TqHybridNodeState;

typedef struct TqHybridBm25Collector
{
	Relation	index;
	TqHybridTidNode *tidNodes;
	uint32		tidNodeCount;
	TqHybridBm25BuildDoc *docs;
	uint32		docCount;
	uint32		docCapacity;
	uint32	   *denseDocLens;
	uint32		denseDocLensCount;
	TqHybridBm25TermTuple *terms;
	uint32		termCount;
	uint32		termCapacity;
	uint32		totalTermCount;
	char	   *termBytes;
	uint32		termBytesUsed;
	uint32		termBytesCapacity;
	uint64		totalDocLen;
	uint32		maxDocLen;
	uint32		uniqueTerms;
	BlockNumber bm25MetaBlkno;
	BlockNumber bm25DocStatsStartBlkno;
	BlockNumber bm25LexiconStartBlkno;
	BlockNumber bm25PostingsStartBlkno;
	BlockNumber bm25BlockMaxStartBlkno;
	uint32		bm25PostingsPages;
	uint32		bm25BlockMaxPages;
	Size		softBudget;
	bool		allowSpill;
	bool		walLoggedWrites;
	struct TqHybridBm25SpillRun *spillRuns;
	uint32		spillRunCount;
	uint32		spillRunCapacity;
} TqHybridBm25Collector;

typedef struct TqHybridBm25SpillRun
{
	BufFile    *file;
	uint32		tupleCount;
} TqHybridBm25SpillRun;

typedef struct TqHybridBm25SpillCursor
{
	TqHybridBm25SpillRun *run;
	uint32		remaining;
	bool		valid;
	uint64		termHash;
	uint32		nodeId;
	uint16		tf;
	uint16		termLen;
	char	   *termBytes;
} TqHybridBm25SpillCursor;

typedef struct TqHybridBm25DebugEntry
{
	Oid			indexOid;
	TqHybridBm25BuildStats stats;
} TqHybridBm25DebugEntry;

static HTAB *tqhybrid_bm25_debug = NULL;
static Oid	tqhybrid_bm25_delta_cursor_index = InvalidOid;
static Oid	tqhybrid_bm25_delta_cursor_relfilenumber = InvalidOid;
static BlockNumber tqhybrid_bm25_delta_cursor_start = InvalidBlockNumber;
static BlockNumber tqhybrid_bm25_delta_cursor_tail = InvalidBlockNumber;
static uint64 tqhybrid_bm25_delta_cursor_generation = 0;
static uint32 tqhybrid_bm25_delta_cursor_pages = 0;

static bool TqHybridBm25PageIsKind(Page page, uint16 pageKind);
static bool TqHybridBm25ReadMeta(Relation index, TqHybridBm25MetaTupleData *meta,
								 BlockNumber *metaBlkno);
static bool TqHybridBm25ReadMetaForUpdate(Relation index, Buffer *outBuf,
										  Page *outPage,
										  TqHybridBm25MetaTuple *outTuple,
										  GenericXLogState **outXlogState);
static void TqHybridBm25SetMetaBlock(Relation index, BlockNumber metaBlkno);
static void TqHybridBm25EnsureWalTail(Relation index, ForkNumber forkNum,
									  Buffer *buf, Page *page,
									  BlockNumber *startBlkno,
									  uint16 pageKind);
static OffsetNumber TqHybridBm25AddWalItem(Relation index, ForkNumber forkNum,
										   Buffer *buf, Page *page,
										   BlockNumber *startBlkno,
										   uint16 pageKind, Item item,
										   Size itemSize, uint32 *pageCount,
										   BlockNumber *insertBlkno);
static uint32 TqHybridBm25CountChainPagesAndTail(Relation index,
												 BlockNumber startBlkno,
												 uint16 pageKind,
												 BlockNumber *tailBlkno);
static void TqHybridStoreDebugStats(Relation index,
									TqHybridBm25Collector *collector,
									uint32 uniqueTerms);

static Size
TqHybridBm25DocStatsTupleSize(uint16 count)
{
	return MAXALIGN(offsetof(TqHybridBm25DocStatsTupleData, docs) +
					sizeof(TqBm25DocStat) * count);
}

static Size
TqHybridBm25PostingsTupleSize(Size payloadBytes)
{
	return MAXALIGN(offsetof(TqHybridBm25PostingsTupleData, payload) +
					payloadBytes);
}

static Size
TqHybridBm25LegacyPostingsTupleSize(uint16 count)
{
	return MAXALIGN(offsetof(TqHybridBm25PostingsTupleData, encoding) +
					sizeof(TqHybridBm25Posting) * count);
}

static Size
TqHybridBm25VarintLen(uint32 value)
{
	Size		len = 1;

	while (value >= 0x80)
	{
		value >>= 7;
		len++;
	}
	return len;
}

static void
TqHybridBm25PutVarint(char **ptr, uint32 value)
{
	while (value >= 0x80)
	{
		*(*ptr)++ = (char) ((value & 0x7f) | 0x80);
		value >>= 7;
	}
	*(*ptr)++ = (char) value;
}

static void
TqHybridBm25PutUint16(char **ptr, uint16 value)
{
	memcpy(*ptr, &value, sizeof(value));
	*ptr += sizeof(value);
}

static bool
TqHybridBm25CanUseDelta16(uint32 prevNodeId, uint32 nodeId)
{
	return nodeId >= prevNodeId && (nodeId - prevNodeId) <= PG_UINT16_MAX;
}

static bool
TqHybridBm25CanUseOffset16(uint32 firstNodeId, uint32 nodeId)
{
	return nodeId >= firstNodeId && (nodeId - firstNodeId) <= PG_UINT16_MAX;
}

static bool
TqHybridBm25PrecomputeTfNormEnabled(TqHybridBm25Collector *collector)
{
	TqHybridOptions *opts = (TqHybridOptions *) collector->index->rd_options;

	return opts != NULL && opts->bm25PrecomputeTfNorm;
}

static uint16
TqHybridBm25QuantizeTfNorm(TqHybridBm25Collector *collector, uint32 nodeId,
						   uint16 tf)
{
	TqHybridOptions *opts = (TqHybridOptions *) collector->index->rd_options;
	double		k1 = opts != NULL ? opts->bm25K1 : 1.2;
	double		b = opts != NULL ? opts->bm25B : 0.75;
	double		avgDocLen = collector->docCount == 0 ? 1.0 :
		(double) collector->totalDocLen / (double) collector->docCount;
	double		docLen = 1.0;
	double		norm;
	double		tfNorm;
	double		scale;
	long		quantized;

	if (collector->denseDocLens != NULL && nodeId < collector->denseDocLensCount &&
		collector->denseDocLens[nodeId] > 0)
		docLen = (double) collector->denseDocLens[nodeId];
	norm = k1 * (1.0 - b + b * docLen / Max(avgDocLen, 1.0));
	tfNorm = ((double) tf * (k1 + 1.0)) / ((double) tf + norm);
	scale = Max(k1 + 1.0, 1.0);
	quantized = lround((tfNorm / scale) * (double) PG_UINT16_MAX);
	if (quantized < 0)
		return 0;
	if (quantized > PG_UINT16_MAX)
		return PG_UINT16_MAX;
	return (uint16) quantized;
}

static bool
TqHybridBm25GetVarint(const char **ptr, const char *end, uint32 *value)
{
	uint32		result = 0;
	uint32		shift = 0;

	while (*ptr < end && shift <= 28)
	{
		unsigned char byte = (unsigned char) *(*ptr)++;

		result |= (uint32) (byte & 0x7f) << shift;
		if ((byte & 0x80) == 0)
		{
			*value = result;
			return true;
		}
		shift += 7;
	}
	return false;
}

static void
TqHybridBm25EncodePosting(char **ptr, uint32 *prevNodeId,
						  uint32 nodeId, uint16 tf)
{
	uint32		delta;

	if (nodeId < *prevNodeId)
		elog(ERROR, "turbohybrid BM25 postings are not sorted by node id");
	delta = nodeId - *prevNodeId;
	TqHybridBm25PutVarint(ptr, delta);
	TqHybridBm25PutVarint(ptr, tf);
	*prevNodeId = nodeId;
}

static void
TqHybridBm25EncodePostingDelta16(char **ptr, uint32 *prevNodeId,
								 uint32 nodeId, uint16 tf)
{
	uint16		delta;

	if (!TqHybridBm25CanUseDelta16(*prevNodeId, nodeId))
		elog(ERROR, "turbohybrid BM25 postings are not delta16 encodable");
	delta = (uint16) (nodeId - *prevNodeId);
	TqHybridBm25PutUint16(ptr, delta);
	TqHybridBm25PutUint16(ptr, tf);
	*prevNodeId = nodeId;
}

static void
TqHybridBm25EncodePostingOffset16(char **ptr, uint32 firstNodeId,
								  uint32 nodeId, uint16 tf)
{
	uint16		offset;

	if (!TqHybridBm25CanUseOffset16(firstNodeId, nodeId))
		elog(ERROR, "turbohybrid BM25 postings are not offset16 encodable");
	offset = (uint16) (nodeId - firstNodeId);
	TqHybridBm25PutUint16(ptr, offset);
	TqHybridBm25PutUint16(ptr, tf);
}

static bool
TqHybridBm25DecodePostingsTuple(TqHybridBm25PostingsTuple tuple,
								Size itemSize,
								TqHybridBm25Posting *postings)
{
	const char *ptr = tuple->payload;
	const char *end = tuple->payload + tuple->payloadBytes;
	uint32		prevNodeId = 0;
	uint16		encoding = tuple->encoding & TQHYBRID_BM25_POSTINGS_ENCODING_MASK;
	bool		hasTfNorm = (tuple->encoding & TQHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16) != 0;
	Size		tfNormBytes = hasTfNorm ? tuple->count * sizeof(uint16) : 0;
	const char *dataEnd = end;

	if (TqHybridBm25PostingsTupleSize(tuple->payloadBytes) != itemSize)
	{
		if (TqHybridBm25LegacyPostingsTupleSize(tuple->count) == itemSize)
		{
			memcpy(postings, &tuple->encoding,
				   sizeof(TqHybridBm25Posting) * tuple->count);
			return true;
		}
		return false;
	}
	if (tfNormBytes > (Size) (end - ptr))
		return false;
	if (hasTfNorm)
		dataEnd = end - tfNormBytes;

	if (encoding == TQHYBRID_BM25_POSTINGS_ENCODING_OFFSET16)
	{
		uint32		prevDecoded = tuple->firstNodeId;

		if ((Size) (dataEnd - ptr) != tuple->count * sizeof(uint16) * 2)
			return false;
		for (uint16 i = 0; i < tuple->count; i++)
		{
			uint16		offset;
			uint16		tf;
			uint32		nodeId;

			memcpy(&offset, ptr, sizeof(offset));
			ptr += sizeof(offset);
			memcpy(&tf, ptr, sizeof(tf));
			ptr += sizeof(tf);
			if (i == 0 && offset != 0)
				return false;
			nodeId = tuple->firstNodeId + offset;
			if (nodeId < prevDecoded)
				return false;
			prevDecoded = nodeId;
			postings[i].nodeId = nodeId;
			postings[i].tf = tf;
			postings[i].reserved = 0;
		}
		if (ptr != dataEnd)
			return false;
		if (hasTfNorm)
		{
			for (uint16 i = 0; i < tuple->count; i++)
			{
				memcpy(&postings[i].reserved, ptr, sizeof(postings[i].reserved));
				ptr += sizeof(postings[i].reserved);
			}
		}
		return ptr == end;
	}

	if (encoding == TQHYBRID_BM25_POSTINGS_ENCODING_DELTA16)
	{
		if ((Size) (dataEnd - ptr) != tuple->count * sizeof(uint16) * 2)
			return false;
		prevNodeId = tuple->firstNodeId;
		for (uint16 i = 0; i < tuple->count; i++)
		{
			uint16		delta;
			uint16		tf;

			memcpy(&delta, ptr, sizeof(delta));
			ptr += sizeof(delta);
			memcpy(&tf, ptr, sizeof(tf));
			ptr += sizeof(tf);
			if (i == 0 && delta != 0)
				return false;
			if ((PG_UINT32_MAX - prevNodeId) < delta)
				return false;
			prevNodeId += delta;
			postings[i].nodeId = prevNodeId;
			postings[i].tf = tf;
			postings[i].reserved = 0;
		}
		if (ptr != dataEnd)
			return false;
		if (hasTfNorm)
		{
			for (uint16 i = 0; i < tuple->count; i++)
			{
				memcpy(&postings[i].reserved, ptr, sizeof(postings[i].reserved));
				ptr += sizeof(postings[i].reserved);
			}
		}
		return ptr == end;
	}

	if (encoding != TQHYBRID_BM25_POSTINGS_ENCODING_DELTA_VARINT)
		return false;

	prevNodeId = tuple->firstNodeId;
	for (uint16 i = 0; i < tuple->count; i++)
	{
		uint32		delta;
		uint32		tf;

		if (!TqHybridBm25GetVarint(&ptr, dataEnd, &delta) ||
			!TqHybridBm25GetVarint(&ptr, dataEnd, &tf) ||
			tf > PG_UINT16_MAX ||
			(PG_UINT32_MAX - prevNodeId) < delta)
			return false;

		if (i == 0 && delta != 0)
			return false;
		prevNodeId += delta;
		postings[i].nodeId = prevNodeId;
		postings[i].tf = (uint16) tf;
		postings[i].reserved = 0;
	}
	if (ptr != dataEnd)
		return false;
	if (hasTfNorm)
	{
		for (uint16 i = 0; i < tuple->count; i++)
		{
			memcpy(&postings[i].reserved, ptr, sizeof(postings[i].reserved));
			ptr += sizeof(postings[i].reserved);
		}
	}

	return ptr == end;
}

static uint16
TqHybridBm25MaxPostingsPerChunk(void)
{
	Size		maxItemSize = (BLCKSZ / 2) - SizeOfPageHeaderData -
		MAXALIGN(sizeof(HnswPageOpaqueData));
	uint32		maxCount = (maxItemSize -
							offsetof(TqHybridBm25PostingsTupleData, payload)) /
		(TqHybridBm25VarintLen(PG_UINT32_MAX) +
		 TqHybridBm25VarintLen(PG_UINT16_MAX));

	maxCount = Max(maxCount, 1);
	if (tqhybrid_debug_postings_chunk_size > 0)
		maxCount = Min(maxCount, (uint32) tqhybrid_debug_postings_chunk_size);
	return (uint16) Min(maxCount, (uint32) PG_UINT16_MAX);
}

static Size
TqHybridBm25LexiconEntrySize(uint16 termLen)
{
	return MAXALIGN(offsetof(TqHybridBm25LexiconEntryData, termBytes) + termLen);
}

static Size
TqHybridBm25DeltaTupleSize(uint16 termCount, uint32 termBytesLen)
{
	return MAXALIGN(offsetof(TqHybridBm25DeltaTupleData, terms) +
					sizeof(TqHybridBm25DeltaTerm) * termCount +
					termBytesLen);
}

static int
TqHybridTidNodeCompare(const void *a, const void *b)
{
	const TqHybridTidNode *ta = (const TqHybridTidNode *) a;
	const TqHybridTidNode *tb = (const TqHybridTidNode *) b;
	int			blockCmp;

	blockCmp = (ItemPointerGetBlockNumber(&ta->tid) > ItemPointerGetBlockNumber(&tb->tid)) -
		(ItemPointerGetBlockNumber(&ta->tid) < ItemPointerGetBlockNumber(&tb->tid));
	if (blockCmp != 0)
		return blockCmp;

	return (ItemPointerGetOffsetNumber(&ta->tid) > ItemPointerGetOffsetNumber(&tb->tid)) -
		(ItemPointerGetOffsetNumber(&ta->tid) < ItemPointerGetOffsetNumber(&tb->tid));
}

static int
TqHybridTermCompareWithBytes(const TqHybridBm25TermTuple *ta,
							 const TqHybridBm25TermTuple *tb,
							 const char *termBytes)
{
	int			cmp;

	if (ta->termHash != tb->termHash)
		return ta->termHash < tb->termHash ? -1 : 1;
	if (ta->termLen != tb->termLen)
		return ta->termLen < tb->termLen ? -1 : 1;

	cmp = memcmp(termBytes + ta->termOffset, termBytes + tb->termOffset,
				 ta->termLen);
	if (cmp != 0)
		return cmp;

	return (ta->nodeId > tb->nodeId) - (ta->nodeId < tb->nodeId);
}

static bool
TqHybridTermEqualIgnoringNode(const TqHybridBm25TermTuple *ta,
							  const TqHybridBm25TermTuple *tb,
							  const char *termBytes)
{
	if (ta->termHash != tb->termHash || ta->termLen != tb->termLen)
		return false;

	return memcmp(termBytes + ta->termOffset, termBytes + tb->termOffset,
				  ta->termLen) == 0;
}

static TqHybridBm25Collector *tqhybrid_active_sort_collector = NULL;

static int
TqHybridTermCompare(const void *a, const void *b)
{
	return TqHybridTermCompareWithBytes((const TqHybridBm25TermTuple *) a,
										(const TqHybridBm25TermTuple *) b,
										tqhybrid_active_sort_collector->termBytes);
}

static uint64
TqHybridHashTerm(const char *term, uint16 len)
{
	uint64		hash = UINT64CONST(1469598103934665603);

	for (uint16 i = 0; i < len; i++)
	{
		hash ^= (unsigned char) term[i];
		hash *= UINT64CONST(1099511628211);
	}

	return hash;
}

static void TqHybridSpillTermRun(TqHybridBm25Collector *collector);

static void
TqHybridCheckBudget(TqHybridBm25Collector *collector)
{
	Size		used = (Size) collector->docCapacity * sizeof(TqHybridBm25BuildDoc) +
		(Size) collector->termCount * sizeof(TqHybridBm25TermTuple) +
		collector->termBytesUsed +
		(Size) collector->tidNodeCount * sizeof(TqHybridTidNode);

	if (used > collector->softBudget)
	{
		if (collector->allowSpill && collector->termCount > 0)
		{
			TqHybridSpillTermRun(collector);
			return;
		}
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid BM25 build collector exceeded maintenance_work_mem"),
				 errdetail("Doc metadata and native graph TID map require %zu bytes before BM25 term spill can help.",
						   used),
				 errhint("Increase maintenance_work_mem for this build.")));
	}
}

static void
TqHybridEnsureSpillRunCapacity(TqHybridBm25Collector *collector)
{
	if (collector->spillRunCount < collector->spillRunCapacity)
		return;

	collector->spillRunCapacity = collector->spillRunCapacity == 0 ? 8 :
		collector->spillRunCapacity * 2;
	if (collector->spillRuns == NULL)
		collector->spillRuns = palloc(sizeof(TqHybridBm25SpillRun) *
									   collector->spillRunCapacity);
	else
		collector->spillRuns = repalloc(collector->spillRuns,
										sizeof(TqHybridBm25SpillRun) *
										collector->spillRunCapacity);
}

static void
TqHybridSortTermRun(TqHybridBm25Collector *collector)
{
	if (collector->termCount <= 1)
		return;

	tqhybrid_active_sort_collector = collector;
	qsort(collector->terms, collector->termCount,
		  sizeof(TqHybridBm25TermTuple), TqHybridTermCompare);
	tqhybrid_active_sort_collector = NULL;
}

static void
TqHybridSpillTermRun(TqHybridBm25Collector *collector)
{
	TqHybridBm25SpillRun *run;

	if (collector->termCount == 0)
		return;

	TqHybridSortTermRun(collector);
	TqHybridEnsureSpillRunCapacity(collector);
	run = &collector->spillRuns[collector->spillRunCount++];
	run->file = BufFileCreateTemp(false);
	run->tupleCount = collector->termCount;

	for (uint32 i = 0; i < collector->termCount; i++)
	{
		TqHybridBm25TermTuple *term = &collector->terms[i];
		char	   *bytes = collector->termBytes + term->termOffset;

		BufFileWrite(run->file, &term->termHash, sizeof(term->termHash));
		BufFileWrite(run->file, &term->nodeId, sizeof(term->nodeId));
		BufFileWrite(run->file, &term->tf, sizeof(term->tf));
		BufFileWrite(run->file, &term->termLen, sizeof(term->termLen));
		BufFileWrite(run->file, bytes, term->termLen);
	}

	if (BufFileSeek(run->file, 0, 0L, SEEK_SET) != 0)
		elog(ERROR, "failed to rewind turbohybrid BM25 spill run");

	collector->termCount = 0;
	collector->termBytesUsed = 0;
}

static void
TqHybridCloseSpillRuns(TqHybridBm25Collector *collector)
{
	for (uint32 i = 0; i < collector->spillRunCount; i++)
	{
		if (collector->spillRuns[i].file != NULL)
		{
			BufFileClose(collector->spillRuns[i].file);
			collector->spillRuns[i].file = NULL;
		}
	}
}

static void
TqHybridEnsureDocCapacity(TqHybridBm25Collector *collector)
{
	if (collector->docCount < collector->docCapacity)
		return;

	collector->docCapacity = collector->docCapacity == 0 ? 1024 : collector->docCapacity * 2;
	if (collector->docs == NULL)
		collector->docs = palloc(sizeof(TqHybridBm25BuildDoc) * collector->docCapacity);
	else
		collector->docs = repalloc(collector->docs,
								   sizeof(TqHybridBm25BuildDoc) * collector->docCapacity);
	TqHybridCheckBudget(collector);
}

static void
TqHybridEnsureTermCapacity(TqHybridBm25Collector *collector)
{
	if (collector->termCount < collector->termCapacity)
		return;

	collector->termCapacity = collector->termCapacity == 0 ? 4096 : collector->termCapacity * 2;
	if (collector->terms == NULL)
		collector->terms = palloc(sizeof(TqHybridBm25TermTuple) * collector->termCapacity);
	else
		collector->terms = repalloc(collector->terms,
									sizeof(TqHybridBm25TermTuple) * collector->termCapacity);
	TqHybridCheckBudget(collector);
}

static uint32
TqHybridAppendTermBytes(TqHybridBm25Collector *collector, const char *term,
						uint16 len)
{
	uint32		offset;

	if ((uint64) collector->termBytesUsed + len > PG_UINT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid BM25 term byte arena exceeded 4GB")));

	while (collector->termBytesUsed + len > collector->termBytesCapacity)
	{
		collector->termBytesCapacity = collector->termBytesCapacity == 0 ? 64 * 1024 :
			collector->termBytesCapacity * 2;
		if (collector->termBytes == NULL)
			collector->termBytes = palloc(collector->termBytesCapacity);
		else
			collector->termBytes = repalloc(collector->termBytes,
											collector->termBytesCapacity);
		TqHybridCheckBudget(collector);
	}

	offset = collector->termBytesUsed;
	memcpy(collector->termBytes + offset, term, len);
	collector->termBytesUsed += len;

	return offset;
}

static bool
TqHybridLookupNodeId(TqHybridBm25Collector *collector, ItemPointer tid,
					 uint32 *nodeId)
{
	TqHybridTidNode key;
	TqHybridTidNode *found;

	key.tid = *tid;
	key.nodeId = 0;
	found = bsearch(&key, collector->tidNodes, collector->tidNodeCount,
					sizeof(TqHybridTidNode), TqHybridTidNodeCompare);
	if (found == NULL)
		return false;

	*nodeId = found->nodeId;
	return true;
}

static uint32
TqHybridDocLen(TSVector vector)
{
	uint32		docLen = 0;
	WordEntry  *entries = ARRPTR(vector);

	for (int i = 0; i < vector->size; i++)
	{
		uint16		tf = POSDATALEN(vector, &entries[i]);

		docLen += tf > 0 ? tf : 1;
	}

	return docLen;
}

static TSVector
TqHybridDetoastTSVector(Datum value, bool *mustFree)
{
	TSVector	vector;

	vector = (TSVector) PG_DETOAST_DATUM(value);
	*mustFree = PointerGetDatum(vector) != value;
	return vector;
}

static void
TqHybridValidateTSVector(TSVector vector)
{
	WordEntry  *entries = ARRPTR(vector);
	char	   *strings = STRPTR(vector);
	Size		vectorSize = VARSIZE_ANY(vector);
	Size		dataOffset = (Size) (strings - (char *) vector);
	Size		dataSize;

	if (dataOffset > vectorSize)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 tsvector payload is invalid")));
	dataSize = vectorSize - dataOffset;

	for (int i = 0; i < vector->size; i++)
	{
		WordEntry  *entry = &entries[i];
		Size		termEnd = (Size) entry->pos + entry->len;

		if (termEnd > dataSize)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("turbohybrid BM25 tsvector lexeme offset is invalid")));

		if (entry->haspos)
		{
			Size		posOffset = SHORTALIGN(termEnd);
			uint16		npos;
			Size		posEnd;

			if (posOffset + sizeof(uint16) > dataSize)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("turbohybrid BM25 tsvector position offset is invalid")));
			npos = POSDATALEN(vector, entry);
			posEnd = posOffset + sizeof(uint16) +
				sizeof(WordEntryPos) * npos;
			if (posEnd > dataSize)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("turbohybrid BM25 tsvector positions are invalid")));
		}
	}
}

static void
TqHybridCollectVectorTerms(TqHybridBm25Collector *collector, uint32 nodeId,
						   TSVector vector)
{
	WordEntry  *entries = ARRPTR(vector);
	char	   *strings = STRPTR(vector);

	for (int i = 0; i < vector->size; i++)
	{
		WordEntry  *entry = &entries[i];
		char	   *term = strings + entry->pos;
		uint16		tf = POSDATALEN(vector, entry);
		TqHybridBm25TermTuple *tuple;
		uint32		termOffset;

		if (entry->len <= 0)
			continue;

		TqHybridEnsureTermCapacity(collector);
		termOffset = TqHybridAppendTermBytes(collector, term, entry->len);
		tuple = &collector->terms[collector->termCount++];
		tuple->termHash = TqHybridHashTerm(term, entry->len);
		tuple->nodeId = nodeId;
		tuple->tf = tf > 0 ? tf : 1;
		tuple->termLen = entry->len;
		tuple->termOffset = termOffset;
		collector->totalTermCount++;
		TqHybridCheckBudget(collector);
	}
}

static void
TqHybridAppendBuildDoc(TqHybridBm25Collector *collector, uint32 nodeId,
					   ItemPointer heapTid, uint32 docLen)
{
	TqHybridBm25BuildDoc *doc;

	TqHybridEnsureDocCapacity(collector);
	doc = &collector->docs[collector->docCount++];
	doc->nodeId = nodeId;
	doc->heaptid = *heapTid;
	doc->docLen = docLen;
	collector->totalDocLen += docLen;
	collector->maxDocLen = Max(collector->maxDocLen, docLen);
}

static void
TqHybridAppendBuildTerm(TqHybridBm25Collector *collector, uint32 nodeId,
						const char *term, uint16 termLen, uint16 tf)
{
	TqHybridBm25TermTuple *tuple;
	uint32		termOffset;

	if (termLen == 0)
		return;

	TqHybridEnsureTermCapacity(collector);
	termOffset = TqHybridAppendTermBytes(collector, term, termLen);
	tuple = &collector->terms[collector->termCount++];
	tuple->termHash = TqHybridHashTerm(term, termLen);
	tuple->nodeId = nodeId;
	tuple->tf = tf > 0 ? tf : 1;
	tuple->termLen = termLen;
	tuple->termOffset = termOffset;
	collector->totalTermCount++;
	TqHybridCheckBudget(collector);
}

static void
TqHybridBm25BuildCallback(Relation index, ItemPointer tid, Datum *values,
						  bool *isnull, bool tupleIsAlive, void *opaque)
{
	TqHybridBm25Collector *collector = (TqHybridBm25Collector *) opaque;
	TSVector	vector;
	bool		mustFree;
	uint32		nodeId;
	uint32		docLen;

	(void) index;
	(void) tupleIsAlive;

	if (isnull[1])
		return;

	vector = TqHybridDetoastTSVector(values[1], &mustFree);
	TqHybridValidateTSVector(vector);
	if (!TqHybridLookupNodeId(collector, tid, &nodeId))
	{
		if (mustFree)
			pfree(vector);
		return;
	}

	docLen = TqHybridDocLen(vector);

	TqHybridAppendBuildDoc(collector, nodeId, tid, docLen);
	TqHybridCollectVectorTerms(collector, nodeId, vector);
	if (mustFree)
		pfree(vector);
}

static TqHybridTidNode *
TqHybridReadNodeMap(Relation index, uint32 *count)
{
	Buffer		metaBuf;
	Page		metaPage;
	HnswMetaPage meta;
	uint32		seen = 0;
	TqHybridTidNode *map;
	int			codeTuplesPerPage;
	int			codePageCount;
	BlockNumber *codeBlknos;
	bool		tqWeighted;

	metaBuf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
	metaPage = BufferGetPage(metaBuf);
	meta = HnswPageGetMeta(metaPage);

	if (meta->magicNumber != HNSW_MAGIC_NUMBER ||
		meta->storageKind != HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE ||
		meta->tqNodeCount == 0 ||
		!BlockNumberIsValid(meta->tqCodeStartBlkno))
	{
		UnlockReleaseBuffer(metaBuf);
		*count = 0;
		return NULL;
	}

	map = palloc0(sizeof(TqHybridTidNode) * meta->tqNodeCount);
	tqWeighted = (meta->tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0;
	codeTuplesPerPage =
		TqGraphTuplesPerPage(TqGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  tqWeighted));
	codePageCount = TqGraphPageCount(meta->tqNodeCount, codeTuplesPerPage);
	codeBlknos = palloc(sizeof(BlockNumber) * codePageCount);
	TqGraphInitBlockMap(codeBlknos, codePageCount);

	for (int pageNo = 0; pageNo < codePageCount; pageNo++)
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
			if (tuple->type != TQ_GRAPH_CODE_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount)
				continue;

			map[seen].tid = tuple->heaptid;
			map[seen].nodeId = tuple->nodeId;
			seen++;
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(codeBlknos);
	UnlockReleaseBuffer(metaBuf);

	qsort(map, seen, sizeof(TqHybridTidNode), TqHybridTidNodeCompare);
	*count = seen;
	return map;
}

static TqHybridNodeState *
TqHybridReadNodeStates(Relation index, HnswMetaPageData *meta, uint32 *count)
{
	TqHybridNodeState *states;
	int			codeTuplesPerPage;
	int			codePageCount;
	BlockNumber *codeBlknos;
	bool		tqWeighted;

	if (!TqGraphReadMeta(index, meta) ||
		meta->storageKind != HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE ||
		meta->tqNodeCount == 0 ||
		!BlockNumberIsValid(meta->tqCodeStartBlkno))
	{
		*count = 0;
		return NULL;
	}

	states = palloc0(sizeof(TqHybridNodeState) * meta->tqNodeCount);
	tqWeighted = (meta->tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0;
	codeTuplesPerPage =
		TqGraphTuplesPerPage(TqGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  tqWeighted));
	codePageCount = TqGraphPageCount(meta->tqNodeCount, codeTuplesPerPage);
	codeBlknos = palloc(sizeof(BlockNumber) * codePageCount);
	TqGraphInitBlockMap(codeBlknos, codePageCount);

	for (int pageNo = 0; pageNo < codePageCount; pageNo++)
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
			if (tuple->type != TQ_GRAPH_CODE_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount)
				continue;

			states[tuple->nodeId].tid = tuple->heaptid;
			states[tuple->nodeId].live =
				(tuple->flags & TQ_GRAPH_NODE_DEAD) == 0;
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(codeBlknos);
	*count = meta->tqNodeCount;
	return states;
}

static uint32
TqHybridReduceUniqueTerms(TqHybridBm25Collector *collector)
{
	uint32		uniqueTerms = 0;

	if (collector->termCount == 0)
		return 0;

	TqHybridSortTermRun(collector);

	for (uint32 i = 0; i < collector->termCount;)
	{
		TqHybridBm25TermTuple *first = &collector->terms[i];
		uint32		df = 0;
		uint64		cf = 0;
		uint32		prevNode = PG_UINT32_MAX;

		uniqueTerms++;
		while (i < collector->termCount &&
			   TqHybridTermEqualIgnoringNode(first, &collector->terms[i],
											 collector->termBytes))
		{
			if (collector->terms[i].nodeId != prevNode)
			{
				df++;
				prevNode = collector->terms[i].nodeId;
			}
			cf += collector->terms[i].tf;
			i++;
		}

		(void) df;
		(void) cf;
	}

	return uniqueTerms;
}

static OffsetNumber
TqHybridBm25AddWalItem(Relation index, ForkNumber forkNum, Buffer *buf,
					   Page *page, BlockNumber *startBlkno,
					   uint16 pageKind, Item item, Size itemSize,
					   uint32 *pageCount, BlockNumber *insertBlkno)
{
	GenericXLogState *xlogState;
	OffsetNumber offno;
	BlockNumber blkno;
	bool		createdStart = !BlockNumberIsValid(*startBlkno);

	TqHybridBm25EnsureWalTail(index, forkNum, buf, page, startBlkno,
							  pageKind);
	if (createdStart && pageCount != NULL)
		(*pageCount)++;

	if (PageGetFreeSpace(*page) < itemSize)
	{
		Buffer		newbuf;
		Page		newpage;
		BlockNumber linkBlkno = BufferGetBlockNumber(*buf);
		BlockNumber initBlkno;

		xlogState = GenericXLogStart(index);
		*page = GenericXLogRegisterBuffer(xlogState, *buf, 0);

		LockRelationForExtension(index, ExclusiveLock);
		newbuf = HnswNewBuffer(index, forkNum);
		UnlockRelationForExtension(index, ExclusiveLock);
		initBlkno = BufferGetBlockNumber(newbuf);
		newpage = GenericXLogRegisterBuffer(xlogState, newbuf,
											GENERIC_XLOG_FULL_IMAGE);

		HnswPageGetOpaque(*page)->nextblkno = initBlkno;
		HnswMarkPageGraphOp(*page, HNSW_GRAPH_OP_PAGE_LINK);
		HnswInitPageKind(newbuf, newpage, pageKind);

		offno = PageAddItem(newpage, item, itemSize, InvalidOffsetNumber,
							false, false);
		if (offno == InvalidOffsetNumber)
			elog(ERROR, "failed to append turbohybrid BM25 WAL item to \"%s\"",
				 RelationGetRelationName(index));
		HnswMarkPageGraphOp(newpage, HNSW_GRAPH_OP_ELEMENT_INSERT);

		GenericXLogFinish(xlogState);
		HnswLogGraphWalRecord(index, forkNum, linkBlkno,
							  HNSW_GRAPH_OP_PAGE_LINK);
		HnswLogGraphWalRecord(index, forkNum, initBlkno,
							  HNSW_GRAPH_OP_PAGE_INIT);
		HnswLogGraphWalRecord(index, forkNum, initBlkno,
							  HNSW_GRAPH_OP_ELEMENT_INSERT);

		UnlockReleaseBuffer(*buf);
		*buf = newbuf;
		*page = BufferGetPage(newbuf);
		blkno = initBlkno;
		if (pageCount != NULL)
			(*pageCount)++;
	}
	else
	{
		blkno = BufferGetBlockNumber(*buf);
		xlogState = GenericXLogStart(index);
		*page = GenericXLogRegisterBuffer(xlogState, *buf, 0);
		offno = PageAddItem(*page, item, itemSize, InvalidOffsetNumber,
							false, false);
		if (offno == InvalidOffsetNumber)
			elog(ERROR, "failed to append turbohybrid BM25 WAL item to \"%s\"",
				 RelationGetRelationName(index));
		HnswMarkPageGraphOp(*page, HNSW_GRAPH_OP_ELEMENT_INSERT);
		GenericXLogFinish(xlogState);
		HnswLogGraphWalRecord(index, forkNum, blkno,
							  HNSW_GRAPH_OP_ELEMENT_INSERT);
		*page = BufferGetPage(*buf);
	}

	if (insertBlkno != NULL)
		*insertBlkno = blkno;
	return offno;
}

static void
TqHybridBm25EnsureWalTail(Relation index, ForkNumber forkNum, Buffer *buf,
						  Page *page, BlockNumber *startBlkno,
						  uint16 pageKind)
{
	BlockNumber blkno = *startBlkno;

	if (BufferIsValid(*buf))
	{
		if (!TqHybridBm25PageIsKind(*page, pageKind))
			elog(ERROR, "unexpected turbohybrid BM25 page kind while appending");
		return;
	}

	if (!BlockNumberIsValid(blkno))
	{
		GenericXLogState *xlogState;

		LockRelationForExtension(index, ExclusiveLock);
		*buf = HnswNewBuffer(index, forkNum);
		UnlockRelationForExtension(index, ExclusiveLock);
		blkno = BufferGetBlockNumber(*buf);
		*startBlkno = blkno;

		xlogState = GenericXLogStart(index);
		*page = GenericXLogRegisterBuffer(xlogState, *buf,
										  GENERIC_XLOG_FULL_IMAGE);
		HnswInitPageKind(*buf, *page, pageKind);
		GenericXLogFinish(xlogState);
		HnswLogGraphWalRecord(index, forkNum, blkno,
							  HNSW_GRAPH_OP_PAGE_INIT);
		*page = BufferGetPage(*buf);
		return;
	}

	for (;;)
	{
		BlockNumber nextblkno;

		*buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
		LockBuffer(*buf, BUFFER_LOCK_SHARE);
		*page = BufferGetPage(*buf);
		if (!TqHybridBm25PageIsKind(*page, pageKind))
		{
			UnlockReleaseBuffer(*buf);
			*buf = InvalidBuffer;
			*page = NULL;
			elog(ERROR, "unexpected turbohybrid BM25 page kind while appending");
		}
		nextblkno = HnswPageGetOpaque(*page)->nextblkno;
		UnlockReleaseBuffer(*buf);

		if (!BlockNumberIsValid(nextblkno))
			break;
		blkno = nextblkno;
	}

	*buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
	LockBuffer(*buf, BUFFER_LOCK_EXCLUSIVE);
	*page = BufferGetPage(*buf);
	if (!TqHybridBm25PageIsKind(*page, pageKind))
	{
		UnlockReleaseBuffer(*buf);
		*buf = InvalidBuffer;
		*page = NULL;
		elog(ERROR, "unexpected turbohybrid BM25 page kind while appending");
	}
}

static OffsetNumber
TqHybridBm25AddItem(Relation index, ForkNumber forkNum, Buffer *buf, Page *page,
					BlockNumber *startBlkno, uint16 pageKind, Item item,
					Size itemSize, uint32 *pageCount, bool useWal,
					BlockNumber *insertBlkno)
{
	OffsetNumber offno;
	Size		maxItemSize = BLCKSZ - SizeOfPageHeaderData -
		MAXALIGN(sizeof(HnswPageOpaqueData));

	if (itemSize > maxItemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid BM25 tuple exceeds page size"),
				 errdetail("Postings are chunked before storage; an oversized tuple indicates a single metadata, lexicon, or delta tuple is too large for one index page."),
				 errhint("Shorten individual lexemes or tsvector values; common-term postings spill and chunk automatically.")));

	if (useWal)
		return TqHybridBm25AddWalItem(index, forkNum, buf, page, startBlkno,
									  pageKind, item, itemSize, pageCount,
									  insertBlkno);

	if (!BufferIsValid(*buf) || PageGetFreeSpace(*page) < itemSize)
	{
		TqGraphAppendPage(index, forkNum, buf, page, pageKind);
		if (!BlockNumberIsValid(*startBlkno))
			*startBlkno = BufferGetBlockNumber(*buf);
		if (pageCount != NULL)
			(*pageCount)++;
	}

	offno = PageAddItem(*page, item, itemSize, InvalidOffsetNumber, false, false);
	if (offno == InvalidOffsetNumber)
		elog(ERROR, "failed to add turbohybrid BM25 item to \"%s\"",
			 RelationGetRelationName(index));

	if (insertBlkno != NULL)
		*insertBlkno = BufferGetBlockNumber(*buf);
	return offno;
}

static void
TqHybridWriteDocStats(TqHybridBm25Collector *collector)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	TqBm25DocStat *dense;
	uint32		nodeId = 0;
	uint16		maxDocsPerTuple;

	if (collector->docCount == 0)
	{
		collector->bm25DocStatsStartBlkno = InvalidBlockNumber;
		return;
	}

	dense = palloc0(sizeof(TqBm25DocStat) * collector->tidNodeCount);
	for (uint32 i = 0; i < collector->docCount; i++)
	{
		if (collector->docs[i].nodeId >= collector->tidNodeCount)
			elog(ERROR, "turbohybrid BM25 doc nodeId out of range");
		dense[collector->docs[i].nodeId].docLen = collector->docs[i].docLen;
	}
	collector->denseDocLens = palloc0(sizeof(uint32) * collector->tidNodeCount);
	collector->denseDocLensCount = collector->tidNodeCount;
	for (uint32 i = 0; i < collector->tidNodeCount; i++)
		collector->denseDocLens[i] = dense[i].docLen;

	maxDocsPerTuple = (BLCKSZ / 2 - offsetof(TqHybridBm25DocStatsTupleData, docs)) /
		sizeof(TqBm25DocStat);
	while (nodeId < collector->tidNodeCount)
	{
		uint16		count = Min((uint32) maxDocsPerTuple,
								collector->tidNodeCount - nodeId);
		Size		size = TqHybridBm25DocStatsTupleSize(count);
		TqHybridBm25DocStatsTuple tuple = palloc0(size);

		tuple->type = TQHYBRID_BM25_DOCSTATS_TUPLE_TYPE;
		tuple->count = count;
		tuple->startNodeId = nodeId;
		memcpy(tuple->docs, &dense[nodeId], sizeof(TqBm25DocStat) * count);

		(void) TqHybridBm25AddItem(collector->index, MAIN_FORKNUM, &buf, &page,
								   &start, HNSW_PAGE_KIND_TQ_BM25_DOCSTATS,
								   (Item) tuple, size, NULL,
								   collector->walLoggedWrites, NULL);
		pfree(tuple);
		nodeId += count;
	}

	if (BufferIsValid(buf))
		UnlockReleaseBuffer(buf);
	collector->bm25DocStatsStartBlkno = start;
	pfree(dense);
}

static OffsetNumber
TqHybridWritePostingsChunk(TqHybridBm25Collector *collector, uint32 termId,
						   uint32 chunkNo, uint32 startIndex, uint32 endIndex,
						   Buffer *buf, Page *page,
						   BlockNumber *postingsBlkno, uint32 *postingsBytes)
{
	uint32		count32 = endIndex - startIndex;
	uint16		count;
	Size		size;
	Size		payloadBytes = 0;
	TqHybridBm25PostingsTuple tuple;
	OffsetNumber offno;
	BlockNumber insertBlkno = InvalidBlockNumber;
	char	   *ptr;
	uint32		prevNodeId;
	bool		useOffset16 = true;
	bool		useDelta16 = true;
	bool		hasTfNorm = TqHybridBm25PrecomputeTfNormEnabled(collector);

	count = (uint16) count32;
	prevNodeId = collector->terms[startIndex].nodeId;
	for (uint32 i = 0; i < count32; i++)
	{
		TqHybridBm25TermTuple *term = &collector->terms[startIndex + i];

		if (!TqHybridBm25CanUseOffset16(collector->terms[startIndex].nodeId,
										term->nodeId))
			useOffset16 = false;
		if (!TqHybridBm25CanUseDelta16(prevNodeId, term->nodeId))
			useDelta16 = false;
		prevNodeId = term->nodeId;
	}
	if (useOffset16 || useDelta16)
		payloadBytes = count32 * sizeof(uint16) * 2;
	else
	{
		prevNodeId = collector->terms[startIndex].nodeId;
		for (uint32 i = 0; i < count32; i++)
		{
			TqHybridBm25TermTuple *term = &collector->terms[startIndex + i];
			uint32		delta = term->nodeId - prevNodeId;

			payloadBytes += TqHybridBm25VarintLen(delta);
			payloadBytes += TqHybridBm25VarintLen(term->tf);
			prevNodeId = term->nodeId;
		}
	}
	if (hasTfNorm)
		payloadBytes += count32 * sizeof(uint16);
	size = TqHybridBm25PostingsTupleSize(payloadBytes);
	tuple = palloc0(size);
	tuple->type = TQHYBRID_BM25_POSTINGS_TUPLE_TYPE;
	tuple->count = count;
	tuple->termId = termId;
	tuple->chunkNo = chunkNo;
	tuple->firstNodeId = collector->terms[startIndex].nodeId;
	tuple->lastNodeId = collector->terms[endIndex - 1].nodeId;
	tuple->nextBlkno = InvalidBlockNumber;
	tuple->nextOffno = InvalidOffsetNumber;
	tuple->encoding = useOffset16 ?
		TQHYBRID_BM25_POSTINGS_ENCODING_OFFSET16 :
		(useDelta16 ?
		 TQHYBRID_BM25_POSTINGS_ENCODING_DELTA16 :
		 TQHYBRID_BM25_POSTINGS_ENCODING_DELTA_VARINT);
	if (hasTfNorm)
		tuple->encoding |= TQHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16;
	tuple->payloadBytes = (uint16) payloadBytes;
	ptr = tuple->payload;
	prevNodeId = tuple->firstNodeId;
	for (uint32 i = 0; i < count32; i++)
	{
		TqHybridBm25TermTuple *term = &collector->terms[startIndex + i];

		tuple->maxTf = Max(tuple->maxTf, term->tf);
		if (useOffset16)
			TqHybridBm25EncodePostingOffset16(&ptr, tuple->firstNodeId,
											  term->nodeId, term->tf);
		else if (useDelta16)
			TqHybridBm25EncodePostingDelta16(&ptr, &prevNodeId, term->nodeId,
											 term->tf);
		else
			TqHybridBm25EncodePosting(&ptr, &prevNodeId, term->nodeId,
									  term->tf);
	}
	if (hasTfNorm)
	{
		for (uint32 i = 0; i < count32; i++)
		{
			TqHybridBm25TermTuple *term = &collector->terms[startIndex + i];

			TqHybridBm25PutUint16(&ptr,
								  TqHybridBm25QuantizeTfNorm(collector,
															 term->nodeId,
															 term->tf));
		}
	}

	offno = TqHybridBm25AddItem(collector->index, MAIN_FORKNUM, buf, page,
								&collector->bm25PostingsStartBlkno,
								HNSW_PAGE_KIND_TQ_BM25_POSTINGS,
								(Item) tuple, size,
								&collector->bm25PostingsPages,
								collector->walLoggedWrites,
								&insertBlkno);
	*postingsBlkno = insertBlkno;
	*postingsBytes = size;
	pfree(tuple);
	return offno;
}

static void
TqHybridLinkPostingsChunk(Relation index, Buffer currentBuf, Page currentPage,
						  BlockNumber prevBlkno, OffsetNumber prevOffno,
						  BlockNumber nextBlkno, OffsetNumber nextOffno)
{
	Buffer		buf;
	Page		page;
	ItemId		iid;
	TqHybridBm25PostingsTuple tuple;
	GenericXLogState *xlogState = NULL;
	bool		modified = false;
	bool		current = BufferIsValid(currentBuf) &&
		BufferGetBlockNumber(currentBuf) == prevBlkno;

	if (current)
	{
		buf = currentBuf;
		page = currentPage;
	}
	else
	{
		buf = ReadBuffer(index, prevBlkno);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);
	}

	if (RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}

	if (prevOffno <= PageGetMaxOffsetNumber(page))
	{
		iid = PageGetItemId(page, prevOffno);
		if (ItemIdIsUsed(iid))
		{
			tuple = (TqHybridBm25PostingsTuple) PageGetItem(page, iid);
			if (tuple->type == TQHYBRID_BM25_POSTINGS_TUPLE_TYPE)
			{
				tuple->nextBlkno = nextBlkno;
				tuple->nextOffno = nextOffno;
				modified = true;
			}
		}
	}

	if (xlogState != NULL)
	{
		if (modified)
			GenericXLogFinish(xlogState);
		else
			GenericXLogAbort(xlogState);
	}
	else if (modified)
		MarkBufferDirty(buf);

	if (!current)
		UnlockReleaseBuffer(buf);
}

static OffsetNumber
TqHybridWritePostings(TqHybridBm25Collector *collector, uint32 termId,
					  uint32 startIndex, uint32 endIndex,
					  Buffer *buf, Page *page,
					  BlockNumber *postingsBlkno, uint32 *postingsBytes,
					  uint32 *postingsChunkCount)
{
	uint16		maxPerChunk = TqHybridBm25MaxPostingsPerChunk();
	OffsetNumber firstOffno = InvalidOffsetNumber;
	BlockNumber firstBlkno = InvalidBlockNumber;
	BlockNumber prevBlkno = InvalidBlockNumber;
	OffsetNumber prevOffno = InvalidOffsetNumber;

	*postingsBytes = 0;
	*postingsChunkCount = 0;
	for (uint32 chunkStart = startIndex; chunkStart < endIndex;)
	{
		uint32		chunkEnd = Min(chunkStart + (uint32) maxPerChunk, endIndex);
		BlockNumber chunkBlkno;
		OffsetNumber chunkOffno;
		uint32		chunkBytes;

		chunkOffno = TqHybridWritePostingsChunk(collector, termId,
												*postingsChunkCount,
												chunkStart, chunkEnd,
												buf, page,
												&chunkBlkno, &chunkBytes);
		if (!OffsetNumberIsValid(firstOffno))
		{
			firstOffno = chunkOffno;
			firstBlkno = chunkBlkno;
		}
		if (BlockNumberIsValid(prevBlkno))
			TqHybridLinkPostingsChunk(collector->index, *buf, *page,
									  prevBlkno, prevOffno,
									  chunkBlkno, chunkOffno);

		prevBlkno = chunkBlkno;
		prevOffno = chunkOffno;
		*postingsBytes += chunkBytes;
		(*postingsChunkCount)++;
		chunkStart = chunkEnd;
	}

	*postingsBlkno = firstBlkno;
	return firstOffno;
}

static OffsetNumber
TqHybridWritePostingsChunkData(TqHybridBm25Collector *collector, uint32 termId,
							   uint32 chunkNo,
							   const TqHybridBm25Posting *postings,
							   uint16 count, Buffer *buf, Page *page,
							   BlockNumber *postingsBlkno,
							   uint32 *postingsBytes)
{
	Size		size;
	Size		payloadBytes = 0;
	TqHybridBm25PostingsTuple tuple;
	OffsetNumber offno;
	BlockNumber insertBlkno = InvalidBlockNumber;
	char	   *ptr;
	uint32		prevNodeId;
	bool		useOffset16 = true;
	bool		useDelta16 = true;
	bool		hasTfNorm = TqHybridBm25PrecomputeTfNormEnabled(collector);

	prevNodeId = postings[0].nodeId;
	for (uint16 i = 0; i < count; i++)
	{
		if (!TqHybridBm25CanUseOffset16(postings[0].nodeId,
										postings[i].nodeId))
			useOffset16 = false;
		if (!TqHybridBm25CanUseDelta16(prevNodeId, postings[i].nodeId))
			useDelta16 = false;
		prevNodeId = postings[i].nodeId;
	}
	if (useOffset16 || useDelta16)
		payloadBytes = count * sizeof(uint16) * 2;
	else
	{
		prevNodeId = postings[0].nodeId;
		for (uint16 i = 0; i < count; i++)
		{
			uint32		delta = postings[i].nodeId - prevNodeId;

			payloadBytes += TqHybridBm25VarintLen(delta);
			payloadBytes += TqHybridBm25VarintLen(postings[i].tf);
			prevNodeId = postings[i].nodeId;
		}
	}
	if (hasTfNorm)
		payloadBytes += count * sizeof(uint16);
	size = TqHybridBm25PostingsTupleSize(payloadBytes);
	tuple = palloc0(size);
	tuple->type = TQHYBRID_BM25_POSTINGS_TUPLE_TYPE;
	tuple->count = count;
	tuple->termId = termId;
	tuple->chunkNo = chunkNo;
	tuple->firstNodeId = postings[0].nodeId;
	tuple->lastNodeId = postings[count - 1].nodeId;
	tuple->nextBlkno = InvalidBlockNumber;
	tuple->nextOffno = InvalidOffsetNumber;
	tuple->encoding = useOffset16 ?
		TQHYBRID_BM25_POSTINGS_ENCODING_OFFSET16 :
		(useDelta16 ?
		 TQHYBRID_BM25_POSTINGS_ENCODING_DELTA16 :
		 TQHYBRID_BM25_POSTINGS_ENCODING_DELTA_VARINT);
	if (hasTfNorm)
		tuple->encoding |= TQHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16;
	tuple->payloadBytes = (uint16) payloadBytes;
	ptr = tuple->payload;
	prevNodeId = tuple->firstNodeId;
	for (uint16 i = 0; i < count; i++)
	{
		tuple->maxTf = Max(tuple->maxTf, postings[i].tf);
		if (useOffset16)
			TqHybridBm25EncodePostingOffset16(&ptr, tuple->firstNodeId,
											  postings[i].nodeId,
											  postings[i].tf);
		else if (useDelta16)
			TqHybridBm25EncodePostingDelta16(&ptr, &prevNodeId,
											 postings[i].nodeId,
											 postings[i].tf);
		else
			TqHybridBm25EncodePosting(&ptr, &prevNodeId, postings[i].nodeId,
									  postings[i].tf);
	}
	if (hasTfNorm)
	{
		for (uint16 i = 0; i < count; i++)
			TqHybridBm25PutUint16(&ptr,
								  TqHybridBm25QuantizeTfNorm(collector,
															 postings[i].nodeId,
															 postings[i].tf));
	}

	offno = TqHybridBm25AddItem(collector->index, MAIN_FORKNUM, buf, page,
								&collector->bm25PostingsStartBlkno,
								HNSW_PAGE_KIND_TQ_BM25_POSTINGS,
								(Item) tuple, size,
								&collector->bm25PostingsPages,
								collector->walLoggedWrites,
								&insertBlkno);
	*postingsBlkno = insertBlkno;
	*postingsBytes = size;
	pfree(tuple);
	return offno;
}

static OffsetNumber
TqHybridWriteBlockMax(TqHybridBm25Collector *collector, uint32 termId,
					  uint32 startIndex, uint32 endIndex,
					  Buffer *buf, Page *page, BlockNumber *blockMaxBlkno)
{
	TqHybridBm25BlockMaxTupleData tuple;
	uint16		maxTf = 0;
	OffsetNumber offno;
	BlockNumber insertBlkno = InvalidBlockNumber;

	memset(&tuple, 0, sizeof(tuple));
	tuple.type = TQHYBRID_BM25_BLOCKMAX_TUPLE_TYPE;
	tuple.count = 1;
	tuple.termId = termId;
	tuple.firstNodeId = collector->terms[startIndex].nodeId;
	tuple.lastNodeId = collector->terms[endIndex - 1].nodeId;
	for (uint32 i = startIndex; i < endIndex; i++)
		maxTf = Max(maxTf, collector->terms[i].tf);
	tuple.maxTf = maxTf;
	tuple.maxScoreUpperBound = (float4) maxTf;

	offno = TqHybridBm25AddItem(collector->index, MAIN_FORKNUM, buf, page,
								&collector->bm25BlockMaxStartBlkno,
								HNSW_PAGE_KIND_TQ_BM25_BLOCKMAX,
								(Item) &tuple, MAXALIGN(sizeof(tuple)),
								&collector->bm25BlockMaxPages,
								collector->walLoggedWrites,
								&insertBlkno);
	*blockMaxBlkno = insertBlkno;
	return offno;
}

static OffsetNumber
TqHybridWriteBlockMaxData(TqHybridBm25Collector *collector, uint32 termId,
						  uint32 firstNodeId, uint32 lastNodeId, uint16 maxTf,
						  Buffer *buf, Page *page, BlockNumber *blockMaxBlkno)
{
	TqHybridBm25BlockMaxTupleData tuple;
	OffsetNumber offno;
	BlockNumber insertBlkno = InvalidBlockNumber;

	memset(&tuple, 0, sizeof(tuple));
	tuple.type = TQHYBRID_BM25_BLOCKMAX_TUPLE_TYPE;
	tuple.count = 1;
	tuple.termId = termId;
	tuple.firstNodeId = firstNodeId;
	tuple.lastNodeId = lastNodeId;
	tuple.maxTf = maxTf;
	tuple.maxScoreUpperBound = (float4) maxTf;

	offno = TqHybridBm25AddItem(collector->index, MAIN_FORKNUM, buf, page,
								&collector->bm25BlockMaxStartBlkno,
								HNSW_PAGE_KIND_TQ_BM25_BLOCKMAX,
								(Item) &tuple, MAXALIGN(sizeof(tuple)),
								&collector->bm25BlockMaxPages,
								collector->walLoggedWrites,
								&insertBlkno);
	*blockMaxBlkno = insertBlkno;
	return offno;
}

static void
TqHybridSpillCursorCloseCurrent(TqHybridBm25SpillCursor *cursor)
{
	if (cursor->termBytes != NULL)
	{
		pfree(cursor->termBytes);
		cursor->termBytes = NULL;
	}
	cursor->valid = false;
}

static bool
TqHybridSpillCursorRead(TqHybridBm25SpillCursor *cursor)
{
	TqHybridSpillCursorCloseCurrent(cursor);
	if (cursor->remaining == 0)
		return false;

	BufFileReadExact(cursor->run->file, &cursor->termHash,
					 sizeof(cursor->termHash));
	BufFileReadExact(cursor->run->file, &cursor->nodeId,
					 sizeof(cursor->nodeId));
	BufFileReadExact(cursor->run->file, &cursor->tf, sizeof(cursor->tf));
	BufFileReadExact(cursor->run->file, &cursor->termLen,
					 sizeof(cursor->termLen));
	cursor->termBytes = palloc(cursor->termLen);
	BufFileReadExact(cursor->run->file, cursor->termBytes, cursor->termLen);
	cursor->remaining--;
	cursor->valid = true;
	return true;
}

static int
TqHybridSpillCursorCompare(const TqHybridBm25SpillCursor *a,
						   const TqHybridBm25SpillCursor *b)
{
	int			cmp;

	if (a->termHash != b->termHash)
		return a->termHash < b->termHash ? -1 : 1;
	if (a->termLen != b->termLen)
		return a->termLen < b->termLen ? -1 : 1;
	cmp = memcmp(a->termBytes, b->termBytes, a->termLen);
	if (cmp != 0)
		return cmp;
	return (a->nodeId > b->nodeId) - (a->nodeId < b->nodeId);
}

static int
TqHybridFindNextSpillCursor(TqHybridBm25SpillCursor *cursors, uint32 count)
{
	int			best = -1;

	for (uint32 i = 0; i < count; i++)
	{
		if (!cursors[i].valid)
			continue;
		if (best < 0 ||
			TqHybridSpillCursorCompare(&cursors[i], &cursors[best]) < 0)
			best = (int) i;
	}

	return best;
}

static bool
TqHybridSpillSameTerm(TqHybridBm25SpillCursor *cursor,
					  uint64 termHash, const char *termBytes, uint16 termLen)
{
	return cursor->termHash == termHash &&
		cursor->termLen == termLen &&
		memcmp(cursor->termBytes, termBytes, termLen) == 0;
}

static void
TqHybridWriteLexiconItem(TqHybridBm25Collector *collector,
						 Buffer *lexBuf, Page *lexPage,
						 BlockNumber *lexStart, uint32 termId,
						 uint64 termHash, const char *termBytes,
						 uint16 termLen, uint32 df, uint32 cf,
						 BlockNumber postingsBlkno,
						 OffsetNumber postingsOffno,
						 uint32 postingsChunkCount, uint32 postingsBytes,
						 BlockNumber blockMaxBlkno,
						 OffsetNumber blockMaxOffno)
{
	Size		lexSize = TqHybridBm25LexiconEntrySize(termLen);
	TqHybridBm25LexiconEntry lex = palloc0(lexSize);

	lex->type = TQHYBRID_BM25_LEXICON_TUPLE_TYPE;
	lex->termLen = termLen;
	lex->termHash = termHash;
	lex->termId = termId;
	lex->df = df;
	lex->cf = cf;
	lex->postingsBlkno = postingsBlkno;
	lex->postingsOffno = postingsOffno;
	lex->postingsChunkCount = postingsChunkCount;
	lex->postingsBytes = postingsBytes;
	lex->blockMaxBlkno = blockMaxBlkno;
	lex->blockMaxOffno = blockMaxOffno;
	memcpy(lex->termBytes, termBytes, termLen);

	(void) TqHybridBm25AddItem(collector->index, MAIN_FORKNUM,
							   lexBuf, lexPage, lexStart,
							   HNSW_PAGE_KIND_TQ_BM25_LEXICON,
							   (Item) lex, lexSize, NULL,
							   collector->walLoggedWrites, NULL);
	pfree(lex);
}

static void
TqHybridWriteLexiconAndPostingsFromRuns(TqHybridBm25Collector *collector)
{
	Buffer		lexBuf = InvalidBuffer;
	Page		lexPage = NULL;
	Buffer		postingsBuf = InvalidBuffer;
	Page		postingsPage = NULL;
	Buffer		blockMaxBuf = InvalidBuffer;
	Page		blockMaxPage = NULL;
	BlockNumber lexStart = InvalidBlockNumber;
	TqHybridBm25SpillCursor *cursors;
	TqHybridBm25Posting *chunk;
	uint16		maxPerChunk = TqHybridBm25MaxPostingsPerChunk();
	uint32		termId = 0;

	cursors = palloc0(sizeof(TqHybridBm25SpillCursor) *
					  collector->spillRunCount);
	chunk = palloc(sizeof(TqHybridBm25Posting) * maxPerChunk);

	for (uint32 i = 0; i < collector->spillRunCount; i++)
	{
		cursors[i].run = &collector->spillRuns[i];
		cursors[i].remaining = collector->spillRuns[i].tupleCount;
		if (BufFileSeek(collector->spillRuns[i].file, 0, 0L, SEEK_SET) != 0)
			elog(ERROR, "failed to rewind turbohybrid BM25 spill run");
		(void) TqHybridSpillCursorRead(&cursors[i]);
	}

	for (;;)
	{
		int			best = TqHybridFindNextSpillCursor(cursors,
													   collector->spillRunCount);
		uint64		termHash;
		uint16		termLen;
		char	   *termBytes;
		uint32		df = 0;
		uint32		cf = 0;
		uint32		prevNode = PG_UINT32_MAX;
		uint16		maxTf = 0;
		uint16		chunkCount = 0;
		uint32		postingsChunkCount = 0;
		uint32		postingsBytes = 0;
		BlockNumber firstPostingsBlkno = InvalidBlockNumber;
		OffsetNumber firstPostingsOffno = InvalidOffsetNumber;
		BlockNumber prevPostingsBlkno = InvalidBlockNumber;
		OffsetNumber prevPostingsOffno = InvalidOffsetNumber;
		BlockNumber blockMaxBlkno;
		OffsetNumber blockMaxOffno;
		uint32		firstNodeId = PG_UINT32_MAX;
		uint32		lastNodeId = 0;

		if (best < 0)
			break;

		termHash = cursors[best].termHash;
		termLen = cursors[best].termLen;
		termBytes = palloc(termLen);
		memcpy(termBytes, cursors[best].termBytes, termLen);

		while (best >= 0 &&
			   TqHybridSpillSameTerm(&cursors[best], termHash, termBytes,
									 termLen))
		{
			uint32		nodeId = cursors[best].nodeId;
			uint16		tf = cursors[best].tf;

			if (nodeId != prevNode)
			{
				df++;
				prevNode = nodeId;
			}
			cf += tf;
			maxTf = Max(maxTf, tf);
			if (firstNodeId == PG_UINT32_MAX)
				firstNodeId = nodeId;
			lastNodeId = nodeId;

			chunk[chunkCount].nodeId = nodeId;
			chunk[chunkCount].tf = tf;
			chunk[chunkCount].reserved = 0;
			chunkCount++;
			if (chunkCount == maxPerChunk)
			{
				BlockNumber chunkBlkno;
				OffsetNumber chunkOffno;
				uint32		chunkBytes;

				chunkOffno = TqHybridWritePostingsChunkData(collector, termId,
															postingsChunkCount,
															chunk, chunkCount,
															&postingsBuf,
															&postingsPage,
															&chunkBlkno,
															&chunkBytes);
				if (!OffsetNumberIsValid(firstPostingsOffno))
				{
					firstPostingsOffno = chunkOffno;
					firstPostingsBlkno = chunkBlkno;
				}
				if (BlockNumberIsValid(prevPostingsBlkno))
					TqHybridLinkPostingsChunk(collector->index, postingsBuf,
											  postingsPage, prevPostingsBlkno,
											  prevPostingsOffno, chunkBlkno,
											  chunkOffno);
				prevPostingsBlkno = chunkBlkno;
				prevPostingsOffno = chunkOffno;
				postingsBytes += chunkBytes;
				postingsChunkCount++;
				chunkCount = 0;
			}

			(void) TqHybridSpillCursorRead(&cursors[best]);
			best = TqHybridFindNextSpillCursor(cursors,
											   collector->spillRunCount);
		}

		if (chunkCount > 0)
		{
			BlockNumber chunkBlkno;
			OffsetNumber chunkOffno;
			uint32		chunkBytes;

			chunkOffno = TqHybridWritePostingsChunkData(collector, termId,
														postingsChunkCount,
														chunk, chunkCount,
														&postingsBuf,
														&postingsPage,
														&chunkBlkno,
														&chunkBytes);
			if (!OffsetNumberIsValid(firstPostingsOffno))
			{
				firstPostingsOffno = chunkOffno;
				firstPostingsBlkno = chunkBlkno;
			}
			if (BlockNumberIsValid(prevPostingsBlkno))
				TqHybridLinkPostingsChunk(collector->index, postingsBuf,
										  postingsPage, prevPostingsBlkno,
										  prevPostingsOffno, chunkBlkno,
										  chunkOffno);
			postingsBytes += chunkBytes;
			postingsChunkCount++;
		}

		blockMaxOffno = TqHybridWriteBlockMaxData(collector, termId,
												  firstNodeId, lastNodeId,
												  maxTf, &blockMaxBuf,
												  &blockMaxPage,
												  &blockMaxBlkno);
		TqHybridWriteLexiconItem(collector, &lexBuf, &lexPage, &lexStart,
								 termId, termHash, termBytes, termLen, df, cf,
								 firstPostingsBlkno, firstPostingsOffno,
								 postingsChunkCount, postingsBytes,
								 blockMaxBlkno, blockMaxOffno);
		pfree(termBytes);
		termId++;
		CHECK_FOR_INTERRUPTS();
	}

	for (uint32 i = 0; i < collector->spillRunCount; i++)
		TqHybridSpillCursorCloseCurrent(&cursors[i]);
	pfree(cursors);
	pfree(chunk);

	if (BufferIsValid(lexBuf))
		UnlockReleaseBuffer(lexBuf);
	if (BufferIsValid(postingsBuf))
		UnlockReleaseBuffer(postingsBuf);
	if (BufferIsValid(blockMaxBuf))
		UnlockReleaseBuffer(blockMaxBuf);

	collector->bm25LexiconStartBlkno = lexStart;
	collector->uniqueTerms = termId;
}

static void
TqHybridWriteLexiconAndPostings(TqHybridBm25Collector *collector)
{
	Buffer		lexBuf = InvalidBuffer;
	Page		lexPage = NULL;
	Buffer		postingsBuf = InvalidBuffer;
	Page		postingsPage = NULL;
	Buffer		blockMaxBuf = InvalidBuffer;
	Page		blockMaxPage = NULL;
	BlockNumber lexStart = InvalidBlockNumber;
	uint32		termId = 0;

	if (collector->spillRunCount > 0)
	{
		TqHybridWriteLexiconAndPostingsFromRuns(collector);
		return;
	}

	for (uint32 i = 0; i < collector->termCount;)
	{
		TqHybridBm25TermTuple *first = &collector->terms[i];
		uint32		startIndex = i;
		uint32		df = 0;
		uint32		cf = 0;
		uint32		prevNode = PG_UINT32_MAX;
		BlockNumber postingsBlkno;
		OffsetNumber postingsOffno;
		uint32		postingsBytes;
		uint32		postingsChunkCount;
		BlockNumber blockMaxBlkno;
		OffsetNumber blockMaxOffno;

		while (i < collector->termCount &&
			   TqHybridTermEqualIgnoringNode(first, &collector->terms[i],
											 collector->termBytes))
		{
			if (collector->terms[i].nodeId != prevNode)
			{
				df++;
				prevNode = collector->terms[i].nodeId;
			}
			cf += collector->terms[i].tf;
			i++;
		}

		postingsOffno = TqHybridWritePostings(collector, termId, startIndex, i,
											  &postingsBuf, &postingsPage,
											  &postingsBlkno, &postingsBytes,
											  &postingsChunkCount);
		blockMaxOffno = TqHybridWriteBlockMax(collector, termId, startIndex, i,
											  &blockMaxBuf, &blockMaxPage,
											  &blockMaxBlkno);

		TqHybridWriteLexiconItem(collector, &lexBuf, &lexPage, &lexStart,
								 termId, first->termHash,
								 collector->termBytes + first->termOffset,
								 first->termLen, df, cf, postingsBlkno,
								 postingsOffno, postingsChunkCount,
								 postingsBytes, blockMaxBlkno, blockMaxOffno);
		termId++;
	}

	if (BufferIsValid(lexBuf))
		UnlockReleaseBuffer(lexBuf);
	if (BufferIsValid(postingsBuf))
		UnlockReleaseBuffer(postingsBuf);
	if (BufferIsValid(blockMaxBuf))
		UnlockReleaseBuffer(blockMaxBuf);

	collector->bm25LexiconStartBlkno = lexStart;
	collector->uniqueTerms = termId;
}

static void
TqHybridWriteMeta(TqHybridBm25Collector *collector)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	TqHybridBm25MetaTupleData tuple;
	TqHybridOptions *opts = (TqHybridOptions *) collector->index->rd_options;

	memset(&tuple, 0, sizeof(tuple));
	tuple.type = TQHYBRID_BM25_META_TUPLE_TYPE;
	if (opts != NULL && opts->bm25PrecomputeTfNorm)
		tuple.reserved2 |= TQHYBRID_BM25_META_FLAG_TFNORM_Q16;
	tuple.bm25Version = TQHYBRID_BM25_VERSION;
	tuple.docCount = collector->docCount;
	tuple.totalDocLen = collector->totalDocLen;
	tuple.avgDocLen = collector->docCount == 0 ? 0.0f :
		(float4) ((double) collector->totalDocLen / (double) collector->docCount);
	tuple.k1 = opts != NULL ? opts->bm25K1 : 1.2f;
	tuple.b = opts != NULL ? opts->bm25B : 0.75f;
	tuple.termCount = collector->uniqueTerms;
	tuple.termTupleCount = collector->totalTermCount;
	tuple.maxDocLen = collector->maxDocLen;
	tuple.docStatsStartBlkno = collector->bm25DocStatsStartBlkno;
	tuple.lexiconStartBlkno = collector->bm25LexiconStartBlkno;
	tuple.postingsStartBlkno = collector->bm25PostingsStartBlkno;
	tuple.blockMaxStartBlkno = collector->bm25BlockMaxStartBlkno;
	tuple.deltaStartBlkno = InvalidBlockNumber;
	tuple.deltaGeneration = 0;
	tuple.deltaDocCount = 0;
	tuple.deltaTotalDocLen = 0;
	tuple.deltaTermCount = 0;
	tuple.postingsPages = collector->bm25PostingsPages;
	tuple.blockMaxPages = collector->bm25BlockMaxPages;
	tuple.deltaPages = 0;
	tuple.lastCompactionGeneration = 0;
	tuple.compactionCount = 0;

	(void) TqHybridBm25AddItem(collector->index, MAIN_FORKNUM, &buf, &page,
							   &start, HNSW_PAGE_KIND_TQ_BM25_META,
							   (Item) &tuple, MAXALIGN(sizeof(tuple)), NULL,
							   false, NULL);
	UnlockReleaseBuffer(buf);
	collector->bm25MetaBlkno = start;
	TqHybridBm25SetMetaBlock(collector->index, start);
}

static void
TqHybridBm25WriteBasePages(TqHybridBm25Collector *collector)
{
	collector->bm25DocStatsStartBlkno = InvalidBlockNumber;
	collector->bm25LexiconStartBlkno = InvalidBlockNumber;
	collector->bm25PostingsStartBlkno = InvalidBlockNumber;
	collector->bm25BlockMaxStartBlkno = InvalidBlockNumber;
	collector->bm25PostingsPages = 0;
	collector->bm25BlockMaxPages = 0;

	TqHybridWriteDocStats(collector);
	TqHybridWriteLexiconAndPostings(collector);
}

static void
TqHybridBm25WriteStorage(TqHybridBm25Collector *collector)
{
	collector->bm25MetaBlkno = InvalidBlockNumber;

	TqHybridBm25WriteBasePages(collector);
	TqHybridWriteMeta(collector);
}

static char *
TqHybridBm25DeltaTermBytes(TqHybridBm25DeltaTuple tuple)
{
	return ((char *) tuple) + offsetof(TqHybridBm25DeltaTupleData, terms) +
		sizeof(TqHybridBm25DeltaTerm) * tuple->termCount;
}

static void
TqHybridBm25ReadDocLens(Relation index, const TqHybridBm25MetaTupleData *meta,
						uint32 nodeCount, uint32 *docLens)
{
	BlockNumber blkno = meta->docStatsStartBlkno;

	memset(docLens, 0, sizeof(uint32) * nodeCount);
	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);
		if (!TqHybridBm25PageIsKind(page, HNSW_PAGE_KIND_TQ_BM25_DOCSTATS))
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("turbohybrid BM25 docstats page has unexpected page kind")));
		}

		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			TqHybridBm25DocStatsTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqHybridBm25DocStatsTuple) PageGetItem(page, iid);
			if (tuple->type != TQHYBRID_BM25_DOCSTATS_TUPLE_TYPE)
				continue;
			for (uint16 i = 0; i < tuple->count; i++)
			{
				uint32		nodeId = tuple->startNodeId + i;

				if (nodeId < nodeCount)
					docLens[nodeId] = tuple->docs[i].docLen;
			}
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
	}
}

static void
TqHybridBm25CollectBasePostings(Relation index,
								const TqHybridBm25MetaTupleData *meta,
								TqHybridBm25Collector *collector,
								const TqHybridNodeState *nodeStates,
								const uint32 *docLens, uint32 nodeCount)
{
	BlockNumber blkno = meta->lexiconStartBlkno;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		lexBuf;
		Page		lexPage;
		HnswPageOpaque lexOpaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		lexBuf = ReadBuffer(index, blkno);
		LockBuffer(lexBuf, BUFFER_LOCK_SHARE);
		lexPage = BufferGetPage(lexBuf);
		lexOpaque = HnswPageGetOpaque(lexPage);
		if (!TqHybridBm25PageIsKind(lexPage, HNSW_PAGE_KIND_TQ_BM25_LEXICON))
		{
			UnlockReleaseBuffer(lexBuf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("turbohybrid BM25 lexicon page has unexpected page kind")));
		}

		nextblkno = lexOpaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(lexPage);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
			{
				ItemId		lexIid = PageGetItemId(lexPage, off);
				TqHybridBm25LexiconEntry lex;
				BlockNumber postBlkno;
				OffsetNumber postOffno;
				uint32		chunkLimit;

				if (!ItemIdIsUsed(lexIid))
					continue;

				lex = (TqHybridBm25LexiconEntry) PageGetItem(lexPage, lexIid);
				if (lex->type != TQHYBRID_BM25_LEXICON_TUPLE_TYPE)
					continue;
				if (!BlockNumberIsValid(lex->postingsBlkno) ||
					!OffsetNumberIsValid(lex->postingsOffno))
					continue;

				postBlkno = lex->postingsBlkno;
				postOffno = lex->postingsOffno;
				chunkLimit = Max(lex->postingsChunkCount, 1);
				for (uint32 chunkNo = 0;
					 chunkNo < chunkLimit && BlockNumberIsValid(postBlkno) &&
					 OffsetNumberIsValid(postOffno);
					 chunkNo++)
				{
					Buffer		postBuf;
					Page		postPage;
					ItemId		postIid;
						TqHybridBm25PostingsTuple postings;
						TqHybridBm25Posting *decoded;
						BlockNumber nextBlkno;
						OffsetNumber nextOffno;

					postBuf = ReadBuffer(index, postBlkno);
					LockBuffer(postBuf, BUFFER_LOCK_SHARE);
					postPage = BufferGetPage(postBuf);
					if (!TqHybridBm25PageIsKind(postPage, HNSW_PAGE_KIND_TQ_BM25_POSTINGS) ||
						postOffno > PageGetMaxOffsetNumber(postPage))
					{
						UnlockReleaseBuffer(postBuf);
						ereport(ERROR,
								(errcode(ERRCODE_DATA_CORRUPTED),
								 errmsg("turbohybrid BM25 postings pointer is invalid")));
					}

					postIid = PageGetItemId(postPage, postOffno);
					if (!ItemIdIsUsed(postIid))
					{
						UnlockReleaseBuffer(postBuf);
						break;
					}

					postings = (TqHybridBm25PostingsTuple) PageGetItem(postPage, postIid);
					if (postings->type != TQHYBRID_BM25_POSTINGS_TUPLE_TYPE ||
						postings->termId != lex->termId)
					{
						UnlockReleaseBuffer(postBuf);
						ereport(ERROR,
								(errcode(ERRCODE_DATA_CORRUPTED),
								 errmsg("turbohybrid BM25 postings tuple is invalid")));
					}
						nextBlkno = postings->nextBlkno;
						nextOffno = postings->nextOffno;
						decoded = palloc(sizeof(TqHybridBm25Posting) *
										 postings->count);
						if (!TqHybridBm25DecodePostingsTuple(postings,
															ItemIdGetLength(postIid),
															decoded))
						{
							pfree(decoded);
							UnlockReleaseBuffer(postBuf);
							ereport(ERROR,
									(errcode(ERRCODE_DATA_CORRUPTED),
									 errmsg("turbohybrid BM25 postings payload is invalid")));
						}

						for (uint16 i = 0; i < postings->count; i++)
						{
							uint32		nodeId = decoded[i].nodeId;

							if (nodeId >= nodeCount || !nodeStates[nodeId].live ||
								docLens[nodeId] == 0)
								continue;

							TqHybridAppendBuildTerm(collector, nodeId, lex->termBytes,
													lex->termLen, decoded[i].tf);
						}

						pfree(decoded);
						UnlockReleaseBuffer(postBuf);
					postBlkno = nextBlkno;
					postOffno = nextOffno;
				}
			}

		UnlockReleaseBuffer(lexBuf);
		blkno = nextblkno;
		CHECK_FOR_INTERRUPTS();
	}
}

static void
TqHybridBm25CollectDelta(Relation index,
						 const TqHybridBm25MetaTupleData *meta,
						 TqHybridBm25Collector *collector,
						 const TqHybridNodeState *nodeStates,
						 bool *docSeen, uint32 nodeCount)
{
	BlockNumber blkno = meta->deltaStartBlkno;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);
		if (!TqHybridBm25PageIsKind(page, HNSW_PAGE_KIND_TQ_BM25_DELTA))
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("turbohybrid BM25 delta page has unexpected page kind")));
		}

		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			TqHybridBm25DeltaTuple tuple;
			char	   *termBytes;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqHybridBm25DeltaTuple) PageGetItem(page, iid);
			if (tuple->type != TQHYBRID_BM25_DELTA_TUPLE_TYPE ||
				tuple->nodeId >= nodeCount ||
				!nodeStates[tuple->nodeId].live)
				continue;

			if (!docSeen[tuple->nodeId])
			{
				TqHybridAppendBuildDoc(collector, tuple->nodeId,
									   &tuple->heaptid, tuple->docLen);
				docSeen[tuple->nodeId] = true;
			}

			termBytes = TqHybridBm25DeltaTermBytes(tuple);
			for (uint16 i = 0; i < tuple->termCount; i++)
			{
				TqHybridBm25DeltaTerm *term = &tuple->terms[i];

				if (term->termOffset + term->termLen > tuple->termBytesLen)
					ereport(ERROR,
							(errcode(ERRCODE_DATA_CORRUPTED),
							 errmsg("turbohybrid BM25 delta term offset is invalid")));

				TqHybridAppendBuildTerm(collector, tuple->nodeId,
										termBytes + term->termOffset,
										term->termLen, term->tf);
			}
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
		CHECK_FOR_INTERRUPTS();
	}
}

static void
TqHybridBm25UpdateCompactedMeta(Relation index,
								TqHybridBm25Collector *collector,
								const TqHybridBm25MetaTupleData *oldMeta)
{
	Buffer		metaBuf;
	Page		metaPage;
	TqHybridBm25MetaTuple metaTuple;
	GenericXLogState *xlogState;
	TqHybridOptions *opts = (TqHybridOptions *) index->rd_options;

	if (!TqHybridBm25ReadMetaForUpdate(index, &metaBuf, &metaPage,
									   &metaTuple, &xlogState))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 metadata is missing")));

	metaTuple->docCount = collector->docCount;
	if (opts != NULL && opts->bm25PrecomputeTfNorm)
		metaTuple->reserved2 |= TQHYBRID_BM25_META_FLAG_TFNORM_Q16;
	else
		metaTuple->reserved2 &= ~TQHYBRID_BM25_META_FLAG_TFNORM_Q16;
	metaTuple->totalDocLen = collector->totalDocLen;
	metaTuple->avgDocLen = collector->docCount == 0 ? 0.0f :
		(float4) ((double) collector->totalDocLen / (double) collector->docCount);
	metaTuple->termCount = collector->uniqueTerms;
	metaTuple->termTupleCount = collector->totalTermCount;
	metaTuple->maxDocLen = collector->maxDocLen;
	metaTuple->docStatsStartBlkno = collector->bm25DocStatsStartBlkno;
	metaTuple->lexiconStartBlkno = collector->bm25LexiconStartBlkno;
	metaTuple->postingsStartBlkno = collector->bm25PostingsStartBlkno;
	metaTuple->blockMaxStartBlkno = collector->bm25BlockMaxStartBlkno;
	metaTuple->deltaStartBlkno = InvalidBlockNumber;
	metaTuple->deltaGeneration = oldMeta->deltaGeneration + 1;
	metaTuple->deltaDocCount = 0;
	metaTuple->deltaTotalDocLen = 0;
	metaTuple->deltaTermCount = 0;
	metaTuple->postingsPages = collector->bm25PostingsPages;
	metaTuple->blockMaxPages = collector->bm25BlockMaxPages;
	metaTuple->deltaPages = 0;
	metaTuple->lastCompactionGeneration = metaTuple->deltaGeneration;
	metaTuple->compactionCount = oldMeta->compactionCount + 1;

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(metaBuf);
	UnlockReleaseBuffer(metaBuf);
}

bool
TqHybridBm25MaybeCompact(Relation index)
{
	MemoryContext ctx;
	MemoryContext oldCtx;
	TqHybridBm25MetaTupleData oldMeta;
	TqHybridBm25Collector collector;
	HnswMetaPageData graphMeta;
	TqHybridNodeState *nodeStates;
	uint32		nodeCount;
	uint32	   *docLens;
	bool	   *docSeen;
	TqHybridOptions *opts = (TqHybridOptions *) index->rd_options;
	int			threshold = opts != NULL ? opts->bm25DeltaCompactionThreshold : 25;
	uint32		uniqueTerms;

	if (!TqHybridBm25ReadMeta(index, &oldMeta, NULL) ||
		oldMeta.deltaDocCount == 0 ||
		!BlockNumberIsValid(oldMeta.deltaStartBlkno))
		return false;

	if ((uint64) oldMeta.deltaDocCount * 100 <
		(uint64) Max(oldMeta.docCount, 1) * (uint64) threshold)
		return false;

	ctx = AllocSetContextCreate(CurrentMemoryContext,
								"TurboHybrid BM25 delta compaction",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);

	nodeStates = TqHybridReadNodeStates(index, &graphMeta, &nodeCount);
	if (nodeCount == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid native graph metadata is missing during BM25 compaction")));

	docLens = palloc0(sizeof(uint32) * nodeCount);
	docSeen = palloc0(sizeof(bool) * nodeCount);
	TqHybridBm25ReadDocLens(index, &oldMeta, nodeCount, docLens);

	memset(&collector, 0, sizeof(collector));
	collector.index = index;
	collector.softBudget = (Size) maintenance_work_mem * 1024L;
	collector.allowSpill = true;
	collector.walLoggedWrites = RelationNeedsWAL(index);
	collector.tidNodeCount = nodeCount;

	for (uint32 nodeId = 0; nodeId < nodeCount; nodeId++)
	{
		if (nodeStates[nodeId].live && docLens[nodeId] > 0)
		{
			TqHybridAppendBuildDoc(&collector, nodeId,
								   &nodeStates[nodeId].tid, docLens[nodeId]);
			docSeen[nodeId] = true;
		}
	}

	TqHybridBm25CollectBasePostings(index, &oldMeta, &collector,
									nodeStates, docLens, nodeCount);
	TqHybridBm25CollectDelta(index, &oldMeta, &collector,
							 nodeStates, docSeen, nodeCount);

	if (collector.spillRunCount > 0)
	{
		TqHybridSpillTermRun(&collector);
		uniqueTerms = 0;
	}
	else
		uniqueTerms = TqHybridReduceUniqueTerms(&collector);
	collector.uniqueTerms = uniqueTerms;
	TqHybridBm25WriteBasePages(&collector);
	TqHybridBm25UpdateCompactedMeta(index, &collector, &oldMeta);
	TqHybridStoreDebugStats(index, &collector, collector.uniqueTerms);
	TqHybridBm25InvalidateCache(index);
	TqGraphInvalidateCaches(index);
	TqHybridCloseSpillRuns(&collector);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(ctx);
	return true;
}

static void
TqHybridStoreDebugStats(Relation index, TqHybridBm25Collector *collector,
						uint32 uniqueTerms)
{
	HASHCTL		ctl;
	TqHybridBm25DebugEntry *entry;
	bool		found;
	Oid			indexOid = RelationGetRelid(index);

	if (tqhybrid_bm25_debug == NULL)
	{
		memset(&ctl, 0, sizeof(ctl));
		ctl.keysize = sizeof(Oid);
		ctl.entrysize = sizeof(TqHybridBm25DebugEntry);
		ctl.hcxt = TopMemoryContext;
		tqhybrid_bm25_debug = hash_create("turbohybrid BM25 debug stats",
										  32, &ctl,
										  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}

	entry = hash_search(tqhybrid_bm25_debug,
						&indexOid,
						HASH_ENTER, &found);
	entry->indexOid = indexOid;
	entry->stats.indexOid = indexOid;
	entry->stats.docCount = collector->docCount;
	entry->stats.totalDocLen = collector->totalDocLen;
	entry->stats.uniqueTerms = uniqueTerms;
	entry->stats.termTupleCount = collector->totalTermCount;
	entry->stats.maxDocLen = collector->maxDocLen;
}

void
TqHybridBm25BuildCollect(Relation heap, Relation index, IndexInfo *indexInfo)
{
	MemoryContext ctx;
	MemoryContext oldCtx;
	TqHybridBm25Collector collector;
	HnswMetaPageData graphMeta;
	uint32		uniqueTerms;

	if (heap == NULL)
		return;

	ctx = AllocSetContextCreate(CurrentMemoryContext,
								"TurboHybrid BM25 build collector",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);

	memset(&collector, 0, sizeof(collector));
	collector.index = index;
	collector.softBudget = (Size) maintenance_work_mem * 1024L;
	collector.allowSpill = true;

	if (!TqGraphReadMeta(index, &graphMeta))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid BM25 collection requires native TurboQuant graph storage")));

	if (graphMeta.tqNodeCount > 0)
	{
		collector.tidNodes = TqHybridReadNodeMap(index, &collector.tidNodeCount);
		if (collector.tidNodeCount == 0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("turbohybrid native graph storage is missing during BM25 collection")));
	}

	TqHybridCheckBudget(&collector);
	(void) table_index_build_scan(heap, index, indexInfo,
								  true, true, TqHybridBm25BuildCallback,
								  &collector, NULL);
	if (collector.spillRunCount > 0)
	{
		TqHybridSpillTermRun(&collector);
		uniqueTerms = 0;
	}
	else
		uniqueTerms = TqHybridReduceUniqueTerms(&collector);
	collector.uniqueTerms = uniqueTerms;
	TqHybridBm25WriteStorage(&collector);
	TqHybridStoreDebugStats(index, &collector, collector.uniqueTerms);
	TqHybridCloseSpillRuns(&collector);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(ctx);
}

static bool
TqHybridBm25ReadMetaForUpdate(Relation index, Buffer *outBuf, Page *outPage,
							  TqHybridBm25MetaTuple *outTuple,
							  GenericXLogState **outXlogState)
{
	HnswMetaPageData graphMeta;
	BlockNumber metaBlkno;
	BlockNumber nblocks;
	OffsetNumber maxoff;

	*outBuf = InvalidBuffer;
	*outPage = NULL;
	*outTuple = NULL;
	*outXlogState = NULL;

	if (!TqGraphReadMeta(index, &graphMeta) ||
		!BlockNumberIsValid(graphMeta.tqBm25MetaStartBlkno))
		return false;

	metaBlkno = graphMeta.tqBm25MetaStartBlkno;
	nblocks = RelationGetNumberOfBlocks(index);
	if (metaBlkno >= nblocks)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 metadata pointer is invalid"),
				 errdetail("Metapage points to block %u, but the index has only %u blocks.",
						   metaBlkno, nblocks)));

	*outBuf = ReadBuffer(index, metaBlkno);
	LockBuffer(*outBuf, BUFFER_LOCK_EXCLUSIVE);
	if (RelationNeedsWAL(index))
	{
		*outXlogState = GenericXLogStart(index);
		*outPage = GenericXLogRegisterBuffer(*outXlogState, *outBuf, 0);
	}
	else
		*outPage = BufferGetPage(*outBuf);

	if (!TqHybridBm25PageIsKind(*outPage, HNSW_PAGE_KIND_TQ_BM25_META))
	{
		if (*outXlogState != NULL)
			GenericXLogAbort(*outXlogState);
		UnlockReleaseBuffer(*outBuf);
		*outBuf = InvalidBuffer;
		*outPage = NULL;
		*outXlogState = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 metadata pointer is invalid"),
				 errdetail("Metapage points to block %u, which is not a BM25 metadata page.",
						   metaBlkno)));
	}

	maxoff = PageGetMaxOffsetNumber(*outPage);
	for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
	{
		ItemId		iid = PageGetItemId(*outPage, off);
		TqHybridBm25MetaTuple tuple;

		if (!ItemIdIsUsed(iid))
			continue;

		tuple = (TqHybridBm25MetaTuple) PageGetItem(*outPage, iid);
		if (tuple->type == TQHYBRID_BM25_META_TUPLE_TYPE)
		{
			*outTuple = tuple;
			return true;
		}
	}

	if (*outXlogState != NULL)
		GenericXLogAbort(*outXlogState);
	UnlockReleaseBuffer(*outBuf);
	*outBuf = InvalidBuffer;
	*outPage = NULL;
	*outXlogState = NULL;
	ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
			 errmsg("turbohybrid BM25 metadata tuple is missing"),
			 errdetail("Metapage points to BM25 metadata block %u, but no metadata tuple was found.",
					   metaBlkno)));
}

static uint32
TqHybridBm25CountChainPagesAndTail(Relation index, BlockNumber startBlkno,
								   uint16 pageKind, BlockNumber *tailBlkno)
{
	uint32		count = 0;
	BlockNumber blkno = startBlkno;

	*tailBlkno = InvalidBlockNumber;
	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		BlockNumber nextblkno;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);
		if (!TqHybridBm25PageIsKind(page, pageKind))
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		nextblkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
		count++;
		*tailBlkno = blkno;
		blkno = nextblkno;
	}

	return count;
}

void
TqHybridBm25AppendDelta(Relation index, uint32 nodeId,
						ItemPointer heapTid, Datum tsvectorDatum)
{
	MemoryContext ctx;
	MemoryContext oldCtx;
	TqHybridBm25Collector collector;
	TSVector	vector;
	bool		mustFree;
	TqHybridBm25DeltaTuple delta;
	char	   *deltaBytes;
	Size		deltaSize;
	Size		maxItemSize;
	BlockNumber deltaStart;
	BlockNumber deltaTail;
	BlockNumber appendStart;
	BlockNumber deltaBlkno = InvalidBlockNumber;
	uint32		deltaPages;
	bool		deltaCursorHit;
	Buffer		metaBuf;
	Page		metaPage;
	TqHybridBm25MetaTuple metaTuple;
	GenericXLogState *xlogState;

	(void) metaPage;

	ctx = AllocSetContextCreate(CurrentMemoryContext,
								"TurboHybrid BM25 delta append",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);

	memset(&collector, 0, sizeof(collector));
	collector.index = index;
	collector.softBudget = (Size) maintenance_work_mem * 1024L;
	vector = TqHybridDetoastTSVector(tsvectorDatum, &mustFree);
	TqHybridValidateTSVector(vector);
	TqHybridCollectVectorTerms(&collector, nodeId, vector);
	if (collector.termCount > PG_UINT16_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid BM25 delta document has too many distinct terms"),
				 errdetail("Delta storage supports at most %u distinct terms per inserted document, but this document has %u.",
						   PG_UINT16_MAX, collector.termCount),
				 errhint("Rebuild the index after loading very large text documents, or reduce the tsvector vocabulary for a single row.")));

	deltaSize = TqHybridBm25DeltaTupleSize((uint16) collector.termCount,
										   collector.termBytesUsed);
	maxItemSize = BLCKSZ - SizeOfPageHeaderData -
		MAXALIGN(sizeof(HnswPageOpaqueData));
	if (deltaSize > maxItemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid BM25 delta tuple exceeds page size"),
				 errdetail("Delta tuple for node %u has %u terms, %u bytes of term text, and %zu bytes total; the maximum index tuple size is %zu bytes.",
						   nodeId, collector.termCount,
						   collector.termBytesUsed, deltaSize, maxItemSize),
				 errhint("Rebuild the index after loading very large text documents, or reduce the tsvector vocabulary for a single row.")));

	if (!TqHybridBm25ReadMetaForUpdate(index, &metaBuf, &metaPage,
									   &metaTuple, &xlogState))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 metadata is missing")));
	delta = palloc0(deltaSize);
	delta->type = TQHYBRID_BM25_DELTA_TUPLE_TYPE;
	delta->termCount = collector.termCount;
	delta->nodeId = nodeId;
	delta->heaptid = *heapTid;
	delta->docLen = TqHybridDocLen(vector);
	delta->termBytesLen = collector.termBytesUsed;
	deltaBytes = TqHybridBm25DeltaTermBytes(delta);
	if (collector.termBytesUsed > 0)
		memcpy(deltaBytes, collector.termBytes, collector.termBytesUsed);
	for (uint32 i = 0; i < collector.termCount; i++)
	{
		delta->terms[i].termHash = collector.terms[i].termHash;
		delta->terms[i].termOffset = collector.terms[i].termOffset;
		delta->terms[i].tf = collector.terms[i].tf;
		delta->terms[i].termLen = collector.terms[i].termLen;
	}

	deltaStart = metaTuple->deltaStartBlkno;
	deltaCursorHit =
		tqhybrid_bm25_delta_cursor_index == RelationGetRelid(index) &&
		tqhybrid_bm25_delta_cursor_relfilenumber == TqGraphRelFileNumber(index) &&
		tqhybrid_bm25_delta_cursor_generation == metaTuple->deltaGeneration &&
		tqhybrid_bm25_delta_cursor_start == metaTuple->deltaStartBlkno;
	if (deltaCursorHit)
	{
		deltaTail = tqhybrid_bm25_delta_cursor_tail;
		deltaPages = tqhybrid_bm25_delta_cursor_pages;
	}
	else
		deltaPages = TqHybridBm25CountChainPagesAndTail(index, deltaStart,
														 HNSW_PAGE_KIND_TQ_BM25_DELTA,
														 &deltaTail);
	if (xlogState != NULL)
		GenericXLogAbort(xlogState);
	UnlockReleaseBuffer(metaBuf);

	appendStart = BlockNumberIsValid(deltaTail) ? deltaTail : deltaStart;
	(void) TqGraphAppendTuple(index, MAIN_FORKNUM, &appendStart,
							  HNSW_PAGE_KIND_TQ_BM25_DELTA,
							  (Item) delta, deltaSize,
							  HNSW_GRAPH_OP_ELEMENT_INSERT,
							  &deltaBlkno);
	if (!BlockNumberIsValid(deltaStart))
	{
		deltaStart = appendStart;
		deltaPages = 1;
	}
	else if (!BlockNumberIsValid(deltaTail))
		deltaPages = TqHybridBm25CountChainPagesAndTail(index, deltaStart,
														 HNSW_PAGE_KIND_TQ_BM25_DELTA,
														 &deltaTail);
	else if (deltaBlkno != deltaTail)
		deltaPages++;
	deltaTail = deltaBlkno;

	if (!TqHybridBm25ReadMetaForUpdate(index, &metaBuf, &metaPage,
									   &metaTuple, &xlogState))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 metadata is missing")));

	metaTuple->deltaStartBlkno = deltaStart;
	metaTuple->deltaGeneration++;
	metaTuple->deltaDocCount++;
	metaTuple->deltaTotalDocLen += delta->docLen;
	metaTuple->deltaTermCount += delta->termCount;
	metaTuple->deltaPages = deltaPages;
	tqhybrid_bm25_delta_cursor_index = RelationGetRelid(index);
	tqhybrid_bm25_delta_cursor_relfilenumber = TqGraphRelFileNumber(index);
	tqhybrid_bm25_delta_cursor_start = metaTuple->deltaStartBlkno;
	tqhybrid_bm25_delta_cursor_tail = deltaTail;
	tqhybrid_bm25_delta_cursor_generation = metaTuple->deltaGeneration;
	tqhybrid_bm25_delta_cursor_pages = metaTuple->deltaPages;

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(metaBuf);
	UnlockReleaseBuffer(metaBuf);

	TqHybridBm25InvalidateCache(index);
	TqGraphInvalidateCaches(index);

	if (mustFree)
		pfree(vector);
	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(ctx);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tq_debug_bm25_stats);
static bool
TqHybridBm25PageIsKind(Page page, uint16 pageKind)
{
	HnswPageOpaque opaque = HnswPageGetOpaque(page);

	return opaque->page_id == HNSW_PAGE_ID &&
		(opaque->pageKind & HNSW_PAGE_KIND_MASK) == pageKind;
}

static bool
TqHybridBm25ReadMeta(Relation index, TqHybridBm25MetaTupleData *meta,
					 BlockNumber *metaBlkno)
{
	HnswMetaPageData graphMeta;
	BlockNumber blkno;
	BlockNumber nblocks;
	Buffer		buf;
	Page		page;
	OffsetNumber maxoff;

	if (!TqGraphReadMeta(index, &graphMeta) ||
		!BlockNumberIsValid(graphMeta.tqBm25MetaStartBlkno))
		return false;

	blkno = graphMeta.tqBm25MetaStartBlkno;
	nblocks = RelationGetNumberOfBlocks(index);
	if (blkno >= nblocks)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 metadata pointer is invalid"),
				 errdetail("Metapage points to block %u, but the index has only %u blocks.",
						   blkno, nblocks)));

	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	if (!TqHybridBm25PageIsKind(page, HNSW_PAGE_KIND_TQ_BM25_META))
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 metadata pointer is invalid"),
				 errdetail("Metapage points to block %u, which is not a BM25 metadata page.",
						   blkno)));
	}

	maxoff = PageGetMaxOffsetNumber(page);
	for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
	{
		ItemId		iid = PageGetItemId(page, off);
		TqHybridBm25MetaTuple tuple;

		if (!ItemIdIsUsed(iid))
			continue;

		tuple = (TqHybridBm25MetaTuple) PageGetItem(page, iid);
		if (tuple->type != TQHYBRID_BM25_META_TUPLE_TYPE)
			continue;

		*meta = *tuple;
		if (metaBlkno != NULL)
			*metaBlkno = blkno;
		UnlockReleaseBuffer(buf);
		return true;
	}

	UnlockReleaseBuffer(buf);
	ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
			 errmsg("turbohybrid BM25 metadata tuple is missing"),
			 errdetail("Metapage points to BM25 metadata block %u, but no metadata tuple was found.",
					   blkno)));
}

static void
TqHybridBm25SetMetaBlock(Relation index, BlockNumber metaBlkno)
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
	if (metap->magicNumber != HNSW_MAGIC_NUMBER ||
		metap->storageKind != HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE)
	{
		if (xlogState != NULL)
			GenericXLogAbort(xlogState);
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid native graph metadata is missing")));
	}

	metap->tqBm25MetaStartBlkno = metaBlkno;
	HnswMarkPageGraphOp(page, HNSW_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
	HnswLogGraphWalRecord(index, MAIN_FORKNUM, HNSW_METAPAGE_BLKNO,
						  HNSW_GRAPH_OP_META_UPDATE);
}

static bool
TqHybridBm25FindLexiconEntry(Relation index, const TqHybridBm25MetaTupleData *meta,
							 const char *term, uint16 termLen,
							 TqHybridBm25LexiconEntryData *entry)
{
	uint64		termHash = TqHybridHashTerm(term, termLen);
	BlockNumber blkno = meta->lexiconStartBlkno;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);
		if (!TqHybridBm25PageIsKind(page, HNSW_PAGE_KIND_TQ_BM25_LEXICON))
		{
			UnlockReleaseBuffer(buf);
			return false;
		}

		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			TqHybridBm25LexiconEntry tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqHybridBm25LexiconEntry) PageGetItem(page, iid);
			if (tuple->type == TQHYBRID_BM25_LEXICON_TUPLE_TYPE &&
				tuple->termHash == termHash &&
				tuple->termLen == termLen &&
				memcmp(tuple->termBytes, term, termLen) == 0)
			{
				memcpy(entry, tuple,
					   offsetof(TqHybridBm25LexiconEntryData, termBytes));
				UnlockReleaseBuffer(buf);
				return true;
			}
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
	}

	return false;
}

static uint32
TqHybridBm25ReadPostingCount(Relation index,
							 const TqHybridBm25LexiconEntryData *entry)
{
	uint32		count = 0;
	BlockNumber blkno = entry->postingsBlkno;
	OffsetNumber offno = entry->postingsOffno;
	uint32		chunkLimit = Max(entry->postingsChunkCount, 1);

	if (!BlockNumberIsValid(entry->postingsBlkno) ||
		!OffsetNumberIsValid(entry->postingsOffno))
		return 0;

	for (uint32 chunkNo = 0;
		 chunkNo < chunkLimit && BlockNumberIsValid(blkno) &&
		 OffsetNumberIsValid(offno);
		 chunkNo++)
	{
		Buffer		buf;
		Page		page;
		ItemId		iid;
		TqHybridBm25PostingsTuple tuple;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		if (!TqHybridBm25PageIsKind(page, HNSW_PAGE_KIND_TQ_BM25_POSTINGS) ||
			offno > PageGetMaxOffsetNumber(page))
		{
			UnlockReleaseBuffer(buf);
			return count;
		}

		iid = PageGetItemId(page, offno);
		if (ItemIdIsUsed(iid))
		{
			tuple = (TqHybridBm25PostingsTuple) PageGetItem(page, iid);
			if (tuple->type == TQHYBRID_BM25_POSTINGS_TUPLE_TYPE)
			{
				count += tuple->count;
				blkno = tuple->nextBlkno;
				offno = tuple->nextOffno;
			}
			else
			{
				blkno = InvalidBlockNumber;
				offno = InvalidOffsetNumber;
			}
		}
		else
		{
			blkno = InvalidBlockNumber;
			offno = InvalidOffsetNumber;
		}

		UnlockReleaseBuffer(buf);
	}

	return count;
}

static void
TqHybridBm25ReadPostingEncodingCounts(Relation index,
									  const TqHybridBm25LexiconEntryData *entry,
									  uint32 *offset16Count,
									  uint32 *delta16Count,
									  uint32 *varintCount,
									  uint32 *tfNormCount,
									  uint32 *unknownCount)
{
	BlockNumber blkno = entry->postingsBlkno;
	OffsetNumber offno = entry->postingsOffno;
	uint32		chunkLimit = Max(entry->postingsChunkCount, 1);

	*offset16Count = 0;
	*delta16Count = 0;
	*varintCount = 0;
	*tfNormCount = 0;
	*unknownCount = 0;
	if (!BlockNumberIsValid(entry->postingsBlkno) ||
		!OffsetNumberIsValid(entry->postingsOffno))
		return;

	for (uint32 chunkNo = 0;
		 chunkNo < chunkLimit && BlockNumberIsValid(blkno) &&
		 OffsetNumberIsValid(offno);
		 chunkNo++)
	{
		Buffer		buf;
		Page		page;
		ItemId		iid;
		TqHybridBm25PostingsTuple tuple;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		if (!TqHybridBm25PageIsKind(page, HNSW_PAGE_KIND_TQ_BM25_POSTINGS) ||
			offno > PageGetMaxOffsetNumber(page))
		{
			UnlockReleaseBuffer(buf);
			return;
		}

		iid = PageGetItemId(page, offno);
		if (!ItemIdIsUsed(iid))
		{
			UnlockReleaseBuffer(buf);
			return;
		}

		tuple = (TqHybridBm25PostingsTuple) PageGetItem(page, iid);
		if (tuple->type != TQHYBRID_BM25_POSTINGS_TUPLE_TYPE)
		{
			UnlockReleaseBuffer(buf);
			return;
		}

		if ((tuple->encoding & TQHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16) != 0)
			(*tfNormCount)++;
		switch (tuple->encoding & TQHYBRID_BM25_POSTINGS_ENCODING_MASK)
		{
			case TQHYBRID_BM25_POSTINGS_ENCODING_OFFSET16:
				(*offset16Count)++;
				break;
			case TQHYBRID_BM25_POSTINGS_ENCODING_DELTA16:
				(*delta16Count)++;
				break;
			case TQHYBRID_BM25_POSTINGS_ENCODING_DELTA_VARINT:
				(*varintCount)++;
				break;
			default:
				(*unknownCount)++;
				break;
		}
		blkno = tuple->nextBlkno;
		offno = tuple->nextOffno;
		UnlockReleaseBuffer(buf);
	}
}

Datum
tq_debug_bm25_stats(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Relation	index;
	TqHybridBm25MetaTupleData meta;
	BlockNumber metaBlkno = InvalidBlockNumber;
	StringInfoData json;

	index = index_open(indexOid, AccessShareLock);
	if (!TqHybridBm25ReadMeta(index, &meta, &metaBlkno))
	{
		index_close(index, AccessShareLock);
		PG_RETURN_NULL();
	}

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"doc_count\":%u,"
					 "\"total_doc_len\":" UINT64_FORMAT ","
					 "\"unique_terms\":%u,"
					 "\"term_count\":%u,"
					 "\"term_tuple_count\":%u,"
					 "\"max_doc_len\":%u,"
					 "\"avg_doc_len_scaled\":%u,"
					 "\"bm25_precompute_tf_norm\":%s,"
					 "\"meta_blkno\":%u,"
					 "\"docstats_start_blkno\":%u,"
					 "\"lexicon_start_blkno\":%u,"
					 "\"postings_start_blkno\":%u,"
					 "\"blockmax_start_blkno\":%u,"
					 "\"delta_start_blkno\":%u,"
					 "\"delta_generation\":" UINT64_FORMAT ","
					 "\"delta_doc_count\":%u,"
					 "\"delta_term_count\":%u,"
					 "\"postings_pages\":%u,"
					 "\"blockmax_pages\":%u,"
					 "\"delta_pages\":%u,"
					 "\"last_compaction_generation\":" UINT64_FORMAT ","
					 "\"compaction_count\":%u}",
					 meta.docCount,
					 meta.totalDocLen,
					 meta.termCount,
					 meta.termCount,
					 meta.termTupleCount,
					 meta.maxDocLen,
					 meta.docCount == 0 ? 0 :
					 (uint32) (((double) meta.totalDocLen * 1000.0) /
							   (double) meta.docCount),
					 (meta.reserved2 & TQHYBRID_BM25_META_FLAG_TFNORM_Q16) != 0 ?
					 "true" : "false",
					 metaBlkno,
					 meta.docStatsStartBlkno,
					 meta.lexiconStartBlkno,
					 meta.postingsStartBlkno,
					 meta.blockMaxStartBlkno,
					 meta.deltaStartBlkno,
					 meta.deltaGeneration,
					 meta.deltaDocCount,
					 meta.deltaTermCount,
					 meta.postingsPages,
					 meta.blockMaxPages,
					 meta.deltaPages,
					 meta.lastCompactionGeneration,
					 meta.compactionCount);

	index_close(index, AccessShareLock);
	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tq_debug_bm25_term_stats);
Datum
tq_debug_bm25_term_stats(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	text	   *termText = PG_GETARG_TEXT_PP(1);
	char	   *term = VARDATA_ANY(termText);
	int			termLen = VARSIZE_ANY_EXHDR(termText);
	Relation	index;
	TqHybridBm25MetaTupleData meta;
	TqHybridBm25LexiconEntryData entry;
	uint32		postingCount;
	uint32		offset16Count;
	uint32		delta16Count;
	uint32		varintCount;
	uint32		tfNormCount;
	uint32		unknownCount;
	StringInfoData json;

	if (termLen < 0 || termLen > PG_UINT16_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid BM25 term is too large")));

	index = index_open(indexOid, AccessShareLock);
	if (!TqHybridBm25ReadMeta(index, &meta, NULL) ||
		!TqHybridBm25FindLexiconEntry(index, &meta, term, (uint16) termLen,
									  &entry))
	{
		index_close(index, AccessShareLock);
		PG_RETURN_NULL();
	}

	postingCount = TqHybridBm25ReadPostingCount(index, &entry);
	TqHybridBm25ReadPostingEncodingCounts(index, &entry, &offset16Count,
										  &delta16Count, &varintCount,
										  &tfNormCount, &unknownCount);
	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"term_id\":%u,"
					 "\"df\":%u,"
					 "\"cf\":%u,"
						 "\"posting_count\":%u,"
						 "\"postings_chunk_count\":%u,"
						 "\"postings_blkno\":%u,"
						 "\"postings_offno\":%u,"
					 "\"postings_bytes\":%u,"
					 "\"posting_encoding_counts\":{"
					 "\"offset16\":%u,"
					 "\"delta16\":%u,"
					 "\"varint\":%u,"
					 "\"tfnorm_q16\":%u,"
					 "\"unknown\":%u},"
					 "\"bm25_precompute_tf_norm\":%s,"
					 "\"blockmax_blkno\":%u,"
					 "\"blockmax_offno\":%u}",
					 entry.termId,
					 entry.df,
						 entry.cf,
						 postingCount,
						 entry.postingsChunkCount,
						 entry.postingsBlkno,
					 entry.postingsOffno,
					 entry.postingsBytes,
					 offset16Count,
					 delta16Count,
					 varintCount,
					 tfNormCount,
					 unknownCount,
					 (meta.reserved2 & TQHYBRID_BM25_META_FLAG_TFNORM_Q16) != 0 ?
					 "true" : "false",
					 entry.blockMaxBlkno,
					 entry.blockMaxOffno);

	index_close(index, AccessShareLock);
	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}
