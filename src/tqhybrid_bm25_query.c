#include "postgres.h"

#include <math.h>
#include <string.h>

#if !defined(TQ_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#endif
#if !defined(TQ_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#include <immintrin.h>
#if !defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
#define TQHYBRID_BM25_AVX2_TARGET __attribute__((target("avx2")))
#else
#define TQHYBRID_BM25_AVX2_TARGET
#endif
#else
#define TQHYBRID_BM25_AVX2_TARGET
#endif

#include "access/genam.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "tsearch/ts_type.h"
#include "utils/builtins.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "hnsw.h"
#include "tqgraph.h"
#include "tqhybrid.h"
#include "tqhybrid_bm25.h"

int			tqhybrid_last_bm25_decode_kernel = TQHYBRID_BM25_KERNEL_SCALAR;
int			tqhybrid_last_bm25_score_kernel = TQHYBRID_BM25_KERNEL_SCALAR;
uint64		tqhybrid_last_bm25_simd_blocks = 0;
uint64		tqhybrid_last_bm25_scalar_tail_postings = 0;

const char *
TqHybridBm25KernelName(int kernel)
{
	switch (kernel)
	{
		case TQHYBRID_BM25_KERNEL_NEON:
			return "neon";
		case TQHYBRID_BM25_KERNEL_AVX2:
			return "avx2";
		case TQHYBRID_BM25_KERNEL_SCALAR:
		default:
			return "scalar";
	}
}

typedef struct TqHybridBm25QueryTerm
{
	char	   *term;
	uint16		termLen;
	uint64		matchBit;
	bool		hasLexicon;
	uint32		baseDf;
	uint32		deltaDf;
	TqHybridBm25LexiconEntryData lexicon;
} TqHybridBm25QueryTerm;

typedef struct TqHybridBm25NodeScore
{
	uint32		nodeId;
	float8		score;
} TqHybridBm25NodeScore;

typedef struct TqHybridBm25AccumulatorEntry
{
	uint32		nodeId;
	float8		score;
	uint32		docLen;
	ItemPointerData heaptid;
	uint64		matchedTerms;
	bool		hasDeltaDoc;
	int32		heapIndex;
} TqHybridBm25AccumulatorEntry;

typedef struct TqHybridBm25Accumulator
{
	HTAB	   *entries;
	TqHybridBm25NodeScore *touched;
	uint32		touchedCount;
	uint32		touchedCapacity;
	TqHybridBm25AccumulatorEntry **topHeap;
	uint32		topHeapCount;
	uint32		topHeapCapacity;
	double		threshold;
	MemoryContext memoryContext;
	TqHybridBm25QueryStats *stats;
} TqHybridBm25Accumulator;

typedef struct TqHybridBm25PostingIterator
{
	Relation	index;
	TqHybridBm25QueryTerm *term;
	double		idf;
	double		avgDocLen;
	const uint32 *docLens;
	const bool *liveNodes;
	uint32		nodeCount;
	uint32		chunkLimit;
	uint32		chunkNo;
	BlockNumber blkno;
	OffsetNumber offno;
	TqHybridBm25Posting *postings;
	uint16		postingsCapacity;
	uint16		count;
	uint16		pos;
	uint16		maxTf;
	uint32		lastNodeId;
	BlockNumber nextBlkno;
	OffsetNumber nextOffno;
	bool		valid;
	MemoryContext memoryContext;
	TqHybridBm25QueryStats *stats;
} TqHybridBm25PostingIterator;

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

typedef struct TqHybridBm25CacheLexiconEntry
{
	uint16		termLen;
	uint64		termHash;
	uint32		termId;
	uint32		df;
	uint32		cf;
	BlockNumber postingsBlkno;
	OffsetNumber postingsOffno;
	uint32		postingsChunkCount;
	uint32		postingsBytes;
	BlockNumber blockMaxBlkno;
	OffsetNumber blockMaxOffno;
	uint32		termOffset;
	char	   *termBytes;
} TqHybridBm25CacheLexiconEntry;

typedef struct TqHybridBm25DeltaCachePosting
{
	uint32		nodeId;
	uint16		tf;
	uint16		reserved;
	uint32		docLen;
	ItemPointerData heaptid;
} TqHybridBm25DeltaCachePosting;

typedef struct TqHybridBm25DeltaCacheEntry
{
	uint16		termLen;
	uint64		termHash;
	uint32		df;
	uint32		postingCount;
	char	   *termBytes;
	TqHybridBm25DeltaCachePosting *postings;
} TqHybridBm25DeltaCacheEntry;

typedef struct TqHybridBm25DeltaBuildEntry
{
	uint16		termLen;
	uint64		termHash;
	uint32		nodeId;
	uint16		tf;
	uint16		reserved;
	uint32		docLen;
	ItemPointerData heaptid;
	char	   *termBytes;
} TqHybridBm25DeltaBuildEntry;

typedef struct TqHybridBm25Cache
{
	Oid			relid;
	Oid			relfilenumber;
	uint16		graphFlags;
	uint32		nodeCount;
	uint32		docCount;
	uint32		termCount;
	uint64		deltaGeneration;
	uint64		lastCompactionGeneration;
	BlockNumber docStatsStartBlkno;
	BlockNumber lexiconStartBlkno;
	BlockNumber postingsStartBlkno;
	BlockNumber blockMaxStartBlkno;
	BlockNumber deltaStartBlkno;
	uint32	   *docLens;
	ItemPointerData *heapTids;
	bool	   *liveNodes;
	bool		docStatsLoaded;
	bool		livenessLoaded;
	TqHybridBm25CacheLexiconEntry *lexicon;
	uint32		lexiconCount;
	char	   *termBytesArena;
	uint32		termBytesArenaUsed;
	uint32		termBytesArenaCapacity;
	bool		deltaCacheBuilt;
	TqHybridBm25DeltaCacheEntry *deltaTerms;
	uint32		deltaTermCount;
	uint32		deltaPostingCount;
	uint64		deltaCacheBytes;
	MemoryContext ctx;
	struct TqHybridBm25Cache *next;
} TqHybridBm25Cache;

static TqHybridBm25Cache *tqhybrid_bm25_cache_list = NULL;

static int TqHybridBm25FindQueryTerm(TqHybridBm25QueryTerm *terms,
									 int termCount, const char *term,
									 uint16 termLen);

static uint64
TqHybridBm25ElapsedUs(instr_time start)
{
	instr_time	end;

	INSTR_TIME_SET_CURRENT(end);
	INSTR_TIME_SUBTRACT(end, start);
	return (uint64) INSTR_TIME_GET_MICROSEC(end);
}

static uint64
TqHybridBm25HashTerm(const char *term, uint16 len)
{
	uint64		hash = UINT64CONST(1469598103934665603);

	for (uint16 i = 0; i < len; i++)
	{
		hash ^= (unsigned char) term[i];
		hash *= UINT64CONST(1099511628211);
	}

	return hash;
}

static bool
TqHybridBm25PageIsKind(Page page, uint16 pageKind)
{
	HnswPageOpaque opaque = HnswPageGetOpaque(page);

	return opaque->page_id == HNSW_PAGE_ID &&
		(opaque->pageKind & HNSW_PAGE_KIND_MASK) == pageKind;
}

static void
TqHybridBm25AccumulatorInit(TqHybridBm25Accumulator *acc,
							MemoryContext memoryContext, uint32 initialSize,
							uint32 topK, TqHybridBm25QueryStats *stats)
{
	HASHCTL		ctl;

	memset(acc, 0, sizeof(*acc));
	memset(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(uint32);
	ctl.entrysize = sizeof(TqHybridBm25AccumulatorEntry);
	ctl.hcxt = memoryContext;

	acc->entries = hash_create("turbohybrid BM25 query accumulator",
							   Max(initialSize, 16),
							   &ctl,
							   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	acc->touchedCapacity = Max(initialSize, 16);
	acc->touched = MemoryContextAllocZero(memoryContext,
										  sizeof(TqHybridBm25NodeScore) *
										  acc->touchedCapacity);
	acc->topHeapCapacity = topK;
	if (topK > 0)
		acc->topHeap = MemoryContextAllocZero(memoryContext,
											  sizeof(TqHybridBm25AccumulatorEntry *) *
											  topK);
	acc->memoryContext = memoryContext;
	acc->stats = stats;
}

static bool
TqHybridBm25AccumulatorHeapLess(TqHybridBm25AccumulatorEntry *a,
								TqHybridBm25AccumulatorEntry *b)
{
	if (a->score < b->score)
		return true;
	if (a->score > b->score)
		return false;
	return a->nodeId > b->nodeId;
}

static void
TqHybridBm25AccumulatorHeapSwap(TqHybridBm25Accumulator *acc, uint32 a,
								uint32 b)
{
	TqHybridBm25AccumulatorEntry *tmp = acc->topHeap[a];

	acc->topHeap[a] = acc->topHeap[b];
	acc->topHeap[b] = tmp;
	acc->topHeap[a]->heapIndex = a;
	acc->topHeap[b]->heapIndex = b;
}

static void
TqHybridBm25AccumulatorHeapSiftUp(TqHybridBm25Accumulator *acc, uint32 pos)
{
	while (pos > 0)
	{
		uint32		parent = (pos - 1) / 2;

		if (!TqHybridBm25AccumulatorHeapLess(acc->topHeap[pos],
											 acc->topHeap[parent]))
			break;
		TqHybridBm25AccumulatorHeapSwap(acc, pos, parent);
		pos = parent;
	}
}

static void
TqHybridBm25AccumulatorHeapSiftDown(TqHybridBm25Accumulator *acc, uint32 pos)
{
	for (;;)
	{
		uint32		left = pos * 2 + 1;
		uint32		right = left + 1;
		uint32		smallest = pos;

		if (left < acc->topHeapCount &&
			TqHybridBm25AccumulatorHeapLess(acc->topHeap[left],
											acc->topHeap[smallest]))
			smallest = left;
		if (right < acc->topHeapCount &&
			TqHybridBm25AccumulatorHeapLess(acc->topHeap[right],
											acc->topHeap[smallest]))
			smallest = right;
		if (smallest == pos)
			break;
		TqHybridBm25AccumulatorHeapSwap(acc, pos, smallest);
		pos = smallest;
	}
}

static void
TqHybridBm25AccumulatorRefreshThreshold(TqHybridBm25Accumulator *acc)
{
	double		oldThreshold = acc->threshold;

	if (acc->topHeapCapacity > 0 &&
		acc->topHeapCount == acc->topHeapCapacity)
		acc->threshold = acc->topHeap[0]->score;
	else
		acc->threshold = 0.0;
	if (acc->stats != NULL &&
		acc->threshold > 0.0 &&
		acc->threshold != oldThreshold)
		acc->stats->wandThresholdUpdates++;
}

static void
TqHybridBm25AccumulatorUpdateTopK(TqHybridBm25Accumulator *acc,
								  TqHybridBm25AccumulatorEntry *entry)
{
	if (acc->topHeapCapacity == 0)
		return;

	if (entry->heapIndex >= 0)
	{
		/*
		 * Scores only increase.  This is a min-heap, so an existing entry can
		 * only move away from the root after a score update.
		 */
		TqHybridBm25AccumulatorHeapSiftDown(acc, (uint32) entry->heapIndex);
	}
	else if (acc->topHeapCount < acc->topHeapCapacity)
	{
		entry->heapIndex = acc->topHeapCount;
		acc->topHeap[acc->topHeapCount++] = entry;
		TqHybridBm25AccumulatorHeapSiftUp(acc, (uint32) entry->heapIndex);
	}
	else if (entry->score > acc->topHeap[0]->score)
	{
		acc->topHeap[0]->heapIndex = -1;
		entry->heapIndex = 0;
		acc->topHeap[0] = entry;
		TqHybridBm25AccumulatorHeapSiftDown(acc, 0);
		if (acc->stats != NULL)
			acc->stats->wandHeapReplacements++;
	}

	TqHybridBm25AccumulatorRefreshThreshold(acc);
}

static TqHybridBm25AccumulatorEntry *
TqHybridBm25AccumulatorLookup(TqHybridBm25Accumulator *acc, uint32 nodeId,
							  bool create)
{
	TqHybridBm25AccumulatorEntry *entry;
	bool		found;

	entry = hash_search(acc->entries, &nodeId,
						create ? HASH_ENTER : HASH_FIND, &found);
	if (entry == NULL || found || !create)
		return entry;

	entry->nodeId = nodeId;
	entry->score = 0.0;
	entry->docLen = 0;
	ItemPointerSetInvalid(&entry->heaptid);
	entry->matchedTerms = 0;
	entry->hasDeltaDoc = false;
	entry->heapIndex = -1;

	if (acc->touchedCount >= acc->touchedCapacity)
	{
		acc->touchedCapacity *= 2;
		acc->touched = repalloc(acc->touched,
								 sizeof(TqHybridBm25NodeScore) *
								 acc->touchedCapacity);
	}
	acc->touched[acc->touchedCount].nodeId = nodeId;
	acc->touched[acc->touchedCount].score = 0.0;
	acc->touchedCount++;

	return entry;
}

static void
TqHybridBm25AccumulatorAddTermScore(TqHybridBm25Accumulator *acc,
									uint32 nodeId, float8 score,
									uint64 matchBit)
{
	TqHybridBm25AccumulatorEntry *entry;

	entry = TqHybridBm25AccumulatorLookup(acc, nodeId, true);
	entry->score += score;
	entry->matchedTerms |= matchBit;
	TqHybridBm25AccumulatorUpdateTopK(acc, entry);
}

static int
TqHybridBm25CacheLexiconCompare(const void *a, const void *b)
{
	const TqHybridBm25CacheLexiconEntry *la =
		(const TqHybridBm25CacheLexiconEntry *) a;
	const TqHybridBm25CacheLexiconEntry *lb =
		(const TqHybridBm25CacheLexiconEntry *) b;
	int			cmp;

	if (la->termHash != lb->termHash)
		return la->termHash < lb->termHash ? -1 : 1;
	if (la->termLen != lb->termLen)
		return la->termLen < lb->termLen ? -1 : 1;

	cmp = memcmp(la->termBytes, lb->termBytes, la->termLen);
	if (cmp != 0)
		return cmp;

	return (la->termId > lb->termId) -
		(la->termId < lb->termId);
}

static int
TqHybridBm25DeltaBuildCompare(const void *a, const void *b)
{
	const TqHybridBm25DeltaBuildEntry *la =
		(const TqHybridBm25DeltaBuildEntry *) a;
	const TqHybridBm25DeltaBuildEntry *lb =
		(const TqHybridBm25DeltaBuildEntry *) b;
	int			cmp;

	if (la->termHash != lb->termHash)
		return la->termHash < lb->termHash ? -1 : 1;
	if (la->termLen != lb->termLen)
		return la->termLen < lb->termLen ? -1 : 1;
	cmp = memcmp(la->termBytes, lb->termBytes, la->termLen);
	if (cmp != 0)
		return cmp;
	return (la->nodeId > lb->nodeId) - (la->nodeId < lb->nodeId);
}

static int
TqHybridBm25DeltaCacheCompare(const void *a, const void *b)
{
	const TqHybridBm25DeltaCacheEntry *la =
		(const TqHybridBm25DeltaCacheEntry *) a;
	const TqHybridBm25DeltaCacheEntry *lb =
		(const TqHybridBm25DeltaCacheEntry *) b;
	int			cmp;

	if (la->termHash != lb->termHash)
		return la->termHash < lb->termHash ? -1 : 1;
	if (la->termLen != lb->termLen)
		return la->termLen < lb->termLen ? -1 : 1;
	cmp = memcmp(la->termBytes, lb->termBytes, la->termLen);
	if (cmp != 0)
		return cmp;
	return 0;
}

static char *
TqHybridBm25DeltaTermBytes(TqHybridBm25DeltaTuple tuple)
{
	return ((char *) tuple) + offsetof(TqHybridBm25DeltaTupleData, terms) +
		sizeof(TqHybridBm25DeltaTerm) * tuple->termCount;
}

static bool
TqHybridBm25ReadMeta(Relation index, TqHybridBm25MetaTupleData *meta)
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
		if (tuple->type == TQHYBRID_BM25_META_TUPLE_TYPE)
		{
			*meta = *tuple;
			UnlockReleaseBuffer(buf);
			return true;
		}
	}

	UnlockReleaseBuffer(buf);
	ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
			 errmsg("turbohybrid BM25 metadata tuple is missing"),
			 errdetail("Metapage points to BM25 metadata block %u, but no metadata tuple was found.",
					   blkno)));
}

bool
TqHybridBm25GetPlanningStats(Relation index, TqHybridBm25PlanningStats *stats)
{
	TqHybridBm25MetaTupleData meta;

	memset(stats, 0, sizeof(*stats));
	if (!TqHybridBm25ReadMeta(index, &meta))
		return false;

	stats->docCount = meta.docCount;
	stats->termCount = meta.termCount;
	stats->termTupleCount = meta.termTupleCount;
	stats->deltaDocCount = meta.deltaDocCount;
	stats->deltaTermCount = meta.deltaTermCount;
	stats->postingsPages = meta.postingsPages;
	stats->blockMaxPages = meta.blockMaxPages;
	stats->deltaPages = meta.deltaPages;
	stats->hasBm25 = true;
	return true;
}

static void
TqHybridBm25LoadDocStats(Relation index, const TqHybridBm25MetaTupleData *meta,
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
			return;
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
TqHybridBm25LoadHeapTids(Relation index, const HnswMetaPageData *meta,
						 ItemPointerData *heapTids, bool *liveNodes)
{
	bool		tqWeighted = (meta->tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0;
	int			codeTuplesPerPage =
		TqGraphTuplesPerPage(TqGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  tqWeighted));
	int			codePageCount = TqGraphPageCount(meta->tqNodeCount,
												 codeTuplesPerPage);
	BlockNumber *codeBlknos;

	memset(heapTids, 0, sizeof(ItemPointerData) * meta->tqNodeCount);
	memset(liveNodes, 0, sizeof(bool) * meta->tqNodeCount);
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
			if (tuple->type == TQ_GRAPH_CODE_TUPLE_TYPE &&
				tuple->nodeId < meta->tqNodeCount)
			{
				heapTids[tuple->nodeId] = tuple->heaptid;
				liveNodes[tuple->nodeId] =
					(tuple->flags & TQ_GRAPH_NODE_DEAD) == 0;
			}
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(codeBlknos);
}

static bool
TqHybridBm25CacheMatches(TqHybridBm25Cache *cache, Relation index,
						 const TqHybridBm25MetaTupleData *bm25Meta,
						 const HnswMetaPageData *graphMeta)
{
	return cache->relid == RelationGetRelid(index) &&
		cache->relfilenumber == TqGraphRelFileNumber(index) &&
		cache->graphFlags == graphMeta->graphFlags &&
		cache->nodeCount == graphMeta->tqNodeCount &&
		cache->docCount == bm25Meta->docCount &&
		cache->termCount == bm25Meta->termCount &&
		cache->deltaGeneration == bm25Meta->deltaGeneration &&
		cache->lastCompactionGeneration == bm25Meta->lastCompactionGeneration &&
		cache->docStatsStartBlkno == bm25Meta->docStatsStartBlkno &&
		cache->lexiconStartBlkno == bm25Meta->lexiconStartBlkno &&
		cache->postingsStartBlkno == bm25Meta->postingsStartBlkno &&
		cache->blockMaxStartBlkno == bm25Meta->blockMaxStartBlkno &&
		cache->deltaStartBlkno == bm25Meta->deltaStartBlkno;
}

static void
TqHybridBm25DropStaleCaches(Relation index,
							const TqHybridBm25MetaTupleData *bm25Meta,
							const HnswMetaPageData *graphMeta)
{
	TqHybridBm25Cache **link = &tqhybrid_bm25_cache_list;

	while (*link != NULL)
	{
		TqHybridBm25Cache *cache = *link;

		if (cache->relid == RelationGetRelid(index) &&
			!TqHybridBm25CacheMatches(cache, index, bm25Meta, graphMeta))
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}
		link = &cache->next;
	}
}

void
TqHybridBm25InvalidateCache(Relation index)
{
	TqHybridBm25Cache **link = &tqhybrid_bm25_cache_list;
	Oid			relid = RelationGetRelid(index);

	while (*link != NULL)
	{
		TqHybridBm25Cache *cache = *link;

		if (cache->relid == relid)
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}
		link = &cache->next;
	}
}

static uint32
TqHybridBm25CacheAppendTermBytes(TqHybridBm25Cache *cache,
								 const char *termBytes, uint16 termLen)
{
	uint32		offset = cache->termBytesArenaUsed;
	uint32		required = offset + termLen;

	if (required > cache->termBytesArenaCapacity)
	{
		uint32		newCapacity = Max(cache->termBytesArenaCapacity, 1024);

		while (newCapacity < required)
			newCapacity *= 2;
		if (cache->termBytesArena == NULL)
			cache->termBytesArena = palloc(newCapacity);
		else
			cache->termBytesArena = repalloc(cache->termBytesArena,
											 newCapacity);
		cache->termBytesArenaCapacity = newCapacity;
	}

	memcpy(cache->termBytesArena + offset, termBytes, termLen);
	cache->termBytesArenaUsed = required;
	return offset;
}

static void
TqHybridBm25LoadLexiconDirectory(Relation index,
								 const TqHybridBm25MetaTupleData *meta,
								 TqHybridBm25Cache *cache)
{
	BlockNumber blkno = meta->lexiconStartBlkno;

	cache->lexicon = palloc0(sizeof(TqHybridBm25CacheLexiconEntry) *
							 Max(meta->termCount, 1));
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
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("turbohybrid BM25 lexicon page has unexpected page kind")));
		}

		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			TqHybridBm25LexiconEntry tuple;
			TqHybridBm25CacheLexiconEntry *entry;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqHybridBm25LexiconEntry) PageGetItem(page, iid);
			if (tuple->type != TQHYBRID_BM25_LEXICON_TUPLE_TYPE)
				continue;
			if (cache->lexiconCount >= meta->termCount)
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("turbohybrid BM25 lexicon has more entries than metadata declares")));
			}

			entry = &cache->lexicon[cache->lexiconCount++];
			entry->termLen = tuple->termLen;
			entry->termHash = tuple->termHash;
			entry->termId = tuple->termId;
			entry->df = tuple->df;
			entry->cf = tuple->cf;
			entry->postingsBlkno = tuple->postingsBlkno;
			entry->postingsOffno = tuple->postingsOffno;
			entry->postingsChunkCount = tuple->postingsChunkCount;
			entry->postingsBytes = tuple->postingsBytes;
			entry->blockMaxBlkno = tuple->blockMaxBlkno;
			entry->blockMaxOffno = tuple->blockMaxOffno;
			entry->termOffset =
				TqHybridBm25CacheAppendTermBytes(cache, tuple->termBytes,
												 tuple->termLen);
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
	}

	for (uint32 i = 0; i < cache->lexiconCount; i++)
		cache->lexicon[i].termBytes =
			cache->termBytesArena + cache->lexicon[i].termOffset;

	if (cache->lexiconCount > 1)
		qsort(cache->lexicon, cache->lexiconCount,
			  sizeof(TqHybridBm25CacheLexiconEntry),
			  TqHybridBm25CacheLexiconCompare);
}

static TqHybridBm25Cache *
TqHybridBm25BuildCache(Relation index,
					   const TqHybridBm25MetaTupleData *bm25Meta,
					   const HnswMetaPageData *graphMeta)
{
	MemoryContext cacheCtx;
	MemoryContext oldCtx;
	TqHybridBm25Cache *cache;

	cacheCtx = AllocSetContextCreate(CacheMemoryContext,
									 "turbohybrid BM25 reader cache",
									 ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(cacheCtx);

	cache = palloc0(sizeof(TqHybridBm25Cache));
	cache->relid = RelationGetRelid(index);
	cache->relfilenumber = TqGraphRelFileNumber(index);
	cache->graphFlags = graphMeta->graphFlags;
	cache->nodeCount = graphMeta->tqNodeCount;
	cache->docCount = bm25Meta->docCount;
	cache->termCount = bm25Meta->termCount;
	cache->deltaGeneration = bm25Meta->deltaGeneration;
	cache->lastCompactionGeneration = bm25Meta->lastCompactionGeneration;
	cache->docStatsStartBlkno = bm25Meta->docStatsStartBlkno;
	cache->lexiconStartBlkno = bm25Meta->lexiconStartBlkno;
	cache->postingsStartBlkno = bm25Meta->postingsStartBlkno;
	cache->blockMaxStartBlkno = bm25Meta->blockMaxStartBlkno;
	cache->deltaStartBlkno = bm25Meta->deltaStartBlkno;
	cache->ctx = cacheCtx;
	TqHybridBm25LoadLexiconDirectory(index, bm25Meta, cache);

	cache->next = tqhybrid_bm25_cache_list;
	tqhybrid_bm25_cache_list = cache;

	MemoryContextSwitchTo(oldCtx);
	return cache;
}

static void
TqHybridBm25EnsureDocStats(Relation index, TqHybridBm25Cache *cache,
						   const TqHybridBm25MetaTupleData *bm25Meta,
						   const HnswMetaPageData *graphMeta)
{
	MemoryContext oldCtx;

	if (cache->docStatsLoaded)
		return;

	oldCtx = MemoryContextSwitchTo(cache->ctx);
	cache->docLens = palloc0(sizeof(uint32) * Max(cache->nodeCount, 1));
	TqHybridBm25LoadDocStats(index, bm25Meta, graphMeta->tqNodeCount,
							 cache->docLens);
	cache->docStatsLoaded = true;
	MemoryContextSwitchTo(oldCtx);
}

static void
TqHybridBm25EnsureLiveness(Relation index, TqHybridBm25Cache *cache,
						   const HnswMetaPageData *graphMeta)
{
	MemoryContext oldCtx;

	if (cache->livenessLoaded)
		return;

	oldCtx = MemoryContextSwitchTo(cache->ctx);
	cache->heapTids = palloc0(sizeof(ItemPointerData) *
							  Max(cache->nodeCount, 1));
	cache->liveNodes = palloc0(sizeof(bool) * Max(cache->nodeCount, 1));
	TqHybridBm25LoadHeapTids(index, graphMeta, cache->heapTids,
							 cache->liveNodes);
	cache->livenessLoaded = true;
	MemoryContextSwitchTo(oldCtx);
}

static TqHybridBm25Cache *
TqHybridBm25GetCache(Relation index,
					 const TqHybridBm25MetaTupleData *bm25Meta,
					 const HnswMetaPageData *graphMeta,
					 TqHybridBm25QueryStats *stats)
{
	TqHybridBm25Cache *cache;
	instr_time	start;

	TqHybridBm25DropStaleCaches(index, bm25Meta, graphMeta);
	for (cache = tqhybrid_bm25_cache_list; cache != NULL; cache = cache->next)
	{
		if (TqHybridBm25CacheMatches(cache, index, bm25Meta, graphMeta))
		{
			if (stats != NULL)
			{
				stats->cacheHit = true;
				stats->cacheDocstatsLoaded = cache->docStatsLoaded;
				stats->cacheLivenessLoaded = cache->livenessLoaded;
			}
			return cache;
		}
	}

	INSTR_TIME_SET_CURRENT(start);
	cache = TqHybridBm25BuildCache(index, bm25Meta, graphMeta);
	if (stats != NULL)
	{
		stats->cacheBuildUs = TqHybridBm25ElapsedUs(start);
		stats->cacheDocstatsLoaded = cache->docStatsLoaded;
		stats->cacheLivenessLoaded = cache->livenessLoaded;
	}
	return cache;
}

static bool
TqHybridBm25CacheFindLexiconEntry(TqHybridBm25Cache *cache,
								  TqHybridBm25QueryTerm *term,
								  TqHybridBm25LexiconEntryData *entry)
{
	uint64		termHash = TqHybridBm25HashTerm(term->term, term->termLen);
	int			lo = 0;
	int			hi = (int) cache->lexiconCount - 1;

	while (lo <= hi)
	{
		int			mid = lo + (hi - lo) / 2;
		TqHybridBm25CacheLexiconEntry *candidate = &cache->lexicon[mid];
		int			cmp;

		if (candidate->termHash != termHash)
			cmp = candidate->termHash < termHash ? -1 : 1;
		else if (candidate->termLen != term->termLen)
			cmp = candidate->termLen < term->termLen ? -1 : 1;
		else
			cmp = memcmp(candidate->termBytes, term->term, term->termLen);

		if (cmp == 0)
		{
			memset(entry, 0, sizeof(*entry));
			entry->type = TQHYBRID_BM25_LEXICON_TUPLE_TYPE;
			entry->termLen = candidate->termLen;
			entry->termHash = candidate->termHash;
			entry->termId = candidate->termId;
			entry->df = candidate->df;
			entry->cf = candidate->cf;
			entry->postingsBlkno = candidate->postingsBlkno;
			entry->postingsOffno = candidate->postingsOffno;
			entry->postingsChunkCount = candidate->postingsChunkCount;
			entry->postingsBytes = candidate->postingsBytes;
			entry->blockMaxBlkno = candidate->blockMaxBlkno;
			entry->blockMaxOffno = candidate->blockMaxOffno;
			return true;
		}
		if (cmp < 0)
			lo = mid + 1;
		else
			hi = mid - 1;
	}

	return false;
}

static TqHybridBm25DeltaCacheEntry *
TqHybridBm25FindDeltaEntry(TqHybridBm25DeltaCacheEntry *deltaTerms,
						   uint32 deltaTermCount,
						   TqHybridBm25QueryTerm *term)
{
	uint64		termHash = TqHybridBm25HashTerm(term->term, term->termLen);
	int			lo = 0;
	int			hi = (int) deltaTermCount - 1;

	while (lo <= hi)
	{
		int			mid = lo + (hi - lo) / 2;
		TqHybridBm25DeltaCacheEntry *candidate = &deltaTerms[mid];
		int			cmp;

		if (candidate->termHash != termHash)
			cmp = candidate->termHash < termHash ? -1 : 1;
		else if (candidate->termLen != term->termLen)
			cmp = candidate->termLen < term->termLen ? -1 : 1;
		else
			cmp = memcmp(candidate->termBytes, term->term, term->termLen);

		if (cmp == 0)
			return candidate;
		if (cmp < 0)
			lo = mid + 1;
		else
			hi = mid - 1;
	}

	return NULL;
}

static bool
TqHybridBm25DeltaBuildSameTerm(TqHybridBm25DeltaBuildEntry *left,
							   TqHybridBm25DeltaBuildEntry *right)
{
	return left->termHash == right->termHash &&
		left->termLen == right->termLen &&
		memcmp(left->termBytes, right->termBytes, left->termLen) == 0;
}

static void
TqHybridBm25BuildDeltaCacheEntries(Relation index,
								   const TqHybridBm25MetaTupleData *meta,
								   TqHybridBm25Cache *cache,
								   TqHybridBm25QueryTerm *filterTerms,
								   int filterTermCount,
								   MemoryContext memoryContext,
								   TqHybridBm25DeltaCacheEntry **outTerms,
								   uint32 *outTermCount,
								   uint32 *outPostingCount,
								   uint64 *outBytes,
								   TqHybridBm25QueryStats *stats)
{
	MemoryContext oldCtx;
	BlockNumber blkno = meta->deltaStartBlkno;
	TqHybridBm25DeltaBuildEntry *buildEntries = NULL;
	uint32		buildCount = 0;
	uint32		buildCapacity = filterTerms == NULL ?
		Max(meta->deltaTermCount, 1) : Max((uint32) filterTermCount, 1);
	uint64		blocksVisited = 0;
	uint64		finalBytes = 0;

	*outTerms = NULL;
	*outTermCount = 0;
	*outPostingCount = 0;
	*outBytes = 0;
	oldCtx = MemoryContextSwitchTo(memoryContext);
	buildEntries = palloc0(sizeof(TqHybridBm25DeltaBuildEntry) *
						   buildCapacity);

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		HnswPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		blocksVisited++;
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
				tuple->nodeId >= cache->nodeCount ||
				!cache->liveNodes[tuple->nodeId])
				continue;

			termBytes = TqHybridBm25DeltaTermBytes(tuple);
			for (uint16 i = 0; i < tuple->termCount; i++)
			{
				TqHybridBm25DeltaTerm *term = &tuple->terms[i];
				TqHybridBm25DeltaBuildEntry *entry;

				if (term->termOffset + term->termLen > tuple->termBytesLen)
					continue;
				if (filterTerms != NULL &&
					TqHybridBm25FindQueryTerm(filterTerms, filterTermCount,
											  termBytes + term->termOffset,
											  term->termLen) < 0)
					continue;
				if (buildCount >= buildCapacity)
				{
					buildCapacity *= 2;
					buildEntries = repalloc(buildEntries,
											sizeof(TqHybridBm25DeltaBuildEntry) *
											buildCapacity);
				}
				entry = &buildEntries[buildCount++];
				entry->termLen = term->termLen;
				entry->termHash = term->termHash;
				entry->nodeId = tuple->nodeId;
				entry->tf = term->tf;
				entry->docLen = tuple->docLen;
				entry->heaptid = tuple->heaptid;
				entry->termBytes = palloc(term->termLen);
				memcpy(entry->termBytes, termBytes + term->termOffset,
					   term->termLen);
			}
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
		CHECK_FOR_INTERRUPTS();
	}

	if (buildCount > 0)
	{
		uint32		termCount = 0;
		uint32		outTerm = 0;

		qsort(buildEntries, buildCount,
			  sizeof(TqHybridBm25DeltaBuildEntry),
			  TqHybridBm25DeltaBuildCompare);
		for (uint32 i = 0; i < buildCount;)
		{
			uint32		j = i + 1;

			while (j < buildCount &&
				   TqHybridBm25DeltaBuildSameTerm(&buildEntries[i],
												  &buildEntries[j]))
				j++;
			termCount++;
			i = j;
		}

		*outTerms = palloc0(sizeof(TqHybridBm25DeltaCacheEntry) *
							 termCount);
		finalBytes += sizeof(TqHybridBm25DeltaCacheEntry) * termCount;
		for (uint32 i = 0; i < buildCount;)
		{
			uint32		j = i + 1;
			TqHybridBm25DeltaCacheEntry *termEntry;

			while (j < buildCount &&
				   TqHybridBm25DeltaBuildSameTerm(&buildEntries[i],
												  &buildEntries[j]))
				j++;

			termEntry = &(*outTerms)[outTerm++];
			termEntry->termLen = buildEntries[i].termLen;
			termEntry->termHash = buildEntries[i].termHash;
			termEntry->df = j - i;
			termEntry->postingCount = j - i;
			termEntry->termBytes = palloc(termEntry->termLen);
			finalBytes += termEntry->termLen;
			memcpy(termEntry->termBytes, buildEntries[i].termBytes,
				   termEntry->termLen);
			termEntry->postings =
				palloc0(sizeof(TqHybridBm25DeltaCachePosting) *
						termEntry->postingCount);
			finalBytes += sizeof(TqHybridBm25DeltaCachePosting) *
				termEntry->postingCount;
			for (uint32 k = i; k < j; k++)
			{
				TqHybridBm25DeltaCachePosting *posting =
					&termEntry->postings[k - i];

				posting->nodeId = buildEntries[k].nodeId;
				posting->tf = buildEntries[k].tf;
				posting->docLen = buildEntries[k].docLen;
				posting->heaptid = buildEntries[k].heaptid;
				pfree(buildEntries[k].termBytes);
			}
			*outPostingCount += termEntry->postingCount;
			i = j;
		}
		*outTermCount = outTerm;
		qsort(*outTerms, *outTermCount,
			  sizeof(TqHybridBm25DeltaCacheEntry),
			  TqHybridBm25DeltaCacheCompare);
	}
	pfree(buildEntries);
	*outBytes = finalBytes;

	if (stats != NULL)
	{
		stats->blocksVisited += blocksVisited;
		stats->deltaBlocksVisited += blocksVisited;
	}
	MemoryContextSwitchTo(oldCtx);
}

static void
TqHybridBm25EnsureDeltaCache(Relation index,
							 const TqHybridBm25MetaTupleData *meta,
							 TqHybridBm25Cache *cache,
							 TqHybridBm25QueryStats *stats)
{
	if (cache->deltaCacheBuilt)
	{
		if (stats != NULL)
			stats->deltaCacheHit = true;
		return;
	}

	TqHybridBm25BuildDeltaCacheEntries(index, meta, cache, NULL, 0,
									   cache->ctx, &cache->deltaTerms,
									   &cache->deltaTermCount,
									   &cache->deltaPostingCount,
									   &cache->deltaCacheBytes, stats);
	cache->deltaCacheBuilt = true;
	cache->deltaCacheBytes = MemoryContextMemAllocated(cache->ctx, true);
}

static int
TqHybridBm25FindQueryTerm(TqHybridBm25QueryTerm *terms, int termCount,
						  const char *term, uint16 termLen)
{
	uint64		termHash = TqHybridBm25HashTerm(term, termLen);

	for (int i = 0; i < termCount; i++)
	{
		if (terms[i].termLen == termLen &&
			TqHybridBm25HashTerm(terms[i].term, terms[i].termLen) == termHash &&
			memcmp(terms[i].term, term, termLen) == 0)
			return i;
	}

	return -1;
}

static bool
TqHybridBm25EvalTsQueryItem(QueryItem *item, char *operands,
							TqHybridBm25QueryTerm *terms, int termCount,
							uint64 matchedTerms)
{
	if (item->type == QI_VAL)
	{
		char	   *term = operands + item->qoperand.distance;
		int			termNo;

		termNo = TqHybridBm25FindQueryTerm(terms, termCount, term,
										   item->qoperand.length);
		return termNo >= 0 && (matchedTerms & terms[termNo].matchBit) != 0;
	}

	if (item->type == QI_OPR)
	{
		QueryItem  *right = item + 1;
		QueryItem  *left = item + item->qoperator.left;

		if (item->qoperator.oper == OP_AND)
			return TqHybridBm25EvalTsQueryItem(left, operands, terms,
											   termCount, matchedTerms) &&
				TqHybridBm25EvalTsQueryItem(right, operands, terms,
											termCount, matchedTerms);
		if (item->qoperator.oper == OP_OR)
			return TqHybridBm25EvalTsQueryItem(left, operands, terms,
											   termCount, matchedTerms) ||
				TqHybridBm25EvalTsQueryItem(right, operands, terms,
											termCount, matchedTerms);
	}

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("turbohybrid BM25 supports OR/AND tsquery terms only")));
}

static bool
TqHybridBm25MatchedQuery(TSQuery query, TqHybridBm25QueryTerm *terms,
						 int termCount, uint64 matchedTerms)
{
	if (query->size == 0)
		return false;
	return TqHybridBm25EvalTsQueryItem(GETQUERY(query), GETOPERAND(query),
									   terms, termCount, matchedTerms);
}

static void
TqHybridBm25CountDeltaDf(TqHybridBm25DeltaCacheEntry *deltaTerms,
						 uint32 deltaTermCount,
						 TqHybridBm25QueryTerm *terms, int termCount)
{
	for (int termNo = 0; termNo < termCount; termNo++)
	{
		TqHybridBm25DeltaCacheEntry *entry;

		entry = TqHybridBm25FindDeltaEntry(deltaTerms, deltaTermCount,
										   &terms[termNo]);
		if (entry != NULL)
			terms[termNo].deltaDf = entry->df;
	}
}

static void
TqHybridBm25ScoreDelta(const TqHybridBm25MetaTupleData *meta,
					   TqHybridBm25DeltaCacheEntry *deltaTerms,
					   uint32 deltaTermCount,
					   TqHybridBm25QueryTerm *terms, int termCount,
					   TqHybridBm25Accumulator *acc,
					   TqHybridBm25QueryStats *stats)
{
	double		corpusDocCount = Max((double) meta->docCount +
									 (double) meta->deltaDocCount, 1.0);
	double		avgDocLen = ((double) meta->totalDocLen +
							 (double) meta->deltaTotalDocLen) / corpusDocCount;

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		TqHybridBm25QueryTerm *term = &terms[termNo];
		TqHybridBm25DeltaCacheEntry *deltaEntry;
		uint32		df;
		double		idf;

		deltaEntry = TqHybridBm25FindDeltaEntry(deltaTerms, deltaTermCount,
												term);
		if (deltaEntry == NULL)
			continue;

		df = term->baseDf + term->deltaDf;
		idf = log(1.0 + (corpusDocCount - (double) df + 0.5) /
				  ((double) df + 0.5));
		for (uint32 i = 0; i < deltaEntry->postingCount; i++)
		{
			TqHybridBm25DeltaCachePosting *posting = &deltaEntry->postings[i];
			double		dl;
			double		norm;
			double		tf = posting->tf;
			TqHybridBm25AccumulatorEntry *entry;

			dl = Max((double) posting->docLen, 1.0);
			norm = (double) meta->k1 *
				(1.0 - (double) meta->b +
				 (double) meta->b * dl / Max(avgDocLen, 1.0));
			entry = TqHybridBm25AccumulatorLookup(acc, posting->nodeId, true);
			entry->docLen = posting->docLen;
			entry->heaptid = posting->heaptid;
			entry->hasDeltaDoc = true;
			entry->score += idf *
				((tf * ((double) meta->k1 + 1.0)) / (tf + norm));
			entry->matchedTerms |= term->matchBit;
			TqHybridBm25AccumulatorUpdateTopK(acc, entry);
			if (stats != NULL)
			{
				stats->postingsDecoded++;
				stats->deltaPostingsDecoded++;
			}

			CHECK_FOR_INTERRUPTS();
		}
	}
}

static int
TqHybridBm25ExtractTerms(TSQuery query, TqHybridBm25QueryTerm **terms,
						 MemoryContext memoryContext)
{
	QueryItem  *items = GETQUERY(query);
	char	   *operands = GETOPERAND(query);
	int			count = 0;
	TqHybridBm25QueryTerm *out;

	for (int i = 0; i < query->size; i++)
	{
		QueryItem  *item = &items[i];

		if (item->type == QI_VAL)
		{
			if (item->qoperand.prefix)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("turbohybrid BM25 prefix tsquery terms are not supported yet")));
			if (item->qoperand.weight != 0)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("turbohybrid BM25 weighted tsquery terms are not supported yet")));
			count++;
		}
		else if (item->type == QI_OPR &&
				 (item->qoperator.oper == OP_NOT ||
				  item->qoperator.oper == OP_PHRASE))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("turbohybrid BM25 supports OR/AND tsquery terms only")));
	}

	out = MemoryContextAllocZero(memoryContext,
								 sizeof(TqHybridBm25QueryTerm) * Max(count, 1));
	count = 0;
	for (int i = 0; i < query->size; i++)
	{
		QueryItem  *item = &items[i];
		char	   *term;

		if (item->type != QI_VAL)
			continue;

		term = operands + item->qoperand.distance;
		if (TqHybridBm25FindQueryTerm(out, count, term,
									  item->qoperand.length) >= 0)
			continue;
		if (count >= 64)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("turbohybrid BM25 supports up to 64 distinct query terms")));
		out[count].term = term;
		out[count].termLen = item->qoperand.length;
		out[count].matchBit = UINT64CONST(1) << count;
		count++;
	}

	*terms = out;
	return count;
}

static int
TqHybridBm25ScoreCompare(const void *a, const void *b)
{
	const TqHybridBm25NodeScore *sa = (const TqHybridBm25NodeScore *) a;
	const TqHybridBm25NodeScore *sb = (const TqHybridBm25NodeScore *) b;

	if (sa->score > sb->score)
		return -1;
	if (sa->score < sb->score)
		return 1;
	return (sa->nodeId > sb->nodeId) - (sa->nodeId < sb->nodeId);
}

static double
TqHybridBm25PostingScore(const TqHybridBm25MetaTupleData *meta, double idf,
						 double avgDocLen, uint16 tf, uint32 docLen)
{
	double		norm;

	norm = (double) meta->k1 *
		(1.0 - (double) meta->b +
		 (double) meta->b * Max((double) docLen, 1.0) / Max(avgDocLen, 1.0));
	return idf * (((double) tf * ((double) meta->k1 + 1.0)) /
				  ((double) tf + norm));
}

static double
TqHybridBm25PostingPrecomputedScore(const TqHybridBm25MetaTupleData *meta,
									double idf, uint16 tfNormQ16)
{
	double		scale = Max((double) meta->k1 + 1.0, 1.0);

	return idf * scale * ((double) tfNormQ16 / (double) PG_UINT16_MAX);
}

static double
TqHybridBm25PostingScoreDecoded(const TqHybridBm25MetaTupleData *meta,
								double idf, double avgDocLen,
								const TqHybridBm25Posting *posting,
								uint32 docLen)
{
	if ((meta->reserved2 & TQHYBRID_BM25_META_FLAG_TFNORM_Q16) != 0)
		return TqHybridBm25PostingPrecomputedScore(meta, idf,
												   posting->reserved);
	return TqHybridBm25PostingScore(meta, idf, avgDocLen, posting->tf,
									docLen);
}

static double
TqHybridBm25BlockUpperBound(const TqHybridBm25MetaTupleData *meta, double idf,
							double avgDocLen, uint16 maxTf)
{
	/*
	 * The configured BM25 b range is [0, 1], so docLen=1 gives the minimum
	 * normalization term and therefore the maximum possible contribution for
	 * any posting in the block.
	 */
	return TqHybridBm25PostingScore(meta, idf, avgDocLen, maxTf, 1);
}

pg_attribute_unused()
static bool
TqHybridBm25NeonAvailable(void)
{
#if !defined(TQ_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
	return tqhybrid_bm25_simd_force == TQHYBRID_BM25_SIMD_FORCE_AUTO ||
		tqhybrid_bm25_simd_force == TQHYBRID_BM25_SIMD_FORCE_NEON;
#else
	return false;
#endif
}

pg_attribute_unused()
static bool
TqHybridBm25Avx2Available(void)
{
#if !defined(TQ_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
	if (tqhybrid_bm25_simd_force != TQHYBRID_BM25_SIMD_FORCE_AUTO &&
		tqhybrid_bm25_simd_force != TQHYBRID_BM25_SIMD_FORCE_AVX2)
		return false;

#if defined(__AVX2__)
	return true;
#elif defined(__GNUC__) || defined(__clang__)
	return __builtin_cpu_supports("avx2");
#else
	return false;
#endif
#else
	return false;
#endif
}

pg_attribute_unused()
static void
TqHybridBm25RecordKernel(TqHybridBm25QueryStats *stats, int kernel)
{
	if (stats == NULL)
		return;
	if (kernel != TQHYBRID_BM25_KERNEL_SCALAR)
	{
		stats->decodeKernel = kernel;
		stats->scoreKernel = kernel;
	}
}

static bool TQHYBRID_BM25_AVX2_TARGET
TqHybridBm25ScoreOffset16TfNormAvx2(const TqHybridBm25PostingsTuple postings,
									Size itemSize,
									const TqHybridBm25MetaTupleData *meta,
									const HnswMetaPageData *graphMeta,
									float8 idf,
									const uint32 *docLens,
									const bool *liveNodes,
									TqHybridBm25Accumulator *acc,
									TqHybridBm25QueryTerm *term,
									TqHybridBm25QueryStats *stats)
{
#if !defined(TQ_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
	const uint16 *pairs;
	const uint16 *tfNorms;
	uint16		count = postings->count;
	Size		pairBytes = (Size) count * sizeof(uint16) * 2;
	Size		tfNormBytes = (Size) count * sizeof(uint16);
	__m256		factor;
	uint16		i = 0;

	if (!TqHybridBm25Avx2Available() ||
		(postings->encoding & TQHYBRID_BM25_POSTINGS_ENCODING_MASK) !=
		TQHYBRID_BM25_POSTINGS_ENCODING_OFFSET16 ||
		(postings->encoding & TQHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16) == 0 ||
		TqHybridBm25PostingsTupleSize(pairBytes + tfNormBytes) != itemSize ||
		postings->payloadBytes != pairBytes + tfNormBytes)
		return false;

	pairs = (const uint16 *) postings->payload;
	tfNorms = (const uint16 *) (postings->payload + pairBytes);
	factor = _mm256_set1_ps((float) (idf * Max((double) meta->k1 + 1.0, 1.0) /
									 (double) PG_UINT16_MAX));
	for (; i + 8 <= count; i += 8)
	{
		__m128i		norm16 = _mm_loadu_si128((const __m128i *) (tfNorms + i));
		__m256		scoreVec = _mm256_mul_ps(
			_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(norm16)), factor);
		uint16		offsets[8];
		float		scores[8];

		for (int lane = 0; lane < 8; lane++)
			offsets[lane] = pairs[(i + lane) * 2];
		_mm256_storeu_ps(scores, scoreVec);
		for (int lane = 0; lane < 8; lane++)
		{
			uint32		nodeId = postings->firstNodeId + offsets[lane];

			if (nodeId >= graphMeta->tqNodeCount || docLens[nodeId] == 0 ||
				!liveNodes[nodeId])
				continue;
			TqHybridBm25AccumulatorAddTermScore(acc, nodeId, scores[lane],
												term->matchBit);
			if (stats != NULL)
				stats->postingsDecoded++;
		}
		if (stats != NULL)
			stats->simdBlocks++;
	}
	for (; i < count; i++)
	{
		uint32		nodeId = postings->firstNodeId + pairs[i * 2];
		double		score = TqHybridBm25PostingPrecomputedScore(meta, idf,
																tfNorms[i]);

		if (nodeId >= graphMeta->tqNodeCount || docLens[nodeId] == 0 ||
			!liveNodes[nodeId])
			continue;
		TqHybridBm25AccumulatorAddTermScore(acc, nodeId, score,
											term->matchBit);
		if (stats != NULL)
		{
			stats->postingsDecoded++;
			stats->scalarTailPostings++;
		}
	}
	TqHybridBm25RecordKernel(stats, TQHYBRID_BM25_KERNEL_AVX2);
	return true;
#else
	(void) postings;
	(void) itemSize;
	(void) meta;
	(void) graphMeta;
	(void) idf;
	(void) docLens;
	(void) liveNodes;
	(void) acc;
	(void) term;
	(void) stats;
	return false;
#endif
}

static bool
TqHybridBm25ScoreOffset16TfNormNeon(const TqHybridBm25PostingsTuple postings,
									Size itemSize,
									const TqHybridBm25MetaTupleData *meta,
									const HnswMetaPageData *graphMeta,
									float8 idf,
									const uint32 *docLens,
									const bool *liveNodes,
									TqHybridBm25Accumulator *acc,
									TqHybridBm25QueryTerm *term,
									TqHybridBm25QueryStats *stats)
{
#if !defined(TQ_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
	const uint16 *pairs;
	const uint16 *tfNorms;
	uint16		count = postings->count;
	Size		pairBytes = (Size) count * sizeof(uint16) * 2;
	Size		tfNormBytes = (Size) count * sizeof(uint16);
	float		factor;
	uint16		i = 0;

	if (!TqHybridBm25NeonAvailable() ||
		(postings->encoding & TQHYBRID_BM25_POSTINGS_ENCODING_MASK) !=
		TQHYBRID_BM25_POSTINGS_ENCODING_OFFSET16 ||
		(postings->encoding & TQHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16) == 0 ||
		TqHybridBm25PostingsTupleSize(pairBytes + tfNormBytes) != itemSize ||
		postings->payloadBytes != pairBytes + tfNormBytes)
		return false;

	pairs = (const uint16 *) postings->payload;
	tfNorms = (const uint16 *) (postings->payload + pairBytes);
	factor = (float) (idf * Max((double) meta->k1 + 1.0, 1.0) /
					  (double) PG_UINT16_MAX);
	for (; i + 8 <= count; i += 8)
	{
		uint16x8x2_t pairVec = vld2q_u16(pairs + (i * 2));
		uint16x8_t normVec = vld1q_u16(tfNorms + i);
		uint32x4_t normLo = vmovl_u16(vget_low_u16(normVec));
		uint32x4_t normHi = vmovl_u16(vget_high_u16(normVec));
		float32x4_t scoreLo = vmulq_n_f32(vcvtq_f32_u32(normLo), factor);
		float32x4_t scoreHi = vmulq_n_f32(vcvtq_f32_u32(normHi), factor);
		uint16		offsets[8];
		float		scores[8];

		vst1q_u16(offsets, pairVec.val[0]);
		vst1q_f32(scores, scoreLo);
		vst1q_f32(scores + 4, scoreHi);
		for (int lane = 0; lane < 8; lane++)
		{
			uint32		nodeId = postings->firstNodeId + offsets[lane];

			if (nodeId >= graphMeta->tqNodeCount || docLens[nodeId] == 0 ||
				!liveNodes[nodeId])
				continue;
			TqHybridBm25AccumulatorAddTermScore(acc, nodeId, scores[lane],
												term->matchBit);
			if (stats != NULL)
				stats->postingsDecoded++;
		}
		if (stats != NULL)
			stats->simdBlocks++;
	}
	for (; i < count; i++)
	{
		uint32		nodeId = postings->firstNodeId + pairs[i * 2];
		double		score = TqHybridBm25PostingPrecomputedScore(meta, idf,
																tfNorms[i]);

		if (nodeId >= graphMeta->tqNodeCount || docLens[nodeId] == 0 ||
			!liveNodes[nodeId])
			continue;
		TqHybridBm25AccumulatorAddTermScore(acc, nodeId, score,
											term->matchBit);
		if (stats != NULL)
		{
			stats->postingsDecoded++;
			stats->scalarTailPostings++;
		}
	}
	TqHybridBm25RecordKernel(stats, TQHYBRID_BM25_KERNEL_NEON);
	return true;
#else
	(void) postings;
	(void) itemSize;
	(void) meta;
	(void) graphMeta;
	(void) idf;
	(void) docLens;
	(void) liveNodes;
	(void) acc;
	(void) term;
	(void) stats;
	return false;
#endif
}

static bool
TqHybridBm25IteratorLoadChunk(TqHybridBm25PostingIterator *it)
{
	Buffer		buf;
	Page		page;
	ItemId		iid;
	TqHybridBm25PostingsTuple tuple;

	it->valid = false;
	it->count = 0;
	it->pos = 0;

	if (it->chunkNo >= it->chunkLimit ||
		!BlockNumberIsValid(it->blkno) ||
		!OffsetNumberIsValid(it->offno))
		return false;

	buf = ReadBuffer(it->index, it->blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	if (!TqHybridBm25PageIsKind(page, HNSW_PAGE_KIND_TQ_BM25_POSTINGS) ||
		it->offno > PageGetMaxOffsetNumber(page))
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 postings pointer is invalid")));
	}

	iid = PageGetItemId(page, it->offno);
	if (!ItemIdIsUsed(iid))
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	tuple = (TqHybridBm25PostingsTuple) PageGetItem(page, iid);
	if (tuple->type != TQHYBRID_BM25_POSTINGS_TUPLE_TYPE ||
		tuple->termId != it->term->lexicon.termId)
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("turbohybrid BM25 postings tuple is invalid")));
	}

	it->count = tuple->count;
	it->lastNodeId = tuple->lastNodeId;
	it->maxTf = tuple->maxTf;
	it->nextBlkno = tuple->nextBlkno;
	it->nextOffno = tuple->nextOffno;
	if (it->count > 0)
	{
			if (it->postingsCapacity < it->count)
			{
				if (it->postings == NULL)
					it->postings = MemoryContextAlloc(it->memoryContext,
													  sizeof(TqHybridBm25Posting) *
													  it->count);
				else
					it->postings = repalloc(it->postings,
											sizeof(TqHybridBm25Posting) *
											it->count);
				it->postingsCapacity = it->count;
			}
			if (!TqHybridBm25DecodePostingsTuple(tuple, ItemIdGetLength(iid),
												 it->postings))
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("turbohybrid BM25 postings payload is invalid")));
			}
			if (it->maxTf == 0)
			{
				for (uint16 i = 0; i < it->count; i++)
				it->maxTf = Max(it->maxTf, it->postings[i].tf);
		}
	}

	UnlockReleaseBuffer(buf);
	if (it->stats != NULL)
		it->stats->blocksVisited++;

	it->blkno = it->nextBlkno;
	it->offno = it->nextOffno;
	it->chunkNo++;
	it->valid = it->count > 0;
	return it->valid;
}

static bool
TqHybridBm25IteratorAdvanceValid(TqHybridBm25PostingIterator *it)
{
	for (;;)
	{
		while (it->valid && it->pos < it->count)
		{
			uint32		nodeId = it->postings[it->pos].nodeId;

			if (nodeId < it->nodeCount && it->docLens[nodeId] != 0 &&
				it->liveNodes[nodeId])
				return true;
			it->pos++;
			if (it->stats != NULL)
				it->stats->postingsDecoded++;
		}
		if (!TqHybridBm25IteratorLoadChunk(it))
			return false;
	}
}

static bool
TqHybridBm25IteratorInit(TqHybridBm25PostingIterator *it, Relation index,
						 TqHybridBm25QueryTerm *term, double idf,
						 double avgDocLen, const uint32 *docLens,
						 const bool *liveNodes, uint32 nodeCount,
						 MemoryContext memoryContext,
						 TqHybridBm25QueryStats *stats)
{
	memset(it, 0, sizeof(*it));
	it->index = index;
	it->term = term;
	it->idf = idf;
	it->avgDocLen = avgDocLen;
	it->docLens = docLens;
	it->liveNodes = liveNodes;
	it->nodeCount = nodeCount;
	it->chunkLimit = Max(term->lexicon.postingsChunkCount, 1);
	it->blkno = term->lexicon.postingsBlkno;
	it->offno = term->lexicon.postingsOffno;
	it->memoryContext = memoryContext;
	it->stats = stats;

	if (!TqHybridBm25IteratorLoadChunk(it))
		return false;
	return TqHybridBm25IteratorAdvanceValid(it);
}

static void
TqHybridBm25IteratorClose(TqHybridBm25PostingIterator *it)
{
	if (it->postings != NULL)
	{
		pfree(it->postings);
		it->postings = NULL;
	}
	it->postingsCapacity = 0;
	it->valid = false;
}

static uint32
TqHybridBm25IteratorNodeId(const TqHybridBm25PostingIterator *it)
{
	if (!it->valid || it->pos >= it->count)
		return PG_UINT32_MAX;
	return it->postings[it->pos].nodeId;
}

static bool
TqHybridBm25IteratorSeekTo(TqHybridBm25PostingIterator *it, uint32 target)
{
	while (it->valid)
	{
		if (it->lastNodeId < target)
		{
			if (it->stats != NULL)
				it->stats->blocksSkipped++;
			if (!TqHybridBm25IteratorLoadChunk(it))
				return false;
			continue;
		}

		while (it->pos < it->count &&
			   it->postings[it->pos].nodeId < target)
		{
			it->pos++;
			if (it->stats != NULL)
				it->stats->postingsDecoded++;
		}
		return TqHybridBm25IteratorAdvanceValid(it);
	}

	return false;
}

static bool
TqHybridBm25IteratorAdvancePast(TqHybridBm25PostingIterator *it,
								uint32 nodeId)
{
	if (nodeId == PG_UINT32_MAX)
		return false;
	return TqHybridBm25IteratorSeekTo(it, nodeId + 1);
}

static double
TqHybridBm25IteratorUpperBound(const TqHybridBm25MetaTupleData *meta,
							   const TqHybridBm25PostingIterator *it)
{
	return TqHybridBm25BlockUpperBound(meta, it->idf, it->avgDocLen,
									   it->maxTf);
}

static void
TqHybridBm25ActiveHeapSiftDown(TqHybridBm25PostingIterator **heap,
							   int heapCount, int pos)
{
	for (;;)
	{
		int			left = pos * 2 + 1;
		int			right = left + 1;
		int			smallest = pos;

		if (left < heapCount &&
			TqHybridBm25IteratorNodeId(heap[left]) <
			TqHybridBm25IteratorNodeId(heap[smallest]))
			smallest = left;
		if (right < heapCount &&
			TqHybridBm25IteratorNodeId(heap[right]) <
			TqHybridBm25IteratorNodeId(heap[smallest]))
			smallest = right;
		if (smallest == pos)
			break;

		{
			TqHybridBm25PostingIterator *tmp = heap[pos];

			heap[pos] = heap[smallest];
			heap[smallest] = tmp;
		}
		pos = smallest;
	}
}

static void
TqHybridBm25OrderActiveIterators(TqHybridBm25PostingIterator **active,
								 TqHybridBm25PostingIterator **ordered,
								 int activeCount,
								 TqHybridBm25QueryStats *stats)
{
	int			heapCount = activeCount;

	if (stats != NULL)
		stats->wandActiveSorts++;

	for (int i = activeCount / 2 - 1; i >= 0; i--)
		TqHybridBm25ActiveHeapSiftDown(active, activeCount, i);

	for (int i = 0; i < activeCount; i++)
	{
		ordered[i] = active[0];
		heapCount--;
		if (heapCount > 0)
		{
			active[0] = active[heapCount];
			TqHybridBm25ActiveHeapSiftDown(active, heapCount, 0);
		}
	}

	memcpy(active, ordered, sizeof(TqHybridBm25PostingIterator *) *
		   activeCount);
}

static double
TqHybridBm25KthScore(TqHybridBm25Accumulator *acc, int32 k)
{
	if (k <= 0 || acc->topHeapCount < (uint32) k)
		return 0.0;
	return acc->threshold;
}

static bool
TqHybridBm25ScoreBaseWand(Relation index,
						  const TqHybridBm25MetaTupleData *meta,
						  const HnswMetaPageData *graphMeta,
						  TSQuery query, TqHybridBm25QueryTerm *terms, int termCount,
						  const uint32 *docLens, const bool *liveNodes,
						  TqHybridBm25Accumulator *acc, int32 k,
						  MemoryContext memoryContext,
						  TqHybridBm25QueryStats *stats)
{
	TqHybridBm25PostingIterator *iterators;
	TqHybridBm25PostingIterator **active;
	TqHybridBm25PostingIterator **ordered;
	int			iteratorCount = 0;
	double		corpusDocCount = Max((double) meta->docCount +
									 (double) meta->deltaDocCount, 1.0);
	double		avgDocLen = ((double) meta->totalDocLen +
							 (double) meta->deltaTotalDocLen) / corpusDocCount;
	bool		used = false;

	iterators = MemoryContextAllocZero(memoryContext,
									   sizeof(TqHybridBm25PostingIterator) *
									   Max(termCount, 1));
	active = MemoryContextAllocZero(memoryContext,
									sizeof(TqHybridBm25PostingIterator *) *
									Max(termCount, 1));
	ordered = MemoryContextAllocZero(memoryContext,
									 sizeof(TqHybridBm25PostingIterator *) *
									 Max(termCount, 1));

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		TqHybridBm25QueryTerm *term = &terms[termNo];
		uint32		df;
		double		idf;

		if (!term->hasLexicon)
			continue;

		df = term->baseDf + term->deltaDf;
		idf = log(1.0 + (corpusDocCount - (double) df + 0.5) /
				  ((double) df + 0.5));
		if (TqHybridBm25IteratorInit(&iterators[iteratorCount], index, term,
									 idf, avgDocLen, docLens, liveNodes,
									 graphMeta->tqNodeCount, memoryContext,
									 stats))
			iteratorCount++;
		else
			TqHybridBm25IteratorClose(&iterators[iteratorCount]);
	}

	while (iteratorCount > 0)
	{
		int			activeCount = 0;
		int			pivot = -1;
		double		upperBound = 0.0;
		double		threshold = TqHybridBm25KthScore(acc, k);
		uint32		pivotNode;

		if (stats != NULL)
			stats->wandIterations++;
		for (int i = 0; i < iteratorCount; i++)
		{
			if (iterators[i].valid)
				active[activeCount++] = &iterators[i];
		}
		if (activeCount == 0)
			break;

		TqHybridBm25OrderActiveIterators(active, ordered, activeCount, stats);

		for (int i = 0; i < activeCount; i++)
		{
			upperBound += TqHybridBm25IteratorUpperBound(meta, active[i]);
			if (acc->touchedCount < (uint32) k || upperBound > threshold)
			{
				pivot = i;
				break;
			}
		}

		if (pivot < 0)
		{
			if (stats != NULL)
				stats->blocksSkipped += activeCount;
			break;
		}

		pivotNode = TqHybridBm25IteratorNodeId(active[pivot]);
		if (pivotNode == PG_UINT32_MAX)
			break;

		if (TqHybridBm25IteratorNodeId(active[0]) == pivotNode)
		{
			double		score = 0.0;
			uint64		matchedTerms = 0;
			bool		matched = false;

			for (int i = 0; i < activeCount; i++)
			{
				TqHybridBm25PostingIterator *it = active[i];

				if (!TqHybridBm25IteratorSeekTo(it, pivotNode))
					continue;
				if (TqHybridBm25IteratorNodeId(it) == pivotNode)
				{
					TqHybridBm25Posting *posting = &it->postings[it->pos];

					score += TqHybridBm25PostingScoreDecoded(meta, it->idf,
															 it->avgDocLen,
															 posting,
															 docLens[pivotNode]);
					matchedTerms |= it->term->matchBit;
					matched = true;
					if (stats != NULL)
						stats->postingsDecoded++;
				}
			}

			if (matched &&
				TqHybridBm25MatchedQuery(query, terms, termCount, matchedTerms))
			{
				used = true;
				if (stats != NULL)
					stats->candidatesScored++;
				if (acc->touchedCount < (uint32) k || score > threshold)
					TqHybridBm25AccumulatorAddTermScore(acc, pivotNode, score,
														matchedTerms);
			}

			for (int i = 0; i < activeCount; i++)
			{
				TqHybridBm25PostingIterator *it = active[i];

				if (TqHybridBm25IteratorNodeId(it) == pivotNode)
					(void) TqHybridBm25IteratorAdvancePast(it, pivotNode);
			}
		}
		else
		{
			for (int i = 0; i < pivot; i++)
				(void) TqHybridBm25IteratorSeekTo(active[i], pivotNode);
		}

		CHECK_FOR_INTERRUPTS();
	}

	for (int i = 0; i < iteratorCount; i++)
		TqHybridBm25IteratorClose(&iterators[i]);

	return used;
}

int
TqHybridBm25TopK(Relation index, TSQuery query, int32 k, bool useWand,
				 MemoryContext memoryContext, TqHybridBm25Result **results,
				 TqHybridBm25QueryStats *stats)
{
	TqHybridBm25MetaTupleData bm25Meta;
	HnswMetaPageData graphMeta;
	TqHybridBm25Cache *cache;
	const uint32 *docLens;
	const ItemPointerData *heapTids;
	const bool *liveNodes;
	TqHybridBm25DeltaCacheEntry *deltaTerms = NULL;
	uint32		deltaTermCount = 0;
	uint32		deltaPostingCount = 0;
	uint64		deltaCacheBytes = 0;
	TqHybridBm25Accumulator acc;
	TqHybridBm25QueryTerm *terms;
	int			termCount;
	int			resolvedTerms = 0;
	int			resultCount;
	bool		usedBaseWand = false;
	TqHybridBm25Posting *decodedScratch = NULL;
	uint16		decodedScratchCapacity = 0;
	MemoryContext oldCtx;

	if (stats != NULL)
	{
		memset(stats, 0, sizeof(*stats));
		stats->decodeKernel = TQHYBRID_BM25_KERNEL_SCALAR;
		stats->scoreKernel = TQHYBRID_BM25_KERNEL_SCALAR;
	}
	tqhybrid_last_bm25_decode_kernel = TQHYBRID_BM25_KERNEL_SCALAR;
	tqhybrid_last_bm25_score_kernel = TQHYBRID_BM25_KERNEL_SCALAR;
	tqhybrid_last_bm25_simd_blocks = 0;
	tqhybrid_last_bm25_scalar_tail_postings = 0;
	*results = NULL;

	if (k <= 0)
		return 0;
	if (!TqHybridBm25ReadMeta(index, &bm25Meta) ||
		!TqGraphReadMeta(index, &graphMeta) ||
		graphMeta.tqNodeCount == 0)
		return 0;

	oldCtx = MemoryContextSwitchTo(memoryContext);
	termCount = TqHybridBm25ExtractTerms(query, &terms, memoryContext);
	if (stats != NULL)
	{
		stats->queryTerms = termCount;
		stats->usedWand = false;
	}
	if (termCount == 0)
	{
		MemoryContextSwitchTo(oldCtx);
		return 0;
	}

	cache = TqHybridBm25GetCache(index, &bm25Meta, &graphMeta, stats);
	if (stats != NULL)
	{
		stats->cacheBytes = MemoryContextMemAllocated(cache->ctx, true);
		stats->cacheLexiconEntries = cache->lexiconCount;
		stats->cacheDocstatsLoaded = cache->docStatsLoaded;
		stats->cacheLivenessLoaded = cache->livenessLoaded;
	}
	TqHybridBm25AccumulatorInit(&acc, memoryContext, (uint32) (k * termCount),
								(uint32) k, stats);

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		if (TqHybridBm25CacheFindLexiconEntry(cache, &terms[termNo],
											  &terms[termNo].lexicon))
		{
			terms[termNo].hasLexicon = true;
			terms[termNo].baseDf = terms[termNo].lexicon.df;
			resolvedTerms++;
		}
	}

	if (resolvedTerms == 0 && bm25Meta.deltaDocCount == 0)
	{
		if (stats != NULL)
			stats->cacheBytes = MemoryContextMemAllocated(cache->ctx, true);
		MemoryContextSwitchTo(oldCtx);
		return 0;
	}

	TqHybridBm25EnsureDocStats(index, cache, &bm25Meta, &graphMeta);
	TqHybridBm25EnsureLiveness(index, cache, &graphMeta);
	docLens = cache->docLens;
	heapTids = cache->heapTids;
	liveNodes = cache->liveNodes;

	if (tqhybrid_bm25_cache_max_mb > 0 && !cache->deltaCacheBuilt)
	{
		TqHybridBm25BuildDeltaCacheEntries(index, &bm25Meta, cache,
										   terms, termCount, memoryContext,
										   &deltaTerms, &deltaTermCount,
										   &deltaPostingCount,
										   &deltaCacheBytes, stats);
	}
	else
	{
		TqHybridBm25EnsureDeltaCache(index, &bm25Meta, cache, stats);
		deltaTerms = cache->deltaTerms;
		deltaTermCount = cache->deltaTermCount;
		deltaPostingCount = cache->deltaPostingCount;
		deltaCacheBytes = cache->deltaCacheBytes;
	}
	TqHybridBm25CountDeltaDf(deltaTerms, deltaTermCount, terms, termCount);
	if (stats != NULL)
	{
		stats->deltaCacheBytes = deltaCacheBytes;
		stats->deltaCacheTerms = deltaTermCount;
		stats->cacheBytes = MemoryContextMemAllocated(cache->ctx, true);
		stats->cacheDocstatsLoaded = cache->docStatsLoaded;
		stats->cacheLivenessLoaded = cache->livenessLoaded;
	}
	(void) deltaPostingCount;
	for (int termNo = 0; termNo < termCount; termNo++)
	{
		if (!terms[termNo].hasLexicon && terms[termNo].deltaDf > 0)
			resolvedTerms++;
	}

	if (useWand && tqhybrid_enable_wand)
	{
		usedBaseWand = TqHybridBm25ScoreBaseWand(index, &bm25Meta,
												 &graphMeta, query, terms,
												 termCount,
												 docLens, liveNodes, &acc, k,
												 memoryContext, stats);
		if (stats != NULL)
			stats->usedWand = usedBaseWand;
	}

	if (!usedBaseWand)
	for (int termNo = 0; termNo < termCount; termNo++)
	{
		float8		idf;
		TqHybridBm25QueryTerm *term = &terms[termNo];
		uint32		df;
		double		corpusDocCount;
		double		avgDocLen;
		BlockNumber postingsBlkno;
		OffsetNumber postingsOffno;
		uint32		chunkLimit;

		if (!term->hasLexicon)
			continue;

		df = term->baseDf + term->deltaDf;
		corpusDocCount = Max((double) bm25Meta.docCount +
							 (double) bm25Meta.deltaDocCount, 1.0);
		avgDocLen = ((double) bm25Meta.totalDocLen +
					 (double) bm25Meta.deltaTotalDocLen) / corpusDocCount;
		idf = log(1.0 + (corpusDocCount - (double) df + 0.5) /
				  ((double) df + 0.5));
		postingsBlkno = term->lexicon.postingsBlkno;
		postingsOffno = term->lexicon.postingsOffno;
		chunkLimit = Max(term->lexicon.postingsChunkCount, 1);

		for (uint32 chunkNo = 0;
			 chunkNo < chunkLimit && BlockNumberIsValid(postingsBlkno) &&
			 OffsetNumberIsValid(postingsOffno);
			 chunkNo++)
		{
			Buffer		buf;
			Page		page;
				ItemId		iid;
				TqHybridBm25PostingsTuple postings;
				BlockNumber nextBlkno;
				OffsetNumber nextOffno;

			buf = ReadBuffer(index, postingsBlkno);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			if (!TqHybridBm25PageIsKind(page, HNSW_PAGE_KIND_TQ_BM25_POSTINGS) ||
				postingsOffno > PageGetMaxOffsetNumber(page))
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("turbohybrid BM25 postings pointer is invalid")));
			}

			iid = PageGetItemId(page, postingsOffno);
			if (!ItemIdIsUsed(iid))
			{
				UnlockReleaseBuffer(buf);
				break;
			}

			postings = (TqHybridBm25PostingsTuple) PageGetItem(page, iid);
			if (postings->type != TQHYBRID_BM25_POSTINGS_TUPLE_TYPE ||
				postings->termId != term->lexicon.termId)
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("turbohybrid BM25 postings tuple is invalid")));
				}
				nextBlkno = postings->nextBlkno;
				nextOffno = postings->nextOffno;
				if (stats != NULL)
					stats->blocksVisited++;
				if (TqHybridBm25ScoreOffset16TfNormAvx2(postings,
														 ItemIdGetLength(iid),
														 &bm25Meta, &graphMeta,
														 idf, docLens,
														 liveNodes, &acc, term,
														 stats) ||
					TqHybridBm25ScoreOffset16TfNormNeon(postings,
														 ItemIdGetLength(iid),
														 &bm25Meta, &graphMeta,
														 idf, docLens,
														 liveNodes, &acc, term,
														 stats))
				{
					UnlockReleaseBuffer(buf);
					postingsBlkno = nextBlkno;
					postingsOffno = nextOffno;
					CHECK_FOR_INTERRUPTS();
					continue;
				}
				if (decodedScratchCapacity < postings->count)
				{
					if (decodedScratch == NULL)
						decodedScratch = MemoryContextAlloc(memoryContext,
															sizeof(TqHybridBm25Posting) *
															postings->count);
					else
						decodedScratch = repalloc(decodedScratch,
												  sizeof(TqHybridBm25Posting) *
												  postings->count);
					decodedScratchCapacity = postings->count;
				}
				if (!TqHybridBm25DecodePostingsTuple(postings,
													 ItemIdGetLength(iid),
													 decodedScratch))
				{
					UnlockReleaseBuffer(buf);
					ereport(ERROR,
							(errcode(ERRCODE_DATA_CORRUPTED),
							 errmsg("turbohybrid BM25 postings payload is invalid")));
				}

				{
					for (uint16 i = 0; i < postings->count; i++)
					{
						TqHybridBm25Posting *posting = &decodedScratch[i];
						uint32		nodeId = posting->nodeId;

					if (nodeId >= graphMeta.tqNodeCount || docLens[nodeId] == 0 ||
						!liveNodes[nodeId])
						continue;

					TqHybridBm25AccumulatorAddTermScore(&acc, nodeId,
														TqHybridBm25PostingScoreDecoded(&bm25Meta,
																						idf,
																						avgDocLen,
																						posting,
																						docLens[nodeId]),
														term->matchBit);
					if (stats != NULL)
							stats->postingsDecoded++;
					}
				}

				UnlockReleaseBuffer(buf);
			postingsBlkno = nextBlkno;
			postingsOffno = nextOffno;
			CHECK_FOR_INTERRUPTS();
		}
	}

	TqHybridBm25ScoreDelta(&bm25Meta, deltaTerms, deltaTermCount,
						   terms, termCount, &acc, stats);

	if (resolvedTerms == 0 || acc.touchedCount == 0)
	{
		MemoryContextSwitchTo(oldCtx);
		return 0;
	}

	{
		uint32		oldTouchedCount = acc.touchedCount;
		uint32		out = 0;

		for (uint32 i = 0; i < oldTouchedCount; i++)
		{
			TqHybridBm25AccumulatorEntry *entry;

			entry = TqHybridBm25AccumulatorLookup(&acc, acc.touched[i].nodeId,
												  false);
			if (entry == NULL ||
				!TqHybridBm25MatchedQuery(query, terms, termCount,
										  entry->matchedTerms))
				continue;
			acc.touched[out] = acc.touched[i];
			acc.touched[out].score = entry->score;
			out++;
		}
		acc.touchedCount = out;
	}

	if (acc.touchedCount == 0)
	{
		MemoryContextSwitchTo(oldCtx);
		return 0;
	}

	qsort(acc.touched, acc.touchedCount, sizeof(TqHybridBm25NodeScore),
		  TqHybridBm25ScoreCompare);

	resultCount = Min((uint32) k, acc.touchedCount);
	*results = MemoryContextAllocZero(memoryContext,
									  sizeof(TqHybridBm25Result) * resultCount);
	for (int i = 0; i < resultCount; i++)
	{
		uint32		nodeId = acc.touched[i].nodeId;
		TqHybridBm25AccumulatorEntry *entry;

		entry = TqHybridBm25AccumulatorLookup(&acc, nodeId, false);
		if (entry == NULL)
			continue;

		(*results)[i].nodeId = nodeId;
		(*results)[i].heaptid = entry->hasDeltaDoc ? entry->heaptid :
			heapTids[nodeId];
		(*results)[i].bm25Score = acc.touched[i].score;
		(*results)[i].rank = i + 1;
	}

	if (stats != NULL)
	{
		if (stats->candidatesScored == 0)
			stats->candidatesScored = acc.touchedCount;
		stats->accumulatorEntries = acc.touchedCount;
		tqhybrid_last_bm25_decode_kernel = stats->decodeKernel;
		tqhybrid_last_bm25_score_kernel = stats->scoreKernel;
		tqhybrid_last_bm25_simd_blocks = stats->simdBlocks;
		tqhybrid_last_bm25_scalar_tail_postings = stats->scalarTailPostings;
	}

	MemoryContextSwitchTo(oldCtx);
	return resultCount;
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tq_debug_bm25_topk);
Datum
tq_debug_bm25_topk(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	TSQuery		query = PG_GETARG_TSQUERY(1);
	int32		k = PG_GETARG_INT32(2);
	bool		useWand = PG_GETARG_BOOL(3);
	Relation	index;
	TqHybridBm25Result *results;
	TqHybridBm25QueryStats stats;
	int			count;
	StringInfoData json;

	index = index_open(indexOid, AccessShareLock);
	count = TqHybridBm25TopK(index, query, k, useWand,
							 CurrentMemoryContext, &results, &stats);
	index_close(index, AccessShareLock);

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"result_count\":%d,"
					 "\"query_terms\":%u,"
					 "\"postings_decoded\":" UINT64_FORMAT ","
					 "\"blocks_visited\":" UINT64_FORMAT ","
					 "\"blocks_skipped\":" UINT64_FORMAT ","
					 "\"candidates_scored\":%u,"
					 "\"accumulator_entries\":%u,"
					 "\"cache_bytes\":" UINT64_FORMAT ","
					 "\"cache_lexicon_entries\":%u,"
					 "\"cache_hit\":%s,"
					 "\"cache_build_us\":" UINT64_FORMAT ","
					 "\"cache_docstats_loaded\":%s,"
					 "\"cache_liveness_loaded\":%s,"
					 "\"delta_blocks_visited\":" UINT64_FORMAT ","
					 "\"delta_postings_decoded\":" UINT64_FORMAT ","
					 "\"delta_cache_bytes\":" UINT64_FORMAT ","
					 "\"delta_cache_terms\":%u,"
					 "\"delta_cache_hit\":%s,"
					 "\"wand_iterations\":" UINT64_FORMAT ","
					 "\"wand_threshold_updates\":" UINT64_FORMAT ","
					 "\"wand_active_sorts\":" UINT64_FORMAT ","
					 "\"wand_heap_replacements\":" UINT64_FORMAT ","
					 "\"used_wand\":%s,"
					 "\"results\":[",
					 count,
					 stats.queryTerms,
					 stats.postingsDecoded,
					 stats.blocksVisited,
					 stats.blocksSkipped,
					 stats.candidatesScored,
					 stats.accumulatorEntries,
					 stats.cacheBytes,
					 stats.cacheLexiconEntries,
					 stats.cacheHit ? "true" : "false",
					 stats.cacheBuildUs,
					 stats.cacheDocstatsLoaded ? "true" : "false",
					 stats.cacheLivenessLoaded ? "true" : "false",
					 stats.deltaBlocksVisited,
					 stats.deltaPostingsDecoded,
					 stats.deltaCacheBytes,
					 stats.deltaCacheTerms,
					 stats.deltaCacheHit ? "true" : "false",
					 stats.wandIterations,
					 stats.wandThresholdUpdates,
					 stats.wandActiveSorts,
					 stats.wandHeapReplacements,
					 stats.usedWand ? "true" : "false");

	for (int i = 0; i < count; i++)
	{
		if (i > 0)
			appendStringInfoChar(&json, ',');
		appendStringInfo(&json,
						 "{\"node_id\":%u,\"rank\":%d,\"score_scaled\":%d}",
						 results[i].nodeId,
						 results[i].rank,
						 (int) rint(results[i].bm25Score * 1000.0));
	}
	appendStringInfoString(&json, "]}");

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}
