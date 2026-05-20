#ifndef TQHYBRID_BM25_H
#define TQHYBRID_BM25_H

#include "postgres.h"

#include "nodes/execnodes.h"
#include "tsearch/ts_type.h"
#include "utils/rel.h"

typedef struct TqHybridBm25BuildDoc
{
	uint32		nodeId;
	ItemPointerData heaptid;
	uint32		docLen;
} TqHybridBm25BuildDoc;

typedef struct TqHybridBm25TermTuple
{
	uint64		termHash;
	uint32		nodeId;
	uint32		termOffset;
	uint16		tf;
	uint16		termLen;
} TqHybridBm25TermTuple;

typedef struct TqHybridBm25BuildStats
{
	Oid			indexOid;
	uint32		docCount;
	uint64		totalDocLen;
	uint32		uniqueTerms;
	uint32		termTupleCount;
	uint32		maxDocLen;
} TqHybridBm25BuildStats;

typedef struct TqHybridBm25Result
{
	uint32		nodeId;
	ItemPointerData heaptid;
	float8		bm25Score;
	int32		rank;
} TqHybridBm25Result;

typedef struct TqHybridBm25QueryStats
{
	uint32		queryTerms;
	uint64		postingsDecoded;
	uint64		blocksVisited;
	uint64		blocksSkipped;
	uint32		candidatesScored;
	uint32		accumulatorEntries;
	uint64		cacheBytes;
	uint32		cacheLexiconEntries;
	bool		cacheHit;
	uint64		cacheBuildUs;
	bool		cacheDocstatsLoaded;
	bool		cacheLivenessLoaded;
	uint64		deltaBlocksVisited;
	uint64		deltaPostingsDecoded;
	uint64		deltaCacheBytes;
	uint32		deltaCacheTerms;
	bool		deltaCacheHit;
	uint64		wandIterations;
	uint64		wandThresholdUpdates;
	uint64		wandActiveSorts;
	uint64		wandHeapReplacements;
	int			decodeKernel;
	int			scoreKernel;
	uint64		simdBlocks;
	uint64		scalarTailPostings;
	bool		usedWand;
} TqHybridBm25QueryStats;

typedef struct TqHybridBm25PlanningStats
{
	uint32		docCount;
	uint32		termCount;
	uint32		termTupleCount;
	uint32		deltaDocCount;
	uint32		deltaTermCount;
	uint32		postingsPages;
	uint32		blockMaxPages;
	uint32		deltaPages;
	bool		hasBm25;
} TqHybridBm25PlanningStats;

#define TQHYBRID_BM25_VERSION 1
#define TQHYBRID_BM25_META_TUPLE_TYPE		0x61
#define TQHYBRID_BM25_DOCSTATS_TUPLE_TYPE	0x62
#define TQHYBRID_BM25_LEXICON_TUPLE_TYPE	0x63
#define TQHYBRID_BM25_POSTINGS_TUPLE_TYPE	0x64
#define TQHYBRID_BM25_BLOCKMAX_TUPLE_TYPE	0x65
#define TQHYBRID_BM25_DELTA_TUPLE_TYPE		0x66
#define TQHYBRID_BM25_META_FLAG_TFNORM_Q16 0x0001
#define TQHYBRID_BM25_POSTINGS_ENCODING_MASK 0x00ff
#define TQHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16 0x8000
#define TQHYBRID_BM25_POSTINGS_ENCODING_DELTA_VARINT 1
#define TQHYBRID_BM25_POSTINGS_ENCODING_DELTA16 2
#define TQHYBRID_BM25_POSTINGS_ENCODING_OFFSET16 3

typedef struct TqHybridBm25MetaTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		reserved2;
	uint32		bm25Version;
	uint32		docCount;
	uint64		totalDocLen;
	float4		avgDocLen;
	float4		k1;
	float4		b;
	uint32		termCount;
	uint32		termTupleCount;
	uint32		maxDocLen;
	BlockNumber docStatsStartBlkno;
	BlockNumber lexiconStartBlkno;
	BlockNumber postingsStartBlkno;
	BlockNumber blockMaxStartBlkno;
	BlockNumber deltaStartBlkno;
	uint64		deltaGeneration;
	uint32		deltaDocCount;
	uint64		deltaTotalDocLen;
	uint32		deltaTermCount;
	uint32		postingsPages;
	uint32		blockMaxPages;
	uint32		deltaPages;
	uint64		lastCompactionGeneration;
	uint32		compactionCount;
} TqHybridBm25MetaTupleData;

typedef TqHybridBm25MetaTupleData *TqHybridBm25MetaTuple;

typedef struct TqBm25DocStat
{
	uint32		docLen;
	uint16		flags;
	uint16		reserved;
} TqBm25DocStat;

typedef struct TqHybridBm25DocStatsTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		count;
	uint32		startNodeId;
	TqBm25DocStat docs[FLEXIBLE_ARRAY_MEMBER];
} TqHybridBm25DocStatsTupleData;

typedef TqHybridBm25DocStatsTupleData *TqHybridBm25DocStatsTuple;

typedef struct TqHybridBm25Posting
{
	uint32		nodeId;
	uint16		tf;
	uint16		reserved;
} TqHybridBm25Posting;

typedef struct TqHybridBm25PostingsTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		count;
	uint32		termId;
	uint32		chunkNo;
	uint32		firstNodeId;
	uint32		lastNodeId;
	BlockNumber nextBlkno;
	OffsetNumber nextOffno;
	uint16		maxTf;
	uint16		encoding;
	uint16		payloadBytes;
	char		payload[FLEXIBLE_ARRAY_MEMBER];
} TqHybridBm25PostingsTupleData;

typedef TqHybridBm25PostingsTupleData *TqHybridBm25PostingsTuple;

typedef struct TqHybridBm25BlockMaxTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		count;
	uint32		termId;
	uint32		firstNodeId;
	uint32		lastNodeId;
	uint16		maxTf;
	uint16		reserved2;
	float4		maxScoreUpperBound;
} TqHybridBm25BlockMaxTupleData;

typedef TqHybridBm25BlockMaxTupleData *TqHybridBm25BlockMaxTuple;

typedef struct TqHybridBm25DeltaTerm
{
	uint64		termHash;
	uint32		termOffset;
	uint16		tf;
	uint16		termLen;
} TqHybridBm25DeltaTerm;

typedef struct TqHybridBm25DeltaTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		termCount;
	uint32		nodeId;
	ItemPointerData heaptid;
	uint32		docLen;
	uint32		termBytesLen;
	TqHybridBm25DeltaTerm terms[FLEXIBLE_ARRAY_MEMBER];
	/* followed by term bytes */
} TqHybridBm25DeltaTupleData;

typedef TqHybridBm25DeltaTupleData *TqHybridBm25DeltaTuple;

typedef struct TqHybridBm25LexiconEntryData
{
	uint8		type;
	uint8		reserved1;
	uint16		termLen;
	uint64		termHash;
	uint32		termId;
	uint32		df;
	uint32		cf;
	BlockNumber postingsBlkno;
	OffsetNumber postingsOffno;
	uint16		reserved2;
	uint32		postingsChunkCount;
	uint32		postingsBytes;
	BlockNumber blockMaxBlkno;
	OffsetNumber blockMaxOffno;
	uint16		reserved3;
	char		termBytes[FLEXIBLE_ARRAY_MEMBER];
} TqHybridBm25LexiconEntryData;

typedef TqHybridBm25LexiconEntryData *TqHybridBm25LexiconEntry;

void		TqHybridBm25BuildCollect(Relation heap, Relation index, IndexInfo *indexInfo);
void		TqHybridBm25AppendDelta(Relation index, uint32 nodeId,
									ItemPointer heapTid, Datum tsvectorDatum);
bool		TqHybridBm25MaybeCompact(Relation index);
void		TqHybridBm25InvalidateCache(Relation index);
bool		TqHybridBm25GetPlanningStats(Relation index,
										 TqHybridBm25PlanningStats *stats);
int			TqHybridBm25TopK(Relation index, TSQuery query, int32 k,
							  bool useWand, MemoryContext memoryContext,
							  TqHybridBm25Result **results,
							  TqHybridBm25QueryStats *stats);
Datum		tq_debug_bm25_stats(PG_FUNCTION_ARGS);
Datum		tq_debug_bm25_term_stats(PG_FUNCTION_ARGS);
Datum		tq_debug_bm25_topk(PG_FUNCTION_ARGS);

typedef enum TqHybridBm25Kernel
{
	TQHYBRID_BM25_KERNEL_SCALAR,
	TQHYBRID_BM25_KERNEL_NEON,
	TQHYBRID_BM25_KERNEL_AVX2,
	TQHYBRID_BM25_KERNEL_AVX512F
}			TqHybridBm25Kernel;

extern int	tqhybrid_last_bm25_decode_kernel;
extern int	tqhybrid_last_bm25_score_kernel;
extern uint64 tqhybrid_last_bm25_simd_blocks;
extern uint64 tqhybrid_last_bm25_scalar_tail_postings;

const char *TqHybridBm25KernelName(int kernel);

#endif
