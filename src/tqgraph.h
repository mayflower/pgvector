#ifndef TQGRAPH_H
#define TQGRAPH_H

#include "postgres.h"

#include "hnsw.h"

#define TQ_GRAPH_CODE_TUPLE_TYPE		0x51
#define TQ_GRAPH_ADJ_TUPLE_TYPE			0x52
#define TQ_GRAPH_EXACT_TUPLE_TYPE		0x53
#define TQ_GRAPH_CORRECTION_TUPLE_TYPE	0x54
#define TQ_GRAPH_EXACT_SLAB_MAGIC		0x54514553U
#define TQ_GRAPH_NODE_DEAD				0x0001
#define TQ_GRAPH_TQ_PLUS				0x0001	/* metapage: ecShift/ecScale correction tuples present */
#define TQ_GRAPH_TQ_WEIGHTED			0x0002	/* metapage: code tuples carry per-vector ec_correction */
#define TQ_GRAPH_TQ_RENORM				0x0004	/* metapage: per-vector scale field stores renormalized l2 / centroid_norm instead of plain l2 */
#define TQ_GRAPH_EXACT_FREE				0x0008	/* metapage: exact vector slabs are omitted */
#define TQ_GRAPH_MAX_ENTRY_POINTS		16
#define TQ_GRAPH_ENTRY_SAMPLE_COUNT		608
#define TQ_GRAPH_MAX_NEIGHBORS			(HNSW_MAX_M * 2)
#define TQ_GRAPH_MAX_PAYLOADS			16
#define TQ_GRAPH_MAX_STORED_LEVEL		8
#define TQ_GRAPH_LOWBIT_L2_RESCORE_EF_MULT 6
#define TQ_GRAPH_LOWBIT_HIGHDIM_L2_TARGET_MULT 2
#define TQ_GRAPH_HIGHDIM_L2_RESCORE_EF_MULT 4
#define TQ_GRAPH_HIGHDIM_L2_TARGET_EF_MULT 8
#define TQ_GRAPH_HIGHDIM_ENTRY_SAMPLE_DIVISOR 4
#define TQ_GRAPH_TIGHT_L2_FILL_MULT		8
#define TQ_GRAPH_PAYLOAD_EXACT_MAX		1024
#define TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS	32
#define TQ_GRAPH_STACK_FRONTIER_CAPACITY	256
#define TQ_GRAPH_STACK_RESCORE_CAPACITY	256

#if defined(__GNUC__) || defined(__clang__)
#define TQ_GRAPH_PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 1)
#else
#define TQ_GRAPH_PREFETCH_READ(addr) ((void) 0)
#endif


/*
 * On-disk code tuple.
 *
 * The "data" flexible array holds the per-vector payloads followed by the
 * encoded code bytes.  When the index is built WITH (tq_weighted = on) and
 * the metapage carries TQ_GRAPH_TQ_WEIGHTED, a single additional float
 * (ec_correction = ⟨X+, M⟩, the per-vector renormalization scalar that
 * qdrant's TQ+ formula needs) is laid down BEFORE data[] — so data[]
 * starts 4 bytes later than in the legacy format.  Existing indexes (no
 * tq_weighted) are byte-for-byte unchanged.
 *
 * Use TqGraphTupleEcCorrection / TqGraphTupleSetEcCorrection /
 * TqGraphTuplePayloads / TqGraphTupleCode helpers — never index data[]
 * directly, since the offset depends on tqWeighted.
 */
typedef struct TqGraphCodeTupleData
{
	uint8		type;
	uint8		level;
	uint16		flags;
	uint32		nodeId;
	ItemPointerData heaptid;
	BlockNumber exactBlkno;
	OffsetNumber exactOffno;
	uint16		payloadMask;
	float		scale;
	float		norm;
	float		correction;	/* ||X+||² (cached at build/insert time) */
	uint8		data[FLEXIBLE_ARRAY_MEMBER];
} TqGraphCodeTupleData;

typedef TqGraphCodeTupleData *TqGraphCodeTuple;

typedef struct TqGraphAdjTupleData
{
	uint8		type;
	uint8		level;
	uint16		count;
	uint32		nodeId;
	uint32		neighbors[FLEXIBLE_ARRAY_MEMBER];
} TqGraphAdjTupleData;

typedef TqGraphAdjTupleData *TqGraphAdjTuple;

typedef struct TqGraphExactTupleData
{
	uint8		type;
	uint8		flags;
	uint16		unused;
	uint32		nodeId;
	char		vector[FLEXIBLE_ARRAY_MEMBER];
} TqGraphExactTupleData;

typedef TqGraphExactTupleData *TqGraphExactTuple;

typedef struct TqGraphExactSlabPageHeaderData
{
	uint32		magic;
	uint16		used;
	uint16		unused;
	uint32		capacity;
	char		data[FLEXIBLE_ARRAY_MEMBER];
} TqGraphExactSlabPageHeaderData;

typedef TqGraphExactSlabPageHeaderData *TqGraphExactSlabPageHeader;

typedef struct TqGraphCorrectionTupleData
{
	uint8		type;
	uint8		field;
	uint16		count;
	uint32		startDim;
	float		values[FLEXIBLE_ARRAY_MEMBER];
} TqGraphCorrectionTupleData;

typedef TqGraphCorrectionTupleData *TqGraphCorrectionTuple;

typedef struct TqGraphBuildNode
{
	ItemPointerData heaptid;
	Vector	   *vector;
	uint8	   *code;
	int32	   *payloads;
	uint16		payloadMask;
	int			level;
	float		scale;
	float		norm;
	float		correction;		/* ||X+||² */
	float		ecCorrection;	/* ⟨X+, M⟩ — only meaningful when state->tqWeighted */
	BlockNumber exactBlkno;
	OffsetNumber exactOffno;
	uint16		flags;
	uint32	  **neighbors;
	int		   *neighborCounts;
} TqGraphBuildNode;

typedef struct TqGraphBuildState
{
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	ForkNumber	forkNum;
	const HnswTypeInfo *typeInfo;
	HnswSupport support;
	MemoryContext ctx;
	TqGraphBuildNode *nodes;
	uint32		nodeCount;
	uint32		nodeCapacity;
	int			dimensions;
	int			m;
	int			efConstruction;
	int			tqBits;
	bool		tqWeighted;
	bool		tqQuantileFit;
	bool		tqRenorm;
	bool		tqExactStorage;
	bool		buildExactDistances;	/* short-circuit quantized fast paths during build */
	int			scoreMode;
	int			maxLevel;
	uint32		entryNodeId;
	int			payloadCount;
	Size		payloadBytes;
	double		reltuples;
	bool		building;
	float	   *ecShift;
	float	   *ecScale;
	int16	   *dPrimeSqI16;	/* TQ+ per-coord weights, quantized to i16 (SIMD path) */
	float		weightScale;	/* TQ+ quantization scale: D'² / max(D'²) · (INT16_MAX-1) */
	double		mmConst;		/* TQ+ Σ ecShift[d]², cached at fit time */
	uint64		buildDistanceWeighted;
	uint32	   *buildVisitedGeneration;
	uint32		buildVisitGeneration;
	MemoryContext buildQueryCtx;
	HnswTqQuery buildTq;
	uint32		buildQueryNodeId;
	bool		buildTqValid;
	uint64		buildDistanceCalls;
	uint64		buildDistanceQuerySplit;
	uint64		buildDistancePacked;
	uint64		buildDistanceCodeCode;
	uint64		buildDistanceExact;
	uint64		buildDistanceFallback;
} TqGraphBuildState;

typedef struct TqGraphResult
{
	ItemPointerData heaptid;
	uint32		nodeId;
	double		distance;
	bool		exactScored;
} TqGraphResult;

typedef struct TqGraphScanNode
{
	ItemPointerData heaptid;
	uint8	   *code;
	int32	   *payloads;
	uint16		payloadMask;
	char	   *exactVector;
	int			level;
	BlockNumber exactBlkno;
	OffsetNumber exactOffno;
	float		scale;
	float		norm;
	float		codeNorm;
	float		ecCorrection;	/* ⟨X+, M⟩, only meaningful when meta->tqFlags & TQ_GRAPH_TQ_WEIGHTED */
	uint16		flags;
	bool		loaded;
} TqGraphScanNode;

typedef struct TqGraphPayloadRef
{
	int16		payloadSlot;
	int32		payloadValue;
	uint32		nodeId;
} TqGraphPayloadRef;

typedef struct TqGraphScanStorage
{
	TqGraphScanNode *nodes;
	uint8	   *codeArena;
	uint8	   *payloadArena;
	char	   *exactArena;
	Size		exactBytes;
	uint32	  **neighbors;
	uint16	   *neighborCounts;
	uint32	   *visitedGeneration;
	uint32	   *visitGeneration;
	bool	   *codePagesLoaded;
	bool	   *adjPagesLoaded;
	BlockNumber *codeBlknos;
	TqGraphPayloadRef *payloadRefs;
	uint32		payloadRefCount;
	int			codeTuplesPerPage;
	int			codePageCount;
	int			adjPageCount;
	int			levelCount;
	bool		cached;
} TqGraphScanStorage;

typedef struct TqGraphNativeCache
{
	Oid			relid;
	Oid			relfilenumber;
	uint32		dimensions;
	uint16		m;
	uint16		graphMaxLevel;
	uint16		graphFlags;
	uint32		tqNodeCount;
	uint32		tqEntryNodeId;
	uint16		tqCodeBytes;
	uint16		tqBits;
	uint16		tqPayloadCount;
	uint16		tqPayloadBytes;
	BlockNumber tqCodeStartBlkno;
	BlockNumber tqAdjStartBlkno;
	BlockNumber tqExactStartBlkno;
	BlockNumber tqCorrectionStartBlkno;
	TqGraphScanStorage storage;
	MemoryContext ctx;
	struct TqGraphNativeCache *next;
} TqGraphNativeCache;

typedef struct TqGraphCorrectionCache
{
	Oid			relid;
	Oid			relfilenumber;
	uint32		dimensions;
	uint16		tqFlags;
	BlockNumber tqCorrectionStartBlkno;
	float	   *ecShift;
	float	   *ecScale;
	MemoryContext ctx;
	struct TqGraphCorrectionCache *next;
} TqGraphCorrectionCache;

typedef struct TqGraphFrontierItem
{
	uint32		nodeId;
	double		distance;
} TqGraphFrontierItem;

typedef struct TqGraphBuildOrderItem
{
	uint32		nodeId;
	uint64		key;
} TqGraphBuildOrderItem;

typedef struct TqGraphRescoreRef
{
	int			resultIndex;
	uint32		nodeId;
	BlockNumber blkno;
	OffsetNumber offno;
} TqGraphRescoreRef;

static inline Size
TqGraphCodeBytesForBits(int dimensions, int bits)
{
	return MAXALIGN(TqCodeSizeForBits(dimensions, bits));
}

static inline Size
TqGraphPayloadBytes(int payloadCount)
{
	return MAXALIGN(sizeof(int32) * payloadCount);
}

static inline Size
TqGraphTupleExtraHeaderBytes(bool tqWeighted)
{
	return tqWeighted ? sizeof(float) : 0;
}

static inline float
TqGraphTupleEcCorrection(TqGraphCodeTuple tuple, bool tqWeighted)
{
	float		value;

	if (!tqWeighted)
		return 0.0f;

	memcpy(&value, tuple->data, sizeof(float));
	return value;
}

static inline void
TqGraphTupleSetEcCorrection(TqGraphCodeTuple tuple, bool tqWeighted, float value)
{
	if (!tqWeighted)
		return;

	memcpy(tuple->data, &value, sizeof(float));
}

static inline int32 *
TqGraphTuplePayloads(TqGraphCodeTuple tuple, bool tqWeighted)
{
	return (int32 *) (tuple->data + TqGraphTupleExtraHeaderBytes(tqWeighted));
}

static inline uint8 *
TqGraphTupleCode(TqGraphCodeTuple tuple, Size payloadBytes, bool tqWeighted)
{
	return tuple->data + TqGraphTupleExtraHeaderBytes(tqWeighted) + payloadBytes;
}

static inline Size
TqGraphCodeTupleSize(int dimensions, int payloadCount, int bits, bool tqWeighted)
{
	return MAXALIGN(offsetof(TqGraphCodeTupleData, data) +
					TqGraphTupleExtraHeaderBytes(tqWeighted) +
					TqGraphPayloadBytes(payloadCount) +
					TqGraphCodeBytesForBits(dimensions, bits));
}

static inline Size
TqGraphAdjTupleSize(int count)
{
	return MAXALIGN(offsetof(TqGraphAdjTupleData, neighbors) + (sizeof(uint32) * count));
}

static inline int
TqGraphLevelM(int m, int level)
{
	return HnswGetLayerM(m, level);
}

static inline int
TqGraphLevelCapacity(int m)
{
	return Min(HnswGetMaxLevel(m), TQ_GRAPH_MAX_STORED_LEVEL) + 1;
}

static inline int
TqGraphAdjRecordCount(HnswMetaPageData *meta)
{
	return meta->tqNodeCount * TqGraphLevelCapacity(meta->m);
}

static inline int
TqGraphAdjSlot(HnswMetaPageData *meta, uint32 nodeId, int level)
{
	return nodeId * TqGraphLevelCapacity(meta->m) + level;
}

static inline int
TqGraphTuplesPerPage(Size tupleSize)
{
	Size		usable = BLCKSZ - MAXALIGN(SizeOfPageHeaderData) -
		MAXALIGN(sizeof(HnswPageOpaqueData));

	return Max(1, (int) (usable / (tupleSize + sizeof(ItemIdData))));
}

static inline int
TqGraphPageCount(uint32 nodeCount, int tuplesPerPage)
{
	if (nodeCount == 0)
		return 0;

	return (nodeCount + tuplesPerPage - 1) / tuplesPerPage;
}

Oid			TqGraphRelFileNumber(Relation index);
void		TqGraphInitBlockMap(BlockNumber *blknos, int count);
bool		TqGraphEnsureBlockMap(Relation index, BlockNumber startBlkno, int pageCount,
							  uint16 pageKind, BlockNumber *blknos);
BlockNumber TqGraphGetChainBlockNumber(Relation index, BlockNumber startBlkno,
									 int pageNo, int pageCount, uint16 pageKind);
BlockNumber TqGraphGetMappedBlockNumber(BlockNumber startBlkno, int pageNo,
									 BlockNumber *blknos);
bool		TqGraphResolveChainBlockNumber(Relation index, BlockNumber startBlkno,
										   int pageNo, int pageCount, uint16 pageKind,
										   BlockNumber *blknos, BlockNumber *blkno);
bool		TqGraphReadMeta(Relation index, HnswMetaPageData *meta);
void		TqGraphFinishPage(Buffer buf);
void		TqGraphAppendPage(Relation index, ForkNumber forkNum, Buffer *buf,
							Page *page, uint16 pageKind);
OffsetNumber TqGraphAppendTuple(Relation index, ForkNumber forkNum,
								BlockNumber *startBlkno, uint16 pageKind,
								Item tuple, Size tupleSize, uint16 graphOpKind,
								BlockNumber *insertBlkno);

bool		TqGraphLoadCorrection(Relation index, int dimensions,
							  float **ecShift, float **ecScale);
void		TqGraphCopyPayloadValues(TqGraphBuildState *state, int32 *payloads,
									 uint16 *payloadMask, Datum *values,
									 bool *isnull);
void		TqGraphExactRescore(Relation index, HnswScanOpaque so, Datum query,
								 HnswMetaPageData *meta,
								 TqGraphScanNode *nodes,
								 TqGraphResult *results, int count);
bool		TqGraphInsertValueInPlace(Relation index, IndexInfo *indexInfo,
									  ItemPointer heap_tid, Datum value,
									  Datum *values, bool *isnull);
int			TqGraphPickLevel(uint32 nodeId, int m);
int			TqGraphTraverse(Relation index, HnswScanOpaque so,
						 HnswMetaPageData *meta,
						 TqGraphScanStorage *storage,
						 TqGraphResult *results, int resultTarget,
						 int searchEf, Datum query, int payloadSlot,
						 int32 payloadValue);
void		TqGraphUpdateMetaPage(Relation index, TqGraphBuildState *state,
								  BlockNumber codeStart, BlockNumber adjStart,
								  BlockNumber exactStart,
								  BlockNumber correctionStart);
bool		TqGraphLoadCodePage(Relation index, HnswScanOpaque so,
								HnswMetaPageData *meta,
								TqGraphScanStorage *storage,
								uint32 nodeId);
bool		TqGraphLoadAdjPage(Relation index, HnswScanOpaque so,
							   HnswMetaPageData *meta,
							   TqGraphScanStorage *storage,
							   uint32 nodeId, int level);
void		TqGraphCollectVacuumStats(Relation index, HnswMetaPageData *meta,
									  int64 *liveNodes, int64 *deadNodes,
									  int64 *adjacencyRefs,
									  int64 *deadNeighborRefs);
bool		TqGraphPayloadRefRange(TqGraphScanStorage *storage, int payloadSlot,
								   int32 payloadValue, uint32 *firstIndex,
								   uint32 *refCount);
void		TqGraphAppendInsertedExact(Relation index, BlockNumber *exactStart,
									   uint32 nodeId, Vector *vector,
									   int dimensions, BlockNumber *exactBlkno,
									   OffsetNumber *exactOffno);
bool		TqGraphExactByteOffsetIsValid(OffsetNumber offno);
bool		TqGraphReadExactVectorInto(Relation index, TqGraphScanNode *node,
									   int dimensions, char *dest,
									   HnswScanOpaque so);
Vector	   *TqGraphReadExactVector(Relation index, TqGraphScanNode *node,
								   int dimensions);
BlockNumber TqGraphWriteExactPages(TqGraphBuildState *state);
void		TqGraphInvalidateCaches(Relation index);
void		TqGraphInitScanStorage(Relation index, HnswMetaPageData *meta,
								   TqGraphScanStorage *storage);
int64		TqGraphGetActiveLimitTupleTarget(void);
double		TqGraphGetActiveEstimatedFilterSelectivity(void);
bool		TqGraphGetActivePayloadInt4Filter(AttrNumber *heap_attno, int32 *value);
void		TqGraphSeedScanContext(HnswScanOpaque so, int64 tuple_target,
							   double estimated_filter_selectivity);

#endif
