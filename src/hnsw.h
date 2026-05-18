#ifndef HNSW_H
#define HNSW_H

#include "postgres.h"

#include <math.h>

#include "access/genam.h"
#include "access/parallel.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "port.h"				/* for random() */
#include "storage/bufpage.h"
#include "storage/condition_variable.h"
#include "storage/lwlock.h"
#include "storage/s_lock.h"
#include "utils/relptr.h"
#include "utils/sampling.h"
#include "vector.h"

#if PG_VERSION_NUM >= 190000
typedef Pointer Item;
#endif

#define HNSW_MAX_DIM 2000
#define HNSW_MAX_NNZ 1000

/* Support functions */
#define HNSW_DISTANCE_PROC 1
#define HNSW_NORM_PROC 2
#define HNSW_TYPE_INFO_PROC 3

#define HNSW_VERSION	1
#define HNSW_MAGIC_NUMBER 0xA953A953
#define HNSW_PAGE_ID	0xFF90

/* Preserved page numbers */
#define HNSW_METAPAGE_BLKNO	0
#define HNSW_HEAD_BLKNO		1	/* first element page */

/* Must correspond to page numbers since page lock is used */
#define HNSW_UPDATE_LOCK 	0
#define HNSW_SCAN_LOCK		1

/* HNSW parameters */
#define HNSW_DEFAULT_M	16
#define HNSW_MIN_M	2
#define HNSW_MAX_M		100
#define HNSW_DEFAULT_EF_CONSTRUCTION	64
#define HNSW_MIN_EF_CONSTRUCTION	4
#define HNSW_MAX_EF_CONSTRUCTION		1000
#define HNSW_DEFAULT_EF_SEARCH	40
#define HNSW_MIN_EF_SEARCH		1
#define HNSW_MAX_EF_SEARCH		1000

/* Turboquant graph routing defaults */
#define TQ_DEFAULT_GRAPH_EF_CONSTRUCTION 128
#define TQ_DEFAULT_GRAPH_EF_SEARCH 64
#define TQ_DEFAULT_GRAPH_OVERSAMPLING 4
#define TQ_GRAPH_EXACT_CACHE_AUTO_MAX_BYTES (16 * 1024 * 1024)
#define TQ_DEFAULT_BITS 4
#define TQ_BITS 4
#define TQ_LUT_WIDTH 16
#define TQ_CODE_SIZE_BITS(dim, bits) ((((dim) * (bits)) + 7) / 8)
#define TQ_CODE_SIZE(dim) (((dim) + 1) / 2)
#define TQ_CODE_SCALE_OFFSET(dim) TQ_CODE_SIZE(dim)
#define TQ_CODE_PAYLOAD_SIZE(dim) (TQ_CODE_SIZE(dim) + sizeof(float))
#define TQ_LUT_SIZE(dim) ((dim) * TQ_LUT_WIDTH * sizeof(float))
#ifndef TQ_GRAPH_ENABLE_SYMMETRIC_I8_DOT
#define TQ_GRAPH_ENABLE_SYMMETRIC_I8_DOT 0
#endif

/* Tuple types */
#define HNSW_ELEMENT_TUPLE_TYPE  1
#define HNSW_NEIGHBOR_TUPLE_TYPE 2

/* Page and storage identities */
#define HNSW_STORAGE_HNSW				0
#define HNSW_STORAGE_TURBOQUANT_GRAPH	1
#define HNSW_STORAGE_TURBOQUANT_FLAT	2
#define HNSW_STORAGE_TURBOQUANT_IVF		3
#define HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE	4
#define HNSW_PAGE_KIND_GRAPH			1
#define HNSW_PAGE_KIND_META				2
#define HNSW_PAGE_KIND_TQ_CODE			3
#define HNSW_PAGE_KIND_TQ_ADJ			4
#define HNSW_PAGE_KIND_TQ_EXACT			5
#define HNSW_PAGE_KIND_TQ_CORRECTION	6
#define HNSW_PAGE_KIND_MASK				0x00ff
#define HNSW_PAGE_GRAPH_OP_SHIFT		8

#define HNSW_GRAPH_OP_NONE				0
#define HNSW_GRAPH_OP_PAGE_INIT			1
#define HNSW_GRAPH_OP_PAGE_LINK			2
#define HNSW_GRAPH_OP_META_UPDATE		3
#define HNSW_GRAPH_OP_ELEMENT_INSERT	4
#define HNSW_GRAPH_OP_NEIGHBOR_INSERT	5
#define HNSW_GRAPH_OP_NEIGHBOR_UPDATE	6
#define HNSW_GRAPH_OP_DUPLICATE_HEAPTID	7
#define HNSW_GRAPH_OP_VACUUM_DELETE		8
#define HNSW_GRAPH_OP_VACUUM_REPAIR		9

/* Make graph robust against non-HOT updates */
#define HNSW_HEAPTIDS 10

#define HNSW_UPDATE_ENTRY_GREATER 1
#define HNSW_UPDATE_ENTRY_ALWAYS 2

/* Build phases */
/* PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE is 1 */
#define PROGRESS_HNSW_PHASE_LOAD		2

#define HNSW_MAX_SIZE (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - sizeof(ItemIdData))
#define HNSW_TUPLE_ALLOC_SIZE BLCKSZ

#define HNSW_ELEMENT_TUPLE_SIZE(size)	MAXALIGN(offsetof(HnswElementTupleData, data) + (size))
#define HNSW_NEIGHBOR_TUPLE_SIZE(level, m)	MAXALIGN(offsetof(HnswNeighborTupleData, indextids) + ((level) + 2) * (m) * sizeof(ItemPointerData))

#define HNSW_NEIGHBOR_ARRAY_SIZE(lm)	(offsetof(HnswNeighborArray, items) + sizeof(HnswCandidate) * (lm))

#define HnswPageGetOpaque(page)	((HnswPageOpaque) PageGetSpecialPointer(page))
#define HnswPageGetMeta(page)	((HnswMetaPageData *) PageGetContents(page))

#if PG_VERSION_NUM >= 150000
#define RandomDouble() pg_prng_double(&pg_global_prng_state)
#define SeedRandom(seed) pg_prng_seed(&pg_global_prng_state, seed)
#else
#define RandomDouble() (((double) random()) / MAX_RANDOM_VALUE)
#define SeedRandom(seed) srandom(seed)
#endif

#define HnswIsElementTuple(tup) ((tup)->type == HNSW_ELEMENT_TUPLE_TYPE)
#define HnswIsNeighborTuple(tup) ((tup)->type == HNSW_NEIGHBOR_TUPLE_TYPE)

/* 2 * M connections for ground layer */
#define HnswGetLayerM(m, layer) (layer == 0 ? (m) * 2 : (m))

/* Optimal ML from paper */
#define HnswGetMl(m) (1 / log(m))

/* Ensure fits on page and in uint8 */
#define HnswGetMaxLevel(m) Min(((BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - offsetof(HnswNeighborTupleData, indextids) - sizeof(ItemIdData)) / (sizeof(ItemPointerData)) / (m)) - 2, 255)

#define HnswGetSearchCandidate(membername, ptr) pairingheap_container(HnswSearchCandidate, membername, ptr)
#define HnswGetSearchCandidateConst(membername, ptr) pairingheap_const_container(HnswSearchCandidate, membername, ptr)

#define HnswGetValue(base, element) PointerGetDatum(HnswPtrAccess(base, (element)->value))

#if PG_VERSION_NUM < 140005
#define relptr_offset(rp) ((rp).relptr_off - 1)
#endif

/* Pointer macros */
#define HnswPtrAccess(base, hp) ((base) == NULL ? (hp).ptr : relptr_access(base, (hp).relptr))
#define HnswPtrStore(base, hp, value) ((base) == NULL ? (void) ((hp).ptr = (value)) : (void) relptr_store(base, (hp).relptr, value))
#define HnswPtrIsNull(base, hp) ((base) == NULL ? (hp).ptr == NULL : relptr_is_null((hp).relptr))
#define HnswPtrEqual(base, hp1, hp2) ((base) == NULL ? (hp1).ptr == (hp2).ptr : relptr_offset((hp1).relptr) == relptr_offset((hp2).relptr))

/* For code paths dedicated to each type */
#define HnswPtrPointer(hp) (hp).ptr
#define HnswPtrOffset(hp) relptr_offset((hp).relptr)

/* Variables */
extern int	hnsw_ef_search;
extern int	hnsw_iterative_scan;
extern int	hnsw_max_scan_tuples;
extern double hnsw_scan_mem_multiplier;
extern bool hnsw_tq_graph_prefetch;
extern bool hnsw_tq_graph_stack_scratch;
extern bool hnsw_tq_graph_lowbit_popcnt;
extern bool hnsw_tq_graph_i8mm;
extern bool hnsw_tq_graph_avxvnni;
extern bool hnsw_tq_graph_avx512vnni;
extern bool hnsw_tq_graph_avx512vpopcntdq;
extern bool hnsw_tq_weighted;
extern bool hnsw_tq_renorm;
extern bool hnsw_tq_query_1bit_asymmetric;
extern int	hnsw_tq_query_1bit_asymmetric_bits;
extern bool hnsw_tq_build_exact_distances;
extern bool hnsw_tq_hadamard_simd;
extern bool hnsw_tq_exact_avx512;
extern int	hnsw_tq_graph_lookahead_prefetch;
extern int	hnsw_tq_graph_lookahead_threshold_kb;
extern int	hnsw_lock_tranche_id;

typedef enum TqRoutingMode
{
	TQ_ROUTING_AUTO = 0,
	TQ_ROUTING_GRAPH = 1,
	TQ_ROUTING_IVF = 2,
	TQ_ROUTING_FLAT = 3,
	TQ_ROUTING_LEGACY_HNSW = 4
}			TqRoutingMode;

typedef enum TqGraphRescoreBand
{
	TQ_GRAPH_RESCORE_BAND_AUTO,
	TQ_GRAPH_RESCORE_BAND_NONE,
	TQ_GRAPH_RESCORE_BAND_EXACT
}			TqGraphRescoreBand;

typedef enum TqGraphExactCache
{
	TQ_GRAPH_EXACT_CACHE_AUTO,
	TQ_GRAPH_EXACT_CACHE_OFF,
	TQ_GRAPH_EXACT_CACHE_ON
}			TqGraphExactCache;

typedef enum TqGraphReorder
{
	TQ_GRAPH_REORDER_AUTO,
	TQ_GRAPH_REORDER_OFF,
	TQ_GRAPH_REORDER_BFS
}			TqGraphReorder;

typedef enum TqGraphLookaheadPrefetch
{
	TQ_GRAPH_LOOKAHEAD_AUTO,	/* size-gated: on iff metadata working-set > threshold */
	TQ_GRAPH_LOOKAHEAD_OFF,
	TQ_GRAPH_LOOKAHEAD_ON
}			TqGraphLookaheadPrefetch;

typedef enum TqScoreMode
{
	TQ_SCORE_L2,
	TQ_SCORE_IP,
	TQ_SCORE_COSINE,
	TQ_SCORE_L1
}			TqScoreMode;

typedef enum TqScoringKernel
{
	TQ_SCORING_SCALAR,
	TQ_SCORING_AVX2,
	TQ_SCORING_AVX512VNNI,
	TQ_SCORING_AVXVNNI,
	TQ_SCORING_ARM_I8MM,
	TQ_SCORING_NEON
}			TqScoringKernel;

typedef enum HnswIterativeScanMode
{
	HNSW_ITERATIVE_SCAN_OFF,
	HNSW_ITERATIVE_SCAN_RELAXED,
	HNSW_ITERATIVE_SCAN_STRICT
}			HnswIterativeScanMode;

typedef struct HnswElementData HnswElementData;
typedef struct HnswNeighborArray HnswNeighborArray;

#define HnswPtrDeclare(type, relptrtype, ptrtype) \
	relptr_declare(type, relptrtype); \
	typedef union { type *ptr; relptrtype relptr; } ptrtype

/* Pointers that can be absolute or relative */
/* Use char for DatumPtr so works with Pointer */
HnswPtrDeclare(HnswElementData, HnswElementRelptr, HnswElementPtr);
HnswPtrDeclare(HnswNeighborArray, HnswNeighborArrayRelptr, HnswNeighborArrayPtr);
HnswPtrDeclare(HnswNeighborArrayPtr, HnswNeighborsRelptr, HnswNeighborsPtr);
HnswPtrDeclare(char, DatumRelptr, DatumPtr);

struct HnswElementData
{
	HnswElementPtr next;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	uint8		heaptidsLength;
	uint8		level;
	uint8		deleted;
	uint8		version;
	uint32		hash;
	HnswNeighborsPtr neighbors;
	BlockNumber blkno;
	OffsetNumber offno;
	OffsetNumber neighborOffno;
	BlockNumber neighborPage;
	DatumPtr	value;
	LWLock		lock;
};

typedef HnswElementData * HnswElement;

typedef struct HnswCandidate
{
	HnswElementPtr element;
	float		distance;
	bool		closer;
}			HnswCandidate;

struct HnswNeighborArray
{
	int			length;
	bool		closerSet;
	HnswCandidate items[FLEXIBLE_ARRAY_MEMBER];
};

typedef struct HnswSearchCandidate
{
	pairingheap_node c_node;
	pairingheap_node w_node;
	HnswElementPtr element;
	double		distance;
}			HnswSearchCandidate;

/* HNSW index options */
typedef struct HnswOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* number of connections */
	int			efConstruction; /* size of dynamic candidate list */
}			HnswOptions;

/* Keep the first fields layout-compatible with HnswOptions */
typedef struct TqOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* graph_m */
	int			efConstruction; /* graph_ef_construction */
	int			routing;
	int			graphEfSearch;
	int			graphOversampling;
	int			graphRescoreBand;
	int			graphExactCache;
	int			graphReorder;
	int			tqBits;
	bool		tqWeighted;		/* tq_weighted: opt into TurboQuant+ per-coord weighted scoring. Distinct from the JSON metadata key "tq_plus" which only indicates that error-correction shift/scale arrays exist on disk. */
	bool		tqQuantileFit;	/* tq_quantile_fit: replace Welford ecShift/ecScale fit with qdrant-style quantile-anchored fit. */
	bool		tqRenorm;		/* tq_renorm: opt into TurboQuant+ renormalization residual correction at encode time. Replaces per-vector pre-quant L2 length with l2_length / centroid_norm for Dot/Cosine. */
	bool		tqExactStorage; /* tq_exact_storage: store full exact vectors for final rescoring. */
}			TqOptions;

typedef struct HnswGraph
{
	/* Graph state */
	slock_t		lock;
	HnswElementPtr head;
	double		indtuples;

	/* Entry state */
	LWLock		entryLock;
	LWLock		entryWaitLock;
	HnswElementPtr entryPoint;

	/* Allocations state */
	LWLock		allocatorLock;
	Size		memoryUsed;
	Size		memoryTotal;

	/* Flushed state */
	LWLock		flushLock;
	bool		flushed;
}			HnswGraph;

typedef struct HnswShared
{
	/* Immutable state */
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;

	/* Worker progress */
	ConditionVariable workersdonecv;

	/* Mutex for mutable state */
	slock_t		mutex;

	/* Mutable state */
	int			nparticipantsdone;
	double		reltuples;
	HnswGraph	graphData;
}			HnswShared;

#define ParallelTableScanFromHnswShared(shared) \
	(ParallelTableScanDesc) ((char *) (shared) + BUFFERALIGN(sizeof(HnswShared)))

typedef struct HnswLeader
{
	ParallelContext *pcxt;
	int			nparticipanttuplesorts;
	HnswShared *hnswshared;
	Snapshot	snapshot;
	char	   *hnswarea;
}			HnswLeader;

typedef struct HnswAllocator
{
	void	   *(*alloc) (Size size, void *state);
	void	   *state;
}			HnswAllocator;

typedef struct HnswTypeInfo
{
	int			maxDimensions;
	Datum		(*normalize) (PG_FUNCTION_ARGS);
	void		(*checkValue) (Pointer v);
}			HnswTypeInfo;

typedef struct HnswSupport
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
}			HnswSupport;

typedef struct HnswQuery
{
	Datum		value;
}			HnswQuery;

typedef struct HnswTqQuery
{
	uint8	   *code;
#if TQ_GRAPH_ENABLE_SYMMETRIC_I8_DOT
	int8	   *queryI8;
#endif
	float	   *lut;
	float	   *queryValues;
	float	   *rawQueryValues;
	float	   *ecShift;
	float	   *ecScale;
	uint8	   *querySignBits;
	/*
	 * Asymmetric 1-bit query encoding.  Bit-plane-decomposed
	 * 8-bit signed query quantization, laid out in 128-dim blocks of
	 * `8 * 16` bytes (8 planes of 16 bytes each).  Recovers query
	 * magnitude information that the symmetric `querySignBits` path
	 * discards.  Populated by TqPrepareQueryAsymBit1 only when
	 * hnsw.tq_query_1bit_asymmetric is on and tq_bits = 1.
	 */
	uint8	   *queryPlanes;
	int64		queryAsymSumSigned;		/* Σ q_signed over all dims (full + tail) */
	float		queryAsymScale;			/* c / q_scale — postprocess multiplier  */
	int			queryAsymNumFullBlocks;	/* full 128-dim blocks                    */
	int			queryAsymTailBytes;		/* 0..15: bytes in trailing partial block */
	int			queryAsymBits;			/* BITS captured at precompute (8/12/16) */
#if defined(__aarch64__) || defined(_M_ARM64) || \
	defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
	int8	   *querySplitLow;
	int8	   *querySplitHigh;
	uint8	   *querySplitLowU8;
	uint8	   *querySplitHighU8;
	int8		querySplitTailLow[16];
	int8		querySplitTailHigh[16];
	uint8		querySplitTailLowU8[16];
	uint8		querySplitTailHighU8[16];
	int			querySplitChunks;
	int			querySplitTailDims;
	float		querySplitPostprocessScale;
	bool		querySplitEnabled;
#endif
#if TQ_GRAPH_ENABLE_SYMMETRIC_I8_DOT
	float		queryScale;
	float		queryCodeNorm;
#endif
	int			dimensions;
	int			bits;
	int			lutWidth;
	Size		codeBytes;
	int			scoreMode;
	int			scoringKernel;
	double		queryNorm;
	double		ecCorrection;
	bool		enabled;
}			HnswTqQuery;

typedef struct HnswBuildState
{
	/* Info */
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	ForkNumber	forkNum;
	const		HnswTypeInfo *typeInfo;

	/* Settings */
	int			dimensions;
	int			m;
	int			efConstruction;

	/* Statistics */
	double		indtuples;
	double		reltuples;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	HnswGraph	graphData;
	HnswGraph  *graph;
	double		ml;
	int			maxLevel;

	/* Memory */
	MemoryContext graphCtx;
	MemoryContext tmpCtx;
	HnswAllocator allocator;

	/* Parallel builds */
	HnswLeader *hnswleader;
	HnswShared *hnswshared;
	char	   *hnswarea;
}			HnswBuildState;

typedef struct HnswMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint32		dimensions;
	uint16		m;
	uint16		efConstruction;
	uint16		storageKind;
	uint16		graphEfSearch;
	uint16		graphOversampling;
	uint16		graphRescoreBand;
	uint16		graphMaxLevel;
	uint16		graphFlags;
	BlockNumber entryBlkno;
	OffsetNumber entryOffno;
	int16		entryLevel;
	BlockNumber insertPage;
	uint32		tqNodeCount;
	uint32		tqEntryNodeId;
	uint16		tqCodeBytes;
	uint16		tqPayloadCount;
	uint16		tqPayloadBytes;
	uint16		tqFlags;
	uint16		tqBits;
	BlockNumber tqCodeStartBlkno;
	BlockNumber tqAdjStartBlkno;
	BlockNumber tqExactStartBlkno;
	BlockNumber tqCorrectionStartBlkno;
}			HnswMetaPageData;

typedef HnswMetaPageData * HnswMetaPage;

typedef struct HnswPageOpaqueData
{
	BlockNumber nextblkno;
	uint16		pageKind;
	uint16		page_id;		/* for identification of HNSW indexes */
}			HnswPageOpaqueData;

typedef HnswPageOpaqueData * HnswPageOpaque;

typedef struct HnswElementTupleData
{
	uint8		type;
	uint8		level;
	uint8		deleted;
	uint8		version;
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	ItemPointerData neighbortid;
	uint16		unused;
	Vector		data;
}			HnswElementTupleData;

typedef HnswElementTupleData * HnswElementTuple;

typedef struct HnswNeighborTupleData
{
	uint8		type;
	uint8		version;
	uint16		count;
	ItemPointerData indextids[FLEXIBLE_ARRAY_MEMBER];
}			HnswNeighborTupleData;

typedef HnswNeighborTupleData * HnswNeighborTuple;

typedef union
{
	struct pointerhash_hash *pointers;
	struct offsethash_hash *offsets;
	struct tidhash_hash *tids;
}			visited_hash;

typedef union
{
	HnswElement element;
	ItemPointerData indextid;
}			HnswUnvisited;

typedef struct HnswScanOpaqueData
{
	const		HnswTypeInfo *typeInfo;
	bool		first;
	List	   *w;
	visited_hash v;
	pairingheap *discarded;
	HnswQuery	q;
	HnswTqQuery tq;
	int			m;
	int			efSearch;
	int			graphOversampling;
	int			graphRescoreBand;
	int64		tuples;
	int64		returnedRows;
	int64		tupleTargetRows;
	double		estimatedFilterSelectivity;
	int			initialEffectiveEfSearch;
	bool		hasTupleTargetRows;
	bool		hasEstimatedFilterSelectivity;
	bool		hasInitialEffectiveEfSearch;
	double		previousDistance;
	Size		maxMemory;
	MemoryContext tmpCtx;
	int64		graphVisitedNodes;
	int64		graphScoredCodes;
	int64		graphCandidateCount;
	int64		graphRescoreCount;
	int64		graphRescorePages;
	int64		graphCodePagesRead;
	int64		graphAdjPagesRead;
	int64		graphEntryPointCount;
	int64		graphPrepareUs;
	int64		graphTraverseUs;
	int64		graphFillUs;
	int64		graphRescoreUs;
	int64		graphSortUs;
	int64		graphTotalUs;
	int			graphStorageKind;
	bool		turboquantGraphScan;
	bool		turboquantFlatScan;
	void	   *tqGraphResults;
	int			tqGraphResultCount;
	int			tqGraphResultIndex;

	/* Support functions */
	HnswSupport support;
}			HnswScanOpaqueData;

typedef HnswScanOpaqueData * HnswScanOpaque;

typedef struct HnswVacuumState
{
	/* Info */
	Relation	index;
	IndexBulkDeleteResult *stats;
	IndexBulkDeleteCallback callback;
	void	   *callback_state;

	/* Settings */
	int			m;
	int			efConstruction;

	/* Support functions */
	HnswSupport support;

	/* Variables */
	struct tidhash_hash *deleted;
	BufferAccessStrategy bas;
	HnswNeighborTuple ntup;
	HnswElementData highestPoint;

	/* Memory */
	MemoryContext tmpCtx;
}			HnswVacuumState;

/* Methods */
int			HnswGetM(Relation index);
int			HnswGetEfConstruction(Relation index);
int			HnswGetEfSearch(Relation index);
int			HnswGetGraphOversampling(Relation index);
int			HnswGetGraphRescoreBand(Relation index);
int			HnswGetGraphExactCache(Relation index);
int			HnswGetGraphReorder(Relation index);
bool		HnswIsTurboquantIndex(Relation index);
bool		HnswUseTqGraph(Relation index);
bool		HnswUseTqNativeGraph(Relation index);
bool		HnswUseTqFlat(Relation index);
bool		HnswUseTqCodes(Relation index);
void		HnswSetForceTurboquantIndex(bool force);
Size		HnswElementTupleSize(Relation index, Pointer value);
FmgrInfo   *HnswOptionalProcInfo(Relation index, uint16 procnum);
void		HnswInitSupport(HnswSupport * support, Relation index);
Datum		HnswNormValue(const HnswTypeInfo * typeInfo, Oid collation, Datum value);
bool		HnswCheckNorm(HnswSupport * support, Datum value);
Buffer		HnswNewBuffer(Relation index, ForkNumber forkNum);
void		HnswInitPage(Buffer buf, Page page);
void		HnswInitPageKind(Buffer buf, Page page, uint16 pageKind);
void		HnswMarkPageGraphOp(Page page, uint16 graphOpKind);
void		HnswInit(void);
void		TqGraphControlInit(void);
void		HnswLogGraphWalRecord(Relation index, ForkNumber forkNum, BlockNumber blkno, uint16 graphOpKind);
const char *HnswGraphWalModeName(void);
bool		HnswGraphCustomWalEnabled(void);
void		HnswRecordGraphScanStats(HnswScanOpaque so);
void		HnswRecordNonGraphScanStats(void);
void		HnswRecordFlatScanStats(void);
const char *HnswTqScoringKernelName(int scoringKernel);
const char *HnswStorageKindName(int storageKind);
int			HnswGetTqBits(Relation index);
bool		HnswGetTqWeightedOption(Relation index);
bool		HnswGetTqRenormOption(Relation index);
bool		HnswGetTqQuantileFitOption(Relation index);
bool		HnswGetTqExactStorageOption(Relation index);
void		HnswPrepareTqQuery(Relation index, HnswSupport * support, Datum value, HnswTqQuery * tq);
void		HnswPrepareTqBuildQuery(Relation index, HnswSupport * support, Datum value, HnswTqQuery * tq);
void		HnswPrepareTqBuildQueryWithCorrection(Relation index, HnswSupport * support, Datum value, HnswTqQuery * tq,
												  const float *ecShift, const float *ecScale);
float		TqPreprocessVector(Vector *vector, double *rotated);
Size		TqCodeSizeForBits(int dimensions, int bits);
int			TqGetCodeComponentBits(const uint8 *code, int i, int bits);
float		TqGetCodeCenterBits(int code, int bits);
float		TqEncodeVectorBits(Vector *vector, uint8 *code, int bits);
float		TqEncodeVectorWithCorrectionBits(Vector *vector, uint8 *code, int bits,
											const float *ecShift, const float *ecScale);
float		TqEncodeVectorWithCorrectionAndXmBits(Vector *vector, uint8 *code, int bits,
											   const float *ecShift, const float *ecScale,
											   float *xmOut);
float		TqEncodeVectorWithCorrectionXmRenormBits(Vector *vector, uint8 *code, int bits,
												  const float *ecShift, const float *ecScale,
												  float *xmOut, float *centroidNormOut);
float		TqEncodeVector(Vector *vector, uint8 *code);
float		TqEncodeVectorWithCorrection(Vector *vector, uint8 *code,
										const float *ecShift, const float *ecScale);
double		TqCodeDistance(const HnswTqQuery *tq, const uint8 *valueCode, float valueScale);
int64		HnswRescoreSearchCandidates(Relation index, HnswSupport * support, HnswQuery * q, List *items);
List	   *HnswSearchLayer(char *base, HnswQuery * q, List *ep, int ef, int lc, Relation index, HnswSupport * support, int m, bool inserting, HnswElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples, int64 tupleLimit, int64 *scoredCodes, HnswTqQuery * tq);
HnswElement HnswGetEntryPoint(Relation index);
void		HnswGetMetaPageInfo(Relation index, int *m, HnswElement * entryPoint);
int			HnswGetMetaPageStorageKind(Relation index);
void	   *HnswAlloc(HnswAllocator * allocator, Size size);
HnswElement HnswInitElement(char *base, ItemPointer tid, int m, double ml, int maxLevel, HnswAllocator * alloc);
HnswElement HnswInitElementFromBlock(BlockNumber blkno, OffsetNumber offno);
void		HnswFindElementNeighbors(char *base, HnswElement element, HnswElement entryPoint, Relation index, HnswSupport * support, int m, int efConstruction, bool existing);
HnswSearchCandidate *HnswEntryCandidate(char *base, HnswElement entryPoint, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec);
void		HnswUpdateMetaPage(Relation index, int updateEntry, HnswElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building);
void		HnswSetNeighborTuple(char *base, HnswNeighborTuple ntup, HnswElement e, int m);
void		HnswAddHeapTid(HnswElement element, ItemPointer heaptid);
HnswNeighborArray *HnswInitNeighborArray(int lm, HnswAllocator * allocator);
void		HnswInitNeighbors(char *base, HnswElement element, int m, HnswAllocator * alloc);
bool		HnswInsertTupleOnDisk(Relation index, HnswSupport * support, Datum value, ItemPointer heaptid, bool building);
void		HnswUpdateNeighborsOnDisk(Relation index, HnswSupport * support, HnswElement e, int m, bool checkExisting, bool building);
void		HnswLoadElementFromTuple(HnswElement element, HnswElementTuple etup, bool loadHeaptids, bool loadVec);
void		HnswLoadElement(HnswElement element, double *distance, HnswQuery * q, Relation index, HnswSupport * support, bool loadVec, double *maxDistance);
bool		HnswFormIndexValue(Datum *out, Datum *values, bool *isnull, const HnswTypeInfo * typeInfo, HnswSupport * support);
void		HnswSetElementTuple(Relation index, char *base, HnswElementTuple etup, HnswElement element);
void		HnswUpdateConnection(char *base, HnswNeighborArray * neighbors, HnswElement newElement, float distance, int lm, int *updateIdx, Relation index, HnswSupport * support);
bool		HnswLoadNeighborTids(HnswElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc);
void		HnswInitLockTranche(void);
const		HnswTypeInfo *HnswGetTypeInfo(Relation index);
PGDLLEXPORT void HnswParallelBuildMain(dsm_segment *seg, shm_toc *toc);

/* Index access methods */
IndexBuildResult *hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo);
IndexBuildResult *turboquantbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		hnswbuildempty(Relation index);
void		turboquantbuildempty(Relation index);
bool		hnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
bool		turboquantinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
IndexBulkDeleteResult *hnswbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *hnswvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
IndexScanDesc hnswbeginscan(Relation index, int nkeys, int norderbys);
IndexScanDesc turboquantbeginscan(Relation index, int nkeys, int norderbys);
void		hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
void		turboquantrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool		hnswgettuple(IndexScanDesc scan, ScanDirection dir);
bool		turboquantgettuple(IndexScanDesc scan, ScanDirection dir);
void		hnswendscan(IndexScanDesc scan);
void		turboquantendscan(IndexScanDesc scan);

IndexBuildResult *tqgraphbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		tqgraphbuildempty(Relation index);
bool		tqgraphinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
IndexBulkDeleteResult *tqgraphbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *tqgraphvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
IndexScanDesc tqgraphbeginscan(Relation index, int nkeys, int norderbys);
void		tqgraphrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool		tqgraphgettuple(IndexScanDesc scan, ScanDirection dir);
void		tqgraphendscan(IndexScanDesc scan);

Datum		turboquanthandler(PG_FUNCTION_ARGS);
Datum		tq_last_scan_stats(PG_FUNCTION_ARGS);
Datum		tq_index_stats(PG_FUNCTION_ARGS);

static inline HnswNeighborArray *
HnswGetNeighbors(char *base, HnswElement element, int lc)
{
	HnswNeighborArrayPtr *neighborList = HnswPtrAccess(base, element->neighbors);

	Assert(element->level >= lc);

	return HnswPtrAccess(base, neighborList[lc]);
}

/* Hash tables */
typedef struct TidHashEntry
{
	ItemPointerData tid;
	char		status;
}			TidHashEntry;

#define SH_PREFIX tidhash
#define SH_ELEMENT_TYPE TidHashEntry
#define SH_KEY_TYPE ItemPointerData
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct PointerHashEntry
{
	uintptr_t	ptr;
	char		status;
}			PointerHashEntry;

#define SH_PREFIX pointerhash
#define SH_ELEMENT_TYPE PointerHashEntry
#define SH_KEY_TYPE uintptr_t
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct OffsetHashEntry
{
	Size		offset;
	char		status;
}			OffsetHashEntry;

#define SH_PREFIX offsethash
#define SH_ELEMENT_TYPE OffsetHashEntry
#define SH_KEY_TYPE Size
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

#endif
