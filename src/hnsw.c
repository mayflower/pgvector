#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "access/amapi.h"
#include "access/genam.h"
#include "access/reloptions.h"
#include "access/sysattr.h"
#if PG_VERSION_NUM >= 150000
#include "access/rmgr.h"
#include "access/xloginsert.h"
#include "access/xlog_internal.h"
#include "access/xlogreader.h"
#endif
#include "commands/progress.h"
#include "commands/vacuum.h"
#include "fmgr.h"
#include "hnsw.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/bitmapset.h"
#include "nodes/pg_list.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "storage/lwlock.h"
#include "utils/float.h"
#include "utils/fmgrprotos.h"
#include "utils/guc.h"
#include "utils/rel.h"
#include "utils/relcache.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"
#include "vector.h"
#include "tqgraph.h"
#include "tqhybrid_bm25.h"

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

static const struct config_enum_entry hnsw_iterative_scan_options[] = {
	{"off", HNSW_ITERATIVE_SCAN_OFF, false},
	{"relaxed_order", HNSW_ITERATIVE_SCAN_RELAXED, false},
	{"strict_order", HNSW_ITERATIVE_SCAN_STRICT, false},
	{NULL, 0, false}
};

static relopt_enum_elt_def tq_routing_relopt_options[] = {
	{"auto", TQ_ROUTING_AUTO},
	{"graph", TQ_ROUTING_GRAPH},
	{"flat", TQ_ROUTING_FLAT},
	{NULL, 0}
};

static relopt_enum_elt_def tq_graph_rescore_band_relopt_options[] = {
	{"auto", TQ_GRAPH_RESCORE_BAND_AUTO},
	{"none", TQ_GRAPH_RESCORE_BAND_NONE},
	{"exact", TQ_GRAPH_RESCORE_BAND_EXACT},
	{NULL, 0}
};

static relopt_enum_elt_def tq_graph_exact_cache_relopt_options[] = {
	{"auto", TQ_GRAPH_EXACT_CACHE_AUTO},
	{"off", TQ_GRAPH_EXACT_CACHE_OFF},
	{"on", TQ_GRAPH_EXACT_CACHE_ON},
	{NULL, 0}
};

static relopt_enum_elt_def tq_graph_reorder_relopt_options[] = {
	{"auto", TQ_GRAPH_REORDER_AUTO},
	{"off", TQ_GRAPH_REORDER_OFF},
	{"bfs", TQ_GRAPH_REORDER_BFS},
	{NULL, 0}
};

static const struct config_enum_entry tq_graph_lookahead_prefetch_options[] = {
	{"auto", TQ_GRAPH_LOOKAHEAD_AUTO, false},
	{"off", TQ_GRAPH_LOOKAHEAD_OFF, false},
	{"on", TQ_GRAPH_LOOKAHEAD_ON, false},
	{NULL, 0, false}
};

static const struct config_enum_entry tq_dense_budget_policy_options[] = {
	{"quality", TQ_DENSE_BUDGET_QUALITY, false},
	{"balanced", TQ_DENSE_BUDGET_BALANCED, false},
	{"latency", TQ_DENSE_BUDGET_LATENCY, false},
	{"auto", TQ_DENSE_BUDGET_AUTO, false},
	{NULL, 0, false}
};

static const struct config_enum_entry tq_rescore_band_policy_options[] = {
	{"exact", TQ_RESCORE_BAND_POLICY_EXACT, false},
	{"limited", TQ_RESCORE_BAND_POLICY_LIMITED, false},
	{"auto", TQ_RESCORE_BAND_POLICY_AUTO, false},
	{"off", TQ_RESCORE_BAND_POLICY_OFF, false},
	{NULL, 0, false}
};

static const struct config_enum_entry tq_simd_force_options[] = {
	{"auto", TQ_SIMD_FORCE_AUTO, false},
	{"scalar", TQ_SIMD_FORCE_SCALAR, false},
	{"avx2", TQ_SIMD_FORCE_AVX2, false},
	{"avxvnni", TQ_SIMD_FORCE_AVXVNNI, false},
	{"avx512vnni", TQ_SIMD_FORCE_AVX512VNNI, false},
	{"neon", TQ_SIMD_FORCE_NEON, false},
	{"arm_sdot", TQ_SIMD_FORCE_ARM_SDOT, false},
	{"arm_i8mm", TQ_SIMD_FORCE_ARM_I8MM, false},
	{NULL, 0, false}
};

static const struct config_enum_entry tq_exact_simd_force_options[] = {
	{"auto", TQ_EXACT_SIMD_FORCE_AUTO, false},
	{"scalar", TQ_EXACT_SIMD_FORCE_SCALAR, false},
	{"neon", TQ_EXACT_SIMD_FORCE_NEON, false},
	{"avx512f", TQ_EXACT_SIMD_FORCE_AVX512F, false},
	{NULL, 0, false}
};

static const struct config_enum_entry tq_graph_batch_scoring_options[] = {
	{"auto", TQ_GRAPH_BATCH_AUTO, false},
	{"off", TQ_GRAPH_BATCH_OFF, false},
	{"on", TQ_GRAPH_BATCH_ON, false},
	{NULL, 0, false}
};

static const struct config_enum_entry tq_graph_avx512_weighted_options[] = {
	{"off", TQ_GRAPH_AVX512_WEIGHTED_OFF, false},
	{"on", TQ_GRAPH_AVX512_WEIGHTED_ON, false},
	{"auto", TQ_GRAPH_AVX512_WEIGHTED_AUTO, false},
	{NULL, 0, false}
};

int			hnsw_ef_search;
int			hnsw_iterative_scan;
int			hnsw_max_scan_tuples;
double		hnsw_scan_mem_multiplier;
bool		hnsw_tq_graph_prefetch;
bool		hnsw_tq_graph_stack_scratch;
bool		hnsw_tq_graph_lowbit_popcnt;
bool		hnsw_tq_graph_i8mm;
bool		hnsw_tq_graph_avxvnni;
bool		hnsw_tq_graph_avx512vnni;
bool		hnsw_tq_graph_avx512vpopcntdq;
bool		hnsw_tq_weighted;
bool		hnsw_tq_renorm;
bool		hnsw_tq_query_1bit_asymmetric;
int			hnsw_tq_query_1bit_asymmetric_bits;
bool		hnsw_tq_build_exact_distances;
bool		hnsw_tq_hadamard_simd;
bool		hnsw_tq_exact_avx512;
int			hnsw_tq_simd_force;
int			hnsw_tq_exact_simd_force;
int			hnsw_tq_graph_batch_scoring;
int			hnsw_tq_graph_batch_size;
int			hnsw_tq_graph_avx512_weighted;
int			hnsw_tq_graph_lookahead_prefetch;
int			hnsw_tq_graph_lookahead_threshold_kb;
int			hnsw_tq_dense_budget_policy;
int			hnsw_tq_dense_max_candidate_multiplier;
double		hnsw_tq_dense_latency_multiplier;
int			hnsw_tq_dense_max_rescore_multiplier;
int			hnsw_tq_rescore_band_policy;
int			hnsw_lock_tranche_id;
static relopt_kind hnsw_relopt_kind;
static relopt_kind tq_relopt_kind;
#define TQ_GRAPH_OP_COUNT				(HNSW_GRAPH_OP_VACUUM_REPAIR + 1)
#define TQ_GRAPH_CUSTOM_RMGR_NAME		"TurboQuantGraph"

#if PG_VERSION_NUM >= 150000
#define TQ_GRAPH_RMGR_ID				RM_EXPERIMENTAL_ID
#define XLOG_TQ_GRAPH_OP				0x00

typedef struct HnswGraphWalRecord
{
	RelFileLocator locator;
	ForkNumber	forknum;
	BlockNumber blkno;
	uint16		graphOpKind;
	uint16		storageKind;
} HnswGraphWalRecord;

static bool tq_graph_custom_wal_registered = false;
static bool tq_graph_custom_wal_collision = false;
#endif

static bool HnswPathHasFilter(IndexPath *path);
static bytea *hnswoptions(Datum reloptions, bool validate);
static bytea *turboquantoptions(Datum reloptions, bool validate);
static IndexAmRoutine *BuildHnswAmRoutine(amoptions_function amoptions, bool turboquant);
static const char *HnswGraphOpName(uint16 graphOpKind);
static void HnswRegisterGraphWalRmgr(void);

/*
 * Assign a tranche ID for our LWLocks. This only needs to be done by one
 * backend, as the tranche ID is remembered in shared memory.
 *
 * This shared memory area is very small, so we just allocate it from the
 * "slop" that PostgreSQL reserves for small allocations like this. If
 * this grows bigger, we should use a shmem_request_hook and
 * RequestAddinShmemSpace() to pre-reserve space for this.
 */
void
HnswInitLockTranche(void)
{
	int		   *tranche_ids;
	bool		found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	tranche_ids = ShmemInitStruct("hnsw LWLock ids",
								  sizeof(int) * 1,
								  &found);
	if (!found)
	{
#if PG_VERSION_NUM >= 190000
		tranche_ids[0] = LWLockNewTrancheId("HnswBuild");
#else
		tranche_ids[0] = LWLockNewTrancheId();
#endif
	}
	hnsw_lock_tranche_id = tranche_ids[0];
	LWLockRelease(AddinShmemInitLock);

#if PG_VERSION_NUM < 190000
	/* Per-backend registration of the tranche ID */
	LWLockRegisterTranche(hnsw_lock_tranche_id, "HnswBuild");
#endif
}

static const char *
HnswGraphOpName(uint16 graphOpKind)
{
	switch (graphOpKind)
	{
		case HNSW_GRAPH_OP_NONE:
			return "none";
		case HNSW_GRAPH_OP_PAGE_INIT:
			return "page_init";
		case HNSW_GRAPH_OP_PAGE_LINK:
			return "page_link";
		case HNSW_GRAPH_OP_META_UPDATE:
			return "meta_update";
		case HNSW_GRAPH_OP_ELEMENT_INSERT:
			return "element_insert";
		case HNSW_GRAPH_OP_NEIGHBOR_INSERT:
			return "neighbor_insert";
		case HNSW_GRAPH_OP_NEIGHBOR_UPDATE:
			return "neighbor_update";
		case HNSW_GRAPH_OP_DUPLICATE_HEAPTID:
			return "duplicate_heaptid";
		case HNSW_GRAPH_OP_VACUUM_DELETE:
			return "vacuum_delete";
		case HNSW_GRAPH_OP_VACUUM_REPAIR:
			return "vacuum_repair";
		default:
			return "unknown";
	}
}

#if PG_VERSION_NUM >= 150000
static void
HnswGraphWalRedo(XLogReaderState *record)
{
	(void) record;
}

static void
HnswGraphWalDesc(StringInfo buf, XLogReaderState *record)
{
	char	   *data = XLogRecGetData(record);
	uint32		len = XLogRecGetDataLen(record);

	if (len == sizeof(HnswGraphWalRecord))
	{
		HnswGraphWalRecord *xlrec = (HnswGraphWalRecord *) data;

		appendStringInfo(buf,
						 "op=%s locator=%u/%u/%u fork=%d blk=%u storage=%u",
						 HnswGraphOpName(xlrec->graphOpKind),
						 xlrec->locator.spcOid,
						 xlrec->locator.dbOid,
						 xlrec->locator.relNumber,
						 xlrec->forknum,
						 xlrec->blkno,
						 xlrec->storageKind);
	}
	else
		appendStringInfo(buf, "malformed graph WAL record length %u", len);
}

static const char *
HnswGraphWalIdentify(uint8 info)
{
	if ((info & ~XLR_INFO_MASK) == XLOG_TQ_GRAPH_OP)
		return "GRAPH_OP";

	return NULL;
}

static const RmgrData HnswGraphRmgrData = {
	.rm_name = TQ_GRAPH_CUSTOM_RMGR_NAME,
	.rm_redo = HnswGraphWalRedo,
	.rm_desc = HnswGraphWalDesc,
	.rm_identify = HnswGraphWalIdentify,
	.rm_startup = NULL,
	.rm_cleanup = NULL,
	.rm_mask = NULL,
	.rm_decode = NULL
};
#endif

static void
HnswRegisterGraphWalRmgr(void)
{
#if PG_VERSION_NUM >= 150000
	if (RmgrIdExists(TQ_GRAPH_RMGR_ID))
	{
		if (strcmp(RmgrTable[TQ_GRAPH_RMGR_ID].rm_name, TQ_GRAPH_CUSTOM_RMGR_NAME) == 0)
			tq_graph_custom_wal_registered = true;
		else
			tq_graph_custom_wal_collision = true;
		return;
	}

	/*
	 * Custom WAL records must be replayable during crash recovery. Only emit
	 * them when the resource manager was installed from shared_preload_libraries.
	 */
	if (!process_shared_preload_libraries_in_progress)
		return;

	RegisterCustomRmgr(TQ_GRAPH_RMGR_ID, &HnswGraphRmgrData);
	tq_graph_custom_wal_registered = true;
	tq_graph_custom_wal_collision = false;
#endif
}

bool
HnswGraphCustomWalEnabled(void)
{
#if PG_VERSION_NUM >= 150000
	return tq_graph_custom_wal_registered && !tq_graph_custom_wal_collision;
#else
	return false;
#endif
}

const char *
HnswGraphWalModeName(void)
{
	if (HnswGraphCustomWalEnabled())
		return "generic_xlog_page_ops_plus_custom_graph_records";

	return "generic_xlog_page_ops";
}

void
HnswLogGraphWalRecord(Relation index, ForkNumber forkNum, BlockNumber blkno, uint16 graphOpKind)
{
#if PG_VERSION_NUM >= 150000
	HnswGraphWalRecord xlrec;

	if (!HnswGraphCustomWalEnabled())
		return;

	if (!HnswUseTqGraph(index))
		return;

	if (!RelationNeedsWAL(index))
		return;

	xlrec.locator = index->rd_locator;
	xlrec.forknum = forkNum;
	xlrec.blkno = blkno;
	xlrec.graphOpKind = graphOpKind;
	xlrec.storageKind = HNSW_STORAGE_TURBOQUANT_GRAPH;

	XLogBeginInsert();
	XLogRegisterData((char *) &xlrec, sizeof(xlrec));
	XLogInsert(TQ_GRAPH_RMGR_ID, XLOG_TQ_GRAPH_OP);
#else
	(void) index;
	(void) forkNum;
	(void) blkno;
	(void) graphOpKind;
#endif
}

/*
 * Initialize index options and variables
 */
void
HnswInit(void)
{
	HnswRegisterGraphWalRmgr();

	if (!process_shared_preload_libraries_in_progress)
		HnswInitLockTranche();

	hnsw_relopt_kind = add_reloption_kind();
	add_int_reloption(hnsw_relopt_kind, "m", "Max number of connections",
					  HNSW_DEFAULT_M, HNSW_MIN_M, HNSW_MAX_M, AccessExclusiveLock);
	add_int_reloption(hnsw_relopt_kind, "ef_construction", "Size of the dynamic candidate list for construction",
					  HNSW_DEFAULT_EF_CONSTRUCTION, HNSW_MIN_EF_CONSTRUCTION, HNSW_MAX_EF_CONSTRUCTION, AccessExclusiveLock);

	tq_relopt_kind = add_reloption_kind();
	add_enum_reloption(tq_relopt_kind, "routing", "Turboquant routing mode",
					   tq_routing_relopt_options, TQ_ROUTING_AUTO,
					   "Valid values are \"auto\", \"graph\", and \"flat\".",
					   AccessExclusiveLock);
	add_int_reloption(tq_relopt_kind, "graph_m", "Max number of graph connections",
					  HNSW_DEFAULT_M, HNSW_MIN_M, HNSW_MAX_M, AccessExclusiveLock);
	add_int_reloption(tq_relopt_kind, "graph_ef_construction", "Size of the dynamic graph candidate list for construction",
					  TQ_DEFAULT_GRAPH_EF_CONSTRUCTION, HNSW_MIN_EF_CONSTRUCTION, HNSW_MAX_EF_CONSTRUCTION, AccessExclusiveLock);
	add_int_reloption(tq_relopt_kind, "graph_ef_search", "Size of the dynamic graph candidate list for search",
					  TQ_DEFAULT_GRAPH_EF_SEARCH, HNSW_MIN_EF_SEARCH, HNSW_MAX_EF_SEARCH, AccessExclusiveLock);
	add_int_reloption(tq_relopt_kind, "graph_oversampling", "Candidate oversampling multiplier for graph scans",
					  TQ_DEFAULT_GRAPH_OVERSAMPLING, 1, 1000, AccessExclusiveLock);
	add_enum_reloption(tq_relopt_kind, "graph_rescore_band", "Final candidate rescore band",
					   tq_graph_rescore_band_relopt_options, TQ_GRAPH_RESCORE_BAND_AUTO,
					   "Valid values are \"auto\", \"none\", and \"exact\".",
					   AccessExclusiveLock);
	add_enum_reloption(tq_relopt_kind, "graph_exact_cache", "Native graph exact vector cache policy",
					   tq_graph_exact_cache_relopt_options, TQ_GRAPH_EXACT_CACHE_AUTO,
					   "Valid values are \"auto\", \"off\", and \"on\".",
					   AccessExclusiveLock);
	add_enum_reloption(tq_relopt_kind, "graph_reorder", "Native graph node layout reorder policy",
					   tq_graph_reorder_relopt_options, TQ_GRAPH_REORDER_AUTO,
					   "Valid values are \"auto\", \"off\", and \"bfs\".",
					   AccessExclusiveLock);
	add_int_reloption(tq_relopt_kind, "tq_bits", "Turboquant code bit width",
					  TQ_DEFAULT_BITS, 1, TQ_DEFAULT_BITS, AccessExclusiveLock);
	add_bool_reloption(tq_relopt_kind, "tq_weighted",
					   "Opt into TurboQuant+ per-coordinate weighted scoring at build and scan time.",
					   false, AccessExclusiveLock);
	add_bool_reloption(tq_relopt_kind, "tq_quantile_fit",
					   "Replace the Welford mean/stddev fit of ecShift/ecScale with a quantile-anchored fit.  More robust to heavy-tailed or skewed per-coord distributions; default off because the Welford fit is the better choice on near-Gaussian data.",
					   false, AccessExclusiveLock);
	add_bool_reloption(tq_relopt_kind, "tq_renorm",
					   "Opt into TurboQuant+ renormalization residual correction at encode time.  Replaces the per-vector pre-quantization L2 length with l2_length / centroid_norm so Dot/Cosine scoring corrects for centroid quantization noise.  Default off; only takes effect when ecShift/ecScale corrections are present (cosine/IP builds) and tq_weighted is on.",
					   false, AccessExclusiveLock);
	add_bool_reloption(tq_relopt_kind, "tq_exact_storage",
					   "Store exact vectors in native TurboQuant graph indexes for final exact rescoring. Set off for compact exact-free quantized-only storage.",
					   true, AccessExclusiveLock);

	DefineCustomIntVariable("hnsw.ef_search", "Sets the size of the dynamic candidate list for search",
							"Valid range is 1..1000.", &hnsw_ef_search,
							HNSW_DEFAULT_EF_SEARCH, HNSW_MIN_EF_SEARCH, HNSW_MAX_EF_SEARCH, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.iterative_scan", "Sets the mode for iterative scans",
							 NULL, &hnsw_iterative_scan,
							 HNSW_ITERATIVE_SCAN_OFF, hnsw_iterative_scan_options, PGC_USERSET, 0, NULL, NULL, NULL);

	/* This is approximate and does not affect the initial scan */
	DefineCustomIntVariable("hnsw.max_scan_tuples", "Sets the max number of tuples to visit for iterative scans",
							NULL, &hnsw_max_scan_tuples,
							20000, 1, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);

	/* Same range as hash_mem_multiplier */
	DefineCustomRealVariable("hnsw.scan_mem_multiplier", "Sets the multiple of work_mem to use for iterative scans",
							 NULL, &hnsw_scan_mem_multiplier,
							 1, 1, 1000, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_graph_prefetch", "Prefetch native turboquant graph code payloads during scoring",
							 NULL, &hnsw_tq_graph_prefetch,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_graph_stack_scratch", "Use stack scratch buffers for small native turboquant graph heaps",
							 NULL, &hnsw_tq_graph_stack_scratch,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_graph_lowbit_popcnt", "Use experimental sign-only popcount routing for 1-bit native turboquant graph scans",
							 "Final candidate rescoring still uses exact vector bytes when enabled by the graph rescore band.",
							 &hnsw_tq_graph_lowbit_popcnt,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_graph_i8mm", "Use experimental ARM I8MM native turboquant graph scoring when available",
							 "This path is opt-in because FIQA/OpenAI validation on Apple ARM showed slower traversal than NEON dot-product.",
							 &hnsw_tq_graph_i8mm,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_graph_avxvnni", "Allow native turboquant graph scoring on AVX-VNNI when the CPU supports it",
							 "Set to off to force fallback to AVX2 — useful for downclock measurements and dispatch tests.",
							 &hnsw_tq_graph_avxvnni,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_graph_avx512vnni", "Allow native turboquant graph scoring on AVX-512 VNNI when the CPU supports it",
							 "Set to off to disable the ZMM kernel — useful for downclock measurements (Skylake-X / Ice Lake server) and to compare AVX-VNNI vs AVX-512 VNNI on hardware that has both.",
							 &hnsw_tq_graph_avx512vnni,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_graph_avx512vpopcntdq", "Allow the AVX-512 VPOPCNTDQ kernel for asymmetric 1-bit scoring",
							 "Set to off to disable the ZMM popcount kernel even when the CPU supports it.  Forces the dispatcher to fall through to the AVX2 nibble-lookup popcount kernel instead.  Useful for parity testing (assert AVX-512 vs AVX2 vs scalar give bit-identical scores) and for downclock measurement on hosts where AVX-512 VPOPCNTDQ might trigger frequency scaling (Ice Lake / Sapphire Rapids).  Default on; on hosts without AVX-512 VPOPCNTDQ this GUC has no effect because the runtime feature check skips the kernel anyway.",
							 &hnsw_tq_graph_avx512vpopcntdq,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_weighted", "Enable TurboQuant+ per-coord weighted scoring at scan time",
							 "Only indexes built WITH (tq_weighted = on) carry the per-vector correction field needed for the TQ+ formula. "
							 "Setting this GUC to off acts as a kill-switch on top of an opt-in index.",
							 &hnsw_tq_weighted,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_renorm", "Enable TurboQuant+ renormalization residual correction at encode time",
							 "Only indexes built WITH (tq_renorm = on) store the renormalized per-vector scaling_factor (= l2_length / centroid_norm) instead of the plain pre-quantization L2 length. "
							 "Setting this GUC to off acts as a kill-switch on top of an opt-in index.",
							 &hnsw_tq_renorm,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_query_1bit_asymmetric", "Use 8-bit asymmetric query encoding for 1-bit TurboQuant scans",
							 "Replaces the symmetric sign-only query encoding at tq_bits = 1 with qdrant-style bit-plane-decomposed signed query quantization.  Recovers query magnitude information that symmetric scoring discards.  Scan-time only; no on-disk format change.  Default off to preserve existing 1-bit scan behaviour.  Width of the query quantization is controlled by hnsw.tq_query_1bit_asymmetric_bits.",
							 &hnsw_tq_query_1bit_asymmetric,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.tq_query_1bit_asymmetric_bits", "Bit width for the asymmetric 1-bit query quantization",
							"Width of the signed-integer query quantization that feeds the bit-plane scorer when hnsw.tq_query_1bit_asymmetric is on.  Supported values: 8 (default, qdrant Query1bitSimd<8>), 12 (qdrant Query1bitSimd<12>, ~10x less quantization error than 8 but ~50%% more kernel work per query), 16 (asymptotic accuracy ceiling, ~100%% more kernel work).  Values outside {8, 12, 16} are clamped to the nearest supported width at scan time.  No effect when asymmetric scoring is off or tq_bits != 1.",
							&hnsw_tq_query_1bit_asymmetric_bits,
							8, 8, 16,
							PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_build_exact_distances", "Use exact f32 distances during TurboQuant graph construction",
							 "Build-time only.  Short-circuits the quantized code-code distance fast paths in TqGraphBuildPackedDistance, routing every neighbour-pruning call through the f32 dot-product kernel using the original Vector* still in memory during build.  The graph topology gets locked in with perfect distances; scan-time scoring still uses the packed codes.  Build is 3-5x slower (no SIMD packed-byte advantage) but graph quality is the maximum achievable at any bit width.  Default off — opt in via SET before CREATE INDEX.  No on-disk format change.",
							 &hnsw_tq_build_exact_distances,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_hadamard_simd", "Use SIMD (AVX2 / NEON) for the Hadamard butterfly inner loops",
							 "Default on — bit-exact with the scalar fallback for h ≥ 4 levels.  Set off to force scalar Hadamard for downclock/parity testing.",
							 &hnsw_tq_hadamard_simd,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomBoolVariable("hnsw.tq_exact_avx512", "Use the explicit AVX-512F kernel for exact-rescore L2 / inner product",
							 "Default off — the autoVec'd FMA path is already near-peak on most modern x86 hosts and avoids the AVX-512 downclock penalty on Skylake-X / Cascade Lake under sustained use.  Turn on after profiling on a target host to confirm it improves p95/p99 without raising p50.",
							 &hnsw_tq_exact_avx512,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.tq_simd_force", "Force or disable native TurboQuant dense SIMD dispatch",
							 "auto uses runtime CPU detection; scalar disables dense SIMD; specific values are honored only when the build and CPU support them, otherwise the dispatcher safely falls back.",
							 &hnsw_tq_simd_force,
							 TQ_SIMD_FORCE_AUTO,
							 tq_simd_force_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.tq_exact_simd_force", "Force or disable exact vector SIMD dispatch",
							 "auto uses the normal exact-distance dispatch; scalar disables explicit exact SIMD; architecture-specific values are honored only when safe.",
							 &hnsw_tq_exact_simd_force,
							 TQ_EXACT_SIMD_FORCE_AUTO,
							 tq_exact_simd_force_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.tq_graph_batch_scoring", "Control native TurboQuant dense graph batch scoring",
							 "auto and on score supported neighbor groups in batches; off routes every candidate through the single-node scorer for parity and benchmarking.",
							 &hnsw_tq_graph_batch_scoring,
							 TQ_GRAPH_BATCH_AUTO,
							 tq_graph_batch_scoring_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.tq_graph_batch_size", "Preferred native TurboQuant dense graph batch size",
							"The current batch kernels process groups of four. Values above four are accepted for benchmark matrix compatibility and use the four-wide kernel until wider kernels are available.",
							&hnsw_tq_graph_batch_size,
							4, 1, 8,
							PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.tq_graph_avx512_weighted", "Control AVX-512 weighted TurboQuant+ code-code scoring",
							 "off keeps weighted TQ+ on scalar/AVX2/NEON paths; on and auto allow the AVX-512BW/DQ weighted kernel when the CPU supports it. Default off because wide AVX-512 can downclock on some hosts.",
							 &hnsw_tq_graph_avx512_weighted,
							 TQ_GRAPH_AVX512_WEIGHTED_OFF,
							 tq_graph_avx512_weighted_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.tq_graph_lookahead_prefetch", "Adjacency-list look-ahead prefetch in the inner neighbor scan loop",
							 "auto = enable when the per-scan metadata working set (storage->nodes + visitedGeneration) exceeds hnsw.tq_graph_lookahead_threshold_kb.  off = always disabled.  on = always enabled.  Default auto: keeps small-corpus scans (where the hardware prefetcher already covers the access pattern) regression-free while turning on for >L3-sized corpora where the explicit hint wins.",
							 &hnsw_tq_graph_lookahead_prefetch,
							 TQ_GRAPH_LOOKAHEAD_AUTO,
							 tq_graph_lookahead_prefetch_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.tq_graph_lookahead_threshold_kb", "Working-set size in KiB above which auto mode of hnsw.tq_graph_lookahead_prefetch enables look-ahead prefetch",
							"Default 24576 (24 MiB) ≈ typical desktop / consumer L3.  Raise on hosts with bigger L3 (server CPUs often have 32-128 MiB) so auto-mode stays off until the working set really spills out of cache.  Only consulted when hnsw.tq_graph_lookahead_prefetch = auto.",
							&hnsw_tq_graph_lookahead_threshold_kb,
							24576, 0, INT_MAX,
							PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.tq_dense_budget_policy", "Control native TurboQuant dense candidate widening",
							 "quality preserves the historical candidate widening; latency caps high-dimensional over-collection; balanced is a middle ground; auto applies latency caps to high-dimensional unfiltered scans.",
							 &hnsw_tq_dense_budget_policy,
							 TQ_DENSE_BUDGET_AUTO,
							 tq_dense_budget_policy_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.tq_dense_max_candidate_multiplier", "Maximum adaptive dense candidate multiplier",
							"Upper bound on native graph candidates relative to requested dense_k or SQL LIMIT when latency/auto budgeting is active.",
							&hnsw_tq_dense_max_candidate_multiplier,
							4, 1, 1000,
							PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomRealVariable("hnsw.tq_dense_latency_multiplier", "Latency-mode dense candidate multiplier",
							 "Candidate multiplier used by hnsw.tq_dense_budget_policy = latency and by high-dimensional auto mode.",
							 &hnsw_tq_dense_latency_multiplier,
							 1.5, 1.0, 1000.0,
							 PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomIntVariable("hnsw.tq_dense_max_rescore_multiplier", "Maximum adaptive exact-rescore multiplier",
							"Caps exact rescore band to requested dense_k times this multiplier when hnsw.tq_rescore_band_policy limits rescoring.",
							&hnsw_tq_dense_max_rescore_multiplier,
							2, 1, 1000,
							PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.tq_rescore_band_policy", "Control native TurboQuant exact-rescore band",
							 "exact preserves historical full-band rescoring; limited caps by LIMIT and dense_k; auto limits high-dimensional unfiltered latency scans; off disables exact rescoring.",
							 &hnsw_tq_rescore_band_policy,
							 TQ_RESCORE_BAND_POLICY_AUTO,
							 tq_rescore_band_policy_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);

	TqGraphControlInit();

	MarkGUCPrefixReserved("hnsw");
}

/*
 * Get the name of index build phase
 */
static char *
hnswbuildphasename(int64 phasenum)
{
	switch (phasenum)
	{
		case PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE:
			return "initializing";
		case PROGRESS_HNSW_PHASE_LOAD:
			return "loading tuples";
		default:
			return NULL;
	}
}

/*
 * Estimate the cost of an index scan
 */
static void
hnswcostestimate_internal(PlannerInfo *root, IndexPath *path, double loop_count,
						  int efSearch, bool penalizeFilteredPath,
						  Cost *indexStartupCost, Cost *indexTotalCost,
						  Selectivity *indexSelectivity, double *indexCorrelation,
						  double *indexPages)
{
	GenericCosts costs;
	int			m;
	double		ratio;
	double		startupPages;
	double		spc_seq_page_cost;
	Relation	index;

	/* Never use index without order */
	if (path->indexorderbys == NIL)
	{
		*indexStartupCost = get_float8_infinity();
		*indexTotalCost = get_float8_infinity();
		*indexSelectivity = 0;
		*indexCorrelation = 0;
		*indexPages = 0;
#if PG_VERSION_NUM >= 180000
		/* See "On disable_cost" thread on pgsql-hackers */
		path->path.disabled_nodes = 2;
#endif
		return;
	}

	MemSet(&costs, 0, sizeof(costs));

	genericcostestimate(root, path, loop_count, &costs);

	index = index_open(path->indexinfo->indexoid, NoLock);
	HnswGetMetaPageInfo(index, &m, NULL);
	index_close(index, NoLock);

	/*
	 * HNSW cost estimation follows a formula that accounts for the total
	 * number of tuples indexed combined with the parameters that most
	 * influence the duration of the index scan, namely: m - the number of
	 * tuples that are scanned in each step of the HNSW graph traversal
	 * ef_search - which influences the total number of steps taken at layer 0
	 *
	 * The source of the vector data can impact how many steps it takes to
	 * converge on the set of vectors to return to the executor. Currently, we
	 * use a hardcoded scaling factor (HNSWScanScalingFactor) to help
	 * influence that, but this could later become a configurable parameter
	 * based on the cost estimations.
	 *
	 * The tuple estimator formula is below:
	 *
	 * numIndexTuples = entryLevel * m + layer0TuplesMax * layer0Selectivity
	 *
	 * "entryLevel * m" represents the floor of tuples we need to scan to get
	 * to layer 0 (L0).
	 *
	 * "layer0TuplesMax" is the estimated total number of tuples we'd scan at
	 * L0 if we weren't discarding already visited tuples as part of the scan.
	 *
	 * "layer0Selectivity" estimates the percentage of tuples that are scanned
	 * at L0, accounting for previously visited tuples, multiplied by the
	 * "scalingFactor" (currently hardcoded).
	 */
	if (path->indexinfo->tuples > 0)
	{
		double		scalingFactor = 0.55;
		int			entryLevel = (int) (log(path->indexinfo->tuples) * HnswGetMl(m));
		int			layer0TuplesMax = HnswGetLayerM(m, 0) * efSearch;
		double		layer0Selectivity = scalingFactor * log(path->indexinfo->tuples) / (log(m) * (1 + log(efSearch)));

		ratio = (entryLevel * m + layer0TuplesMax * layer0Selectivity) / path->indexinfo->tuples;

		if (ratio > 1)
			ratio = 1;
	}
	else
		ratio = 1;

	get_tablespace_page_costs(path->indexinfo->reltablespace, NULL, &spc_seq_page_cost);

	/* Startup cost is cost before returning the first row */
	costs.indexStartupCost = costs.indexTotalCost * ratio;

	/* Adjust cost if needed since TOAST not included in seq scan cost */
	startupPages = costs.numIndexPages * ratio;
	if (startupPages > path->indexinfo->rel->pages && ratio < 0.5)
	{
		/* Change all page cost from random to sequential */
		costs.indexStartupCost -= startupPages * (costs.spc_random_page_cost - spc_seq_page_cost);

		/* Remove cost of extra pages */
		costs.indexStartupCost -= (startupPages - path->indexinfo->rel->pages) * spc_seq_page_cost;
	}

	*indexStartupCost = costs.indexStartupCost;
	*indexTotalCost = costs.indexTotalCost;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
	*indexPages = costs.numIndexPages;
}

static void
hnswcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
				 Cost *indexStartupCost, Cost *indexTotalCost,
				 Selectivity *indexSelectivity, double *indexCorrelation,
				 double *indexPages)
{
	hnswcostestimate_internal(root, path, loop_count, hnsw_ef_search, true,
							  indexStartupCost, indexTotalCost,
							  indexSelectivity, indexCorrelation, indexPages);
}

static void
turboquantcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
					   Cost *indexStartupCost, Cost *indexTotalCost,
					   Selectivity *indexSelectivity, double *indexCorrelation,
					   double *indexPages)
{
	Relation	index = index_open(path->indexinfo->indexoid, NoLock);
	bool		useNativeGraph = HnswUseTqNativeGraph(index);
	int			efSearch = HnswGetEfSearch(index);

	index_close(index, NoLock);

	hnswcostestimate_internal(root, path, loop_count, efSearch, !useNativeGraph,
							  indexStartupCost, indexTotalCost,
							  indexSelectivity, indexCorrelation, indexPages);

	if (useNativeGraph)
	{
		double		costMultiplier = HnswPathHasFilter(path) ? 0.02 : 0.2;

		/*
		 * Native TurboQuant graph scans traverse compact cached codes and
		 * adjacency lists, then exact-rescore only the final candidate band.
		 * The physical index also stores exact vector bytes for rescore, so
		 * generic page costs overstate the startup work for ordered scans.
		 * With scalar filters, the scan widens its final candidate band using
		 * planner selectivity instead of falling back to a bitmap filter plus
		 * full sort, so discount those ordered top-k paths further.
		 */
		*indexStartupCost *= costMultiplier;
		*indexTotalCost *= costMultiplier;
	}
}

static bool
HnswPathHasFilter(IndexPath *path)
{
	int			indexAttno = path->indexinfo->indexkeys[0];
	ListCell   *lc;

	foreach(lc, path->indexinfo->indrestrictinfo)
	{
		RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);
		Bitmapset  *attrs = NULL;

		pull_varattnos((Node *) rinfo->clause, path->indexinfo->rel->relid, &attrs);

		if (attrs == NULL)
			continue;

		if (bms_membership(attrs) != BMS_SINGLETON ||
			!bms_is_member(indexAttno - FirstLowInvalidHeapAttributeNumber, attrs))
			return true;
	}

	return false;
}

/*
 * Parse and validate the reloptions
 */
static bytea *
hnswoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"m", RELOPT_TYPE_INT, offsetof(HnswOptions, m)},
		{"ef_construction", RELOPT_TYPE_INT, offsetof(HnswOptions, efConstruction)},
	};

	return (bytea *) build_reloptions(reloptions, validate,
									  hnsw_relopt_kind,
									  sizeof(HnswOptions),
									  tab, lengthof(tab));
}

/*
 * Parse and validate turboquant reloptions. The first fields intentionally
 * map graph_m and graph_ef_construction onto HnswOptions-compatible storage
 * so the native HNSW graph code can build and maintain the index.
 */
static bytea *
turboquantoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"graph_m", RELOPT_TYPE_INT, offsetof(TqOptions, m)},
		{"graph_ef_construction", RELOPT_TYPE_INT, offsetof(TqOptions, efConstruction)},
		{"routing", RELOPT_TYPE_ENUM, offsetof(TqOptions, routing)},
		{"graph_ef_search", RELOPT_TYPE_INT, offsetof(TqOptions, graphEfSearch)},
		{"graph_oversampling", RELOPT_TYPE_INT, offsetof(TqOptions, graphOversampling)},
		{"graph_rescore_band", RELOPT_TYPE_ENUM, offsetof(TqOptions, graphRescoreBand)},
		{"graph_exact_cache", RELOPT_TYPE_ENUM, offsetof(TqOptions, graphExactCache)},
		{"graph_reorder", RELOPT_TYPE_ENUM, offsetof(TqOptions, graphReorder)},
		{"tq_bits", RELOPT_TYPE_INT, offsetof(TqOptions, tqBits)},
		{"tq_weighted", RELOPT_TYPE_BOOL, offsetof(TqOptions, tqWeighted)},
		{"tq_quantile_fit", RELOPT_TYPE_BOOL, offsetof(TqOptions, tqQuantileFit)},
		{"tq_renorm", RELOPT_TYPE_BOOL, offsetof(TqOptions, tqRenorm)},
		{"tq_exact_storage", RELOPT_TYPE_BOOL, offsetof(TqOptions, tqExactStorage)},
	};

	TqOptions  *opts = (TqOptions *) build_reloptions(reloptions, validate,
													  tq_relopt_kind,
													  sizeof(TqOptions),
													  tab, lengthof(tab));

	if (validate && opts != NULL &&
		opts->tqBits != 1 && opts->tqBits != 2 && opts->tqBits != TQ_DEFAULT_BITS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid value %d for option \"tq_bits\"", opts->tqBits),
				 errdetail("Valid values are \"1\", \"2\", and \"4\".")));

	if (validate && opts != NULL &&
		(opts->m < HNSW_MIN_M || opts->m > HNSW_MAX_M))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("value %d out of bounds for option \"graph_m\"", opts->m),
				 errdetail("Valid values are between \"%d\" and \"%d\".",
						   HNSW_MIN_M, HNSW_MAX_M)));

	return (bytea *) opts;
}

static IndexBulkDeleteResult *
turboquantbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
					 IndexBulkDeleteCallback callback, void *callback_state)
{
	if (HnswUseTqNativeGraph(info->index))
		return tqgraphbulkdelete(info, stats, callback, callback_state);
	if (!HnswUseTqFlat(info->index))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant requires a native graph-compatible opclass"),
				 errhint("Use the hnsw or ivfflat access method directly for this opclass.")));

	return hnswbulkdelete(info, stats, callback, callback_state);
}

static IndexBulkDeleteResult *
turboquantvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	if (HnswUseTqNativeGraph(info->index))
		return tqgraphvacuumcleanup(info, stats);
	if (!HnswUseTqFlat(info->index))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant requires a native graph-compatible opclass"),
				 errhint("Use the hnsw or ivfflat access method directly for this opclass.")));

	return hnswvacuumcleanup(info, stats);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tq_index_stats);
Datum
tq_index_stats(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Relation	index;
	Buffer		buf;
	Page		page;
	HnswMetaPage metap;
	HnswPageOpaque opaque;
	BlockNumber nblocks;
	uint16		storageKind;
	uint16		graphM;
	uint16		graphEfConstruction;
	uint16		graphEfSearch;
	uint16		graphOversampling;
	uint16		graphRescoreBand;
	uint16		graphMaxLevel;
	uint16		graphFlags;
	uint16		tqFlags;
	uint16		tqBits;
	BlockNumber tqCorrectionStartBlkno;
	BlockNumber tqBm25MetaStartBlkno;
	HnswMetaPageData metaSnapshot;
	int16		entryLevel;
	uint16		metaPageKind;
	uint16		metaLastGraphOp;
	uint16		firstGraphPageKind = 0;
	uint16		firstGraphLastGraphOp = 0;
	int64		graphPageCount = 0;
	int64		graphTaggedPageCount = 0;
	int64		graphUnknownLastOpCount = 0;
	int64		graphLastOpCounts[TQ_GRAPH_OP_COUNT] = {0};
	int64		tqLiveNodeCount = 0;
	int64		tqDeadNodeCount = 0;
	int64		graphAdjacencyRefCount = 0;
	int64		graphDeadNeighborRefCount = 0;
	bool		hasBm25Meta = false;
	TqHybridBm25MetaTupleData bm25Meta;
	StringInfoData json;

	index = index_open(indexOid, AccessShareLock);

	nblocks = RelationGetNumberOfBlocks(index);
	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);
	opaque = HnswPageGetOpaque(page);

	if (unlikely(metap->magicNumber != HNSW_MAGIC_NUMBER))
		elog(ERROR, "hnsw index is not valid");

	metaSnapshot = *metap;
	storageKind = metap->storageKind;
	graphM = metap->m;
	graphEfConstruction = metap->efConstruction;
	graphEfSearch = metap->graphEfSearch;
	graphOversampling = metap->graphOversampling;
	graphRescoreBand = metap->graphRescoreBand;
	graphMaxLevel = metap->graphMaxLevel;
	graphFlags = metap->graphFlags;
	tqFlags = metap->tqFlags;
	tqBits = metap->tqBits != 0 ? metap->tqBits : TQ_DEFAULT_BITS;
	tqCorrectionStartBlkno = metap->tqCorrectionStartBlkno;
	tqBm25MetaStartBlkno = metap->tqBm25MetaStartBlkno > HNSW_METAPAGE_BLKNO ?
		metap->tqBm25MetaStartBlkno : InvalidBlockNumber;
	entryLevel = metap->entryLevel;
	metaPageKind = opaque->pageKind & HNSW_PAGE_KIND_MASK;
	metaLastGraphOp = opaque->pageKind >> HNSW_PAGE_GRAPH_OP_SHIFT;

	UnlockReleaseBuffer(buf);

	if (nblocks > HNSW_METAPAGE_BLKNO + 1)
	{
		buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO + 1);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);
		if (opaque->page_id == HNSW_PAGE_ID)
		{
			firstGraphPageKind = opaque->pageKind & HNSW_PAGE_KIND_MASK;
			firstGraphLastGraphOp = opaque->pageKind >> HNSW_PAGE_GRAPH_OP_SHIFT;
		}
		UnlockReleaseBuffer(buf);
	}

	if (BlockNumberIsValid(tqBm25MetaStartBlkno))
	{
		bool		foundBm25Tuple = false;

		if (tqBm25MetaStartBlkno >= nblocks)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("turbohybrid BM25 metadata pointer is invalid"),
					 errdetail("Metapage points to block %u, but the index has only %u blocks.",
							   tqBm25MetaStartBlkno, nblocks)));

		buf = ReadBuffer(index, tqBm25MetaStartBlkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);

		if (opaque->page_id != HNSW_PAGE_ID ||
			(opaque->pageKind & HNSW_PAGE_KIND_MASK) != HNSW_PAGE_KIND_TQ_BM25_META)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("turbohybrid BM25 metadata pointer is invalid"),
					 errdetail("Metapage points to block %u, which is not a BM25 metadata page.",
							   tqBm25MetaStartBlkno)));
		}

		for (OffsetNumber off = FirstOffsetNumber;
			 off <= PageGetMaxOffsetNumber(page);
			 off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			TqHybridBm25MetaTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (TqHybridBm25MetaTuple) PageGetItem(page, iid);
			if (tuple->type == TQHYBRID_BM25_META_TUPLE_TYPE)
			{
				bm25Meta = *tuple;
				hasBm25Meta = true;
				foundBm25Tuple = true;
				break;
			}
		}

		UnlockReleaseBuffer(buf);
		if (!foundBm25Tuple)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("turbohybrid BM25 metadata tuple is missing"),
					 errdetail("Metapage points to BM25 metadata block %u, but no metadata tuple was found.",
							   tqBm25MetaStartBlkno)));
	}

	for (BlockNumber blkno = HNSW_METAPAGE_BLKNO; blkno < nblocks; blkno++)
	{
		uint16		lastGraphOp;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = HnswPageGetOpaque(page);

		if (opaque->page_id == HNSW_PAGE_ID)
		{
			graphPageCount++;
			lastGraphOp = opaque->pageKind >> HNSW_PAGE_GRAPH_OP_SHIFT;
			if (lastGraphOp != HNSW_GRAPH_OP_NONE)
				graphTaggedPageCount++;
			if (lastGraphOp < TQ_GRAPH_OP_COUNT)
				graphLastOpCounts[lastGraphOp]++;
			else
				graphUnknownLastOpCount++;
		}

		UnlockReleaseBuffer(buf);
	}

	if (storageKind == HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE)
		TqGraphCollectVacuumStats(index, &metaSnapshot,
								  &tqLiveNodeCount,
								  &tqDeadNodeCount,
								  &graphAdjacencyRefCount,
								  &graphDeadNeighborRefCount);

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"storage_kind\":\"%s\","
					 "\"graph_m\":%u,"
					 "\"graph_ef_construction\":%u,"
					 "\"graph_ef_search\":%u,"
					 "\"graph_oversampling\":%u,"
					 "\"graph_rescore_band\":%u,"
					 "\"graph_max_level\":%u,"
					 "\"graph_flags\":%u,"
					 "\"tq_flags\":%u,"
					 "\"tq_bits\":%u,"
					 "\"tq_plus\":%s,"
					 "\"tq_exact_storage\":%s,"
					 "\"tq_correction_start_block\":%u,"
					 "\"entry_level\":%d,"
					 "\"meta_page_kind\":%u,"
					 "\"meta_last_graph_op\":%u,"
					 "\"first_graph_page_kind\":%u,"
					 "\"first_graph_last_graph_op\":%u,"
					 "\"graph_page_count\":" INT64_FORMAT ","
					 "\"graph_tagged_page_count\":" INT64_FORMAT ","
					 "\"tq_live_node_count\":" INT64_FORMAT ","
					 "\"tq_dead_node_count\":" INT64_FORMAT ","
					 "\"graph_adjacency_ref_count\":" INT64_FORMAT ","
					 "\"graph_dead_neighbor_refs\":" INT64_FORMAT ","
					 "\"hybrid\":%s,"
					 "\"bm25_meta_start_block\":%u,"
					 "\"bm25_doc_count\":%u,"
					 "\"bm25_live_doc_count\":" INT64_FORMAT ","
					 "\"bm25_dead_doc_count\":" INT64_FORMAT ","
					 "\"bm25_avgdl\":%.6g,"
					 "\"bm25_term_count\":%u,"
					 "\"bm25_postings_pages\":%u,"
					 "\"bm25_blockmax_pages\":%u,"
					 "\"bm25_delta_pages\":%u,"
					 "\"bm25_delta_generation\":" UINT64_FORMAT ","
					 "\"bm25_last_compaction\":\"%s\","
					 "\"graph_page_last_op_counts\":{"
					 "\"none\":" INT64_FORMAT ","
					 "\"page_init\":" INT64_FORMAT ","
					 "\"page_link\":" INT64_FORMAT ","
					 "\"meta_update\":" INT64_FORMAT ","
					 "\"element_insert\":" INT64_FORMAT ","
					 "\"neighbor_insert\":" INT64_FORMAT ","
					 "\"neighbor_update\":" INT64_FORMAT ","
					 "\"duplicate_heaptid\":" INT64_FORMAT ","
					 "\"vacuum_delete\":" INT64_FORMAT ","
					 "\"vacuum_repair\":" INT64_FORMAT ","
					 "\"unknown\":" INT64_FORMAT "},"
					 "\"graph_page_op_tag_mode\":\"page_opaque_high_bits\","
					 "\"graph_wal_mode\":\"%s\","
					 "\"graph_custom_wal_records\":%s}",
					 HnswStorageKindName(storageKind),
					 graphM,
					 graphEfConstruction,
					 graphEfSearch,
					 graphOversampling,
					 graphRescoreBand,
					 graphMaxLevel,
					 graphFlags,
					 tqFlags,
					 tqBits,
					 tqFlags != 0 && tqCorrectionStartBlkno != InvalidBlockNumber ? "true" : "false",
					 (tqFlags & TQ_GRAPH_EXACT_FREE) != 0 ? "false" : "true",
					 tqCorrectionStartBlkno,
					 entryLevel,
					 metaPageKind,
					 metaLastGraphOp,
					 firstGraphPageKind,
					 firstGraphLastGraphOp,
					 graphPageCount,
					 graphTaggedPageCount,
					 tqLiveNodeCount,
					 tqDeadNodeCount,
					 graphAdjacencyRefCount,
					 graphDeadNeighborRefCount,
					 hasBm25Meta ? "true" : "false",
					 tqBm25MetaStartBlkno,
					 hasBm25Meta ? bm25Meta.docCount + bm25Meta.deltaDocCount : 0,
					 hasBm25Meta ? (bm25Meta.lastCompactionGeneration > 0 &&
									bm25Meta.deltaDocCount == 0 ?
									(int64) bm25Meta.docCount :
									Max((int64) (bm25Meta.docCount + bm25Meta.deltaDocCount) -
										Min((int64) (bm25Meta.docCount + bm25Meta.deltaDocCount),
											tqDeadNodeCount), 0)) : 0,
					 hasBm25Meta ? (bm25Meta.lastCompactionGeneration > 0 &&
									bm25Meta.deltaDocCount == 0 ? 0 :
									Min((int64) (bm25Meta.docCount + bm25Meta.deltaDocCount),
										tqDeadNodeCount)) : 0,
					 hasBm25Meta ? (double) (bm25Meta.totalDocLen + bm25Meta.deltaTotalDocLen) /
					 Max((double) (bm25Meta.docCount + bm25Meta.deltaDocCount), 1.0) : 0.0,
					 hasBm25Meta ? bm25Meta.termCount : 0,
					 hasBm25Meta ? bm25Meta.postingsPages : 0,
					 hasBm25Meta ? bm25Meta.blockMaxPages : 0,
					 hasBm25Meta ? bm25Meta.deltaPages : 0,
					 hasBm25Meta ? bm25Meta.deltaGeneration : 0,
					 hasBm25Meta && bm25Meta.lastCompactionGeneration > 0 ?
					 psprintf("generation " UINT64_FORMAT,
							  bm25Meta.lastCompactionGeneration) : "never",
					 graphLastOpCounts[HNSW_GRAPH_OP_NONE],
					 graphLastOpCounts[HNSW_GRAPH_OP_PAGE_INIT],
					 graphLastOpCounts[HNSW_GRAPH_OP_PAGE_LINK],
					 graphLastOpCounts[HNSW_GRAPH_OP_META_UPDATE],
					 graphLastOpCounts[HNSW_GRAPH_OP_ELEMENT_INSERT],
					 graphLastOpCounts[HNSW_GRAPH_OP_NEIGHBOR_INSERT],
					 graphLastOpCounts[HNSW_GRAPH_OP_NEIGHBOR_UPDATE],
					 graphLastOpCounts[HNSW_GRAPH_OP_DUPLICATE_HEAPTID],
					 graphLastOpCounts[HNSW_GRAPH_OP_VACUUM_DELETE],
					 graphLastOpCounts[HNSW_GRAPH_OP_VACUUM_REPAIR],
					 graphUnknownLastOpCount,
					 HnswGraphWalModeName(),
					 HnswGraphCustomWalEnabled() ? "true" : "false");
	index_close(index, AccessShareLock);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}

/*
 * Validate catalog entries for the specified operator class
 */
static bool
hnswvalidate(Oid opclassoid)
{
	return true;
}

/*
 * Define index handler
 *
 * See https://www.postgresql.org/docs/current/index-api.html
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnswhandler);
Datum
hnswhandler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(BuildHnswAmRoutine(hnswoptions, false));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(turboquanthandler);
Datum
turboquanthandler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(BuildHnswAmRoutine(turboquantoptions, true));
}

static IndexAmRoutine *
BuildHnswAmRoutine(amoptions_function amoptions, bool turboquant)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = turboquant ? 5 : 3;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
#if PG_VERSION_NUM >= 180000
	amroutine->amcanhash = false;
	amroutine->amconsistentequality = false;
	amroutine->amconsistentordering = false;
#endif
	amroutine->amcanbackward = false;	/* can change direction mid-scan */
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = false;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
#if PG_VERSION_NUM >= 170000
	amroutine->amcanbuildparallel = true;
#endif
	amroutine->amcaninclude = turboquant;
	amroutine->amusemaintenanceworkmem = false; /* not used during VACUUM */
#if PG_VERSION_NUM >= 160000
	amroutine->amsummarizing = false;
#endif
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;

	/* Interface functions */
	amroutine->ambuild = turboquant ? turboquantbuild : hnswbuild;
	amroutine->ambuildempty = turboquant ? turboquantbuildempty : hnswbuildempty;
	amroutine->aminsert = turboquant ? turboquantinsert : hnswinsert;
#if PG_VERSION_NUM >= 170000
	amroutine->aminsertcleanup = NULL;
#endif
	amroutine->ambulkdelete = turboquant ? turboquantbulkdelete : hnswbulkdelete;
	amroutine->amvacuumcleanup = turboquant ? turboquantvacuumcleanup : hnswvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = turboquant ? turboquantcostestimate : hnswcostestimate;
#if PG_VERSION_NUM >= 180000
	amroutine->amgettreeheight = NULL;
#endif
	amroutine->amoptions = amoptions;
	amroutine->amproperty = NULL;	/* TODO AMPROP_DISTANCE_ORDERABLE */
	amroutine->ambuildphasename = hnswbuildphasename;
	amroutine->amvalidate = hnswvalidate;
#if PG_VERSION_NUM >= 140000
	amroutine->amadjustmembers = NULL;
#endif
	amroutine->ambeginscan = turboquant ? turboquantbeginscan : hnswbeginscan;
	amroutine->amrescan = turboquant ? turboquantrescan : hnswrescan;
	amroutine->amgettuple = turboquant ? turboquantgettuple : hnswgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = turboquant ? turboquantendscan : hnswendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;

	/* Interface functions to support parallel index scans */
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

#if PG_VERSION_NUM >= 180000
	amroutine->amtranslatestrategy = NULL;
	amroutine->amtranslatecmptype = NULL;
#endif

	return amroutine;
}
