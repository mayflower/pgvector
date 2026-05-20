#include "postgres.h"

#include "fmgr.h"
#include "hnsw.h"
#include "lib/stringinfo.h"
#include "tqgraph.h"
#include "tqhybrid.h"
#include "tqhybrid_bm25.h"
#include "utils/fmgrprotos.h"

#define TQ_SCAN_ORCHESTRATION_NONE		0
#define TQ_SCAN_ORCHESTRATION_GRAPH		1
#define TQ_SCAN_ORCHESTRATION_FLAT		2

static int64 tq_last_graph_visited_nodes = 0;
static int64 tq_last_graph_scored_codes = 0;
static int64 tq_last_graph_batch_scored_codes = 0;
static int64 tq_last_graph_scalar_scored_codes = 0;
static int tq_last_graph_batch_kernel = TQ_SCORING_SCALAR;
static int tq_last_weighted_code_code_kernel = TQ_SCORING_SCALAR;
static int64 tq_last_graph_candidate_count = 0;
static int64 tq_last_graph_rescore_count = 0;
static int64 tq_last_graph_rescore_pages = 0;
static int64 tq_last_graph_code_pages_read = 0;
static int64 tq_last_graph_adj_pages_read = 0;
static int64 tq_last_graph_entry_point_count = 0;
static int64 tq_last_graph_prepare_us = 0;
static int64 tq_last_graph_traverse_us = 0;
static int64 tq_last_graph_entry_us = 0;
static int64 tq_last_graph_base_us = 0;
static int64 tq_last_graph_batch_us = 0;
static int64 tq_last_graph_heap_us = 0;
static int64 tq_last_graph_fill_us = 0;
static int64 tq_last_graph_rescore_us = 0;
static int64 tq_last_graph_sort_us = 0;
static int64 tq_last_graph_total_us = 0;
static int64 tq_last_graph_dense_requested_k = 0;
static int64 tq_last_graph_effective_result_target = 0;
static int64 tq_last_graph_effective_search_ef = 0;
static int64 tq_last_graph_effective_rescore_band = 0;
static double tq_last_graph_highdim_widening_multiplier = 1.0;
static int tq_last_graph_widening_reason = TQ_DENSE_WIDENING_NONE;
static int tq_last_graph_dense_budget_policy = TQ_DENSE_BUDGET_AUTO;
static int tq_last_graph_rescore_band_policy = TQ_RESCORE_BAND_POLICY_AUTO;
static int tq_last_graph_scoring_kernel = TQ_SCORING_SCALAR;
static int tq_last_exact_vector_kernel = TQ_EXACT_KERNEL_SCALAR;
static bool tq_last_exact_vector_kernel_recorded = false;
static int tq_last_graph_storage_kind = HNSW_STORAGE_HNSW;
static int tq_last_scan_orchestration = TQ_SCAN_ORCHESTRATION_NONE;
static int tq_last_graph_tq_bits = 0;
static bool tq_last_graph_query_split_active = false;

static const char *TqScanOrchestrationName(void);
static const char *TqExactKernelName(int kernel);
static const char *TqGraphAvx512WeightedModeName(int mode);

void
HnswRecordExactVectorKernel(int kernel)
{
	tq_last_exact_vector_kernel = kernel;
	tq_last_exact_vector_kernel_recorded = true;
}

void
HnswRecordWeightedCodeCodeKernel(int kernel)
{
	tq_last_weighted_code_code_kernel = kernel;
}

static int
TqExpectedExactKernel(void)
{
	if (hnsw_tq_exact_simd_force == TQ_EXACT_SIMD_FORCE_SCALAR)
		return TQ_EXACT_KERNEL_SCALAR;
#if !defined(TQ_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
	if (hnsw_tq_exact_simd_force != TQ_EXACT_SIMD_FORCE_AVX512F)
		return TQ_EXACT_KERNEL_NEON;
#endif
#if !defined(TQ_DISABLE_SIMD) && defined(__x86_64__)
	if (hnsw_tq_exact_simd_force == TQ_EXACT_SIMD_FORCE_AVX512F)
		return TQ_EXACT_KERNEL_AVX512F;
	return TQ_EXACT_KERNEL_AVX2;
#endif
	return TQ_EXACT_KERNEL_AUTOVEC_FMA;
}

void
HnswRecordGraphScanStats(HnswScanOpaque so)
{
	if (so == NULL || !so->turboquantGraphScan)
		return;

	tq_last_scan_orchestration = TQ_SCAN_ORCHESTRATION_GRAPH;
	tq_last_graph_visited_nodes = so->graphVisitedNodes;
	tq_last_graph_scored_codes = so->graphScoredCodes;
	tq_last_graph_batch_scored_codes = so->graphBatchScoredCodes;
	tq_last_graph_scalar_scored_codes = so->graphScalarScoredCodes;
	tq_last_graph_batch_kernel = so->graphBatchKernel;
	tq_last_graph_candidate_count = so->graphCandidateCount;
	tq_last_graph_rescore_count = so->graphRescoreCount;
	tq_last_graph_rescore_pages = so->graphRescorePages;
	tq_last_graph_code_pages_read = so->graphCodePagesRead;
	tq_last_graph_adj_pages_read = so->graphAdjPagesRead;
	tq_last_graph_entry_point_count = so->graphEntryPointCount;
	tq_last_graph_prepare_us = so->graphPrepareUs;
	tq_last_graph_traverse_us = so->graphTraverseUs;
	tq_last_graph_entry_us = so->graphEntryUs;
	tq_last_graph_base_us = so->graphBaseUs;
	tq_last_graph_batch_us = so->graphBatchUs;
	tq_last_graph_heap_us = so->graphHeapUs;
	tq_last_graph_fill_us = so->graphFillUs;
	tq_last_graph_rescore_us = so->graphRescoreUs;
	tq_last_graph_sort_us = so->graphSortUs;
	tq_last_graph_total_us = so->graphTotalUs;
	tq_last_graph_dense_requested_k = so->graphDenseRequestedK;
	tq_last_graph_effective_result_target = so->graphEffectiveResultTarget;
	tq_last_graph_effective_search_ef = so->graphEffectiveSearchEf;
	tq_last_graph_effective_rescore_band = so->graphEffectiveRescoreBand;
	tq_last_graph_highdim_widening_multiplier = so->graphHighdimWideningMultiplier;
	tq_last_graph_widening_reason = so->graphWideningReason;
	tq_last_graph_dense_budget_policy = so->graphDenseBudgetPolicy;
	tq_last_graph_rescore_band_policy = so->graphRescoreBandPolicy;
	tq_last_graph_scoring_kernel = so->tq.enabled ? so->tq.scoringKernel : TQ_SCORING_SCALAR;
	if (!tq_last_exact_vector_kernel_recorded && so->graphRescoreCount > 0)
		tq_last_exact_vector_kernel = TqExpectedExactKernel();
	tq_last_graph_storage_kind = so->graphStorageKind;
	tq_last_graph_tq_bits = so->tq.enabled ? so->tq.bits : 0;
#if defined(__aarch64__) || defined(_M_ARM64) || \
	defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
	tq_last_graph_query_split_active = so->tq.enabled && so->tq.querySplitEnabled;
#else
	tq_last_graph_query_split_active = false;
#endif
	tq_last_exact_vector_kernel_recorded = false;
}

void
HnswRecordNonGraphScanStats(void)
{
	tq_last_scan_orchestration = TQ_SCAN_ORCHESTRATION_NONE;
	tq_last_graph_visited_nodes = 0;
	tq_last_graph_scored_codes = 0;
	tq_last_graph_batch_scored_codes = 0;
	tq_last_graph_scalar_scored_codes = 0;
	tq_last_graph_batch_kernel = TQ_SCORING_SCALAR;
	tq_last_weighted_code_code_kernel = TQ_SCORING_SCALAR;
	tq_last_graph_candidate_count = 0;
	tq_last_graph_rescore_count = 0;
	tq_last_graph_rescore_pages = 0;
	tq_last_graph_code_pages_read = 0;
	tq_last_graph_adj_pages_read = 0;
	tq_last_graph_entry_point_count = 0;
	tq_last_graph_prepare_us = 0;
	tq_last_graph_traverse_us = 0;
	tq_last_graph_entry_us = 0;
	tq_last_graph_base_us = 0;
	tq_last_graph_batch_us = 0;
	tq_last_graph_heap_us = 0;
	tq_last_graph_fill_us = 0;
	tq_last_graph_rescore_us = 0;
	tq_last_graph_sort_us = 0;
	tq_last_graph_total_us = 0;
	tq_last_graph_dense_requested_k = 0;
	tq_last_graph_effective_result_target = 0;
	tq_last_graph_effective_search_ef = 0;
	tq_last_graph_effective_rescore_band = 0;
	tq_last_graph_highdim_widening_multiplier = 1.0;
	tq_last_graph_widening_reason = TQ_DENSE_WIDENING_NONE;
	tq_last_graph_dense_budget_policy = TQ_DENSE_BUDGET_AUTO;
	tq_last_graph_rescore_band_policy = TQ_RESCORE_BAND_POLICY_AUTO;
	tq_last_graph_scoring_kernel = TQ_SCORING_SCALAR;
	tq_last_exact_vector_kernel = TQ_EXACT_KERNEL_SCALAR;
	tq_last_exact_vector_kernel_recorded = false;
	tq_last_graph_storage_kind = HNSW_STORAGE_HNSW;
	tq_last_graph_tq_bits = 0;
	tq_last_graph_query_split_active = false;
}

void
HnswRecordFlatScanStats(void)
{
	HnswRecordNonGraphScanStats();
	tq_last_scan_orchestration = TQ_SCAN_ORCHESTRATION_FLAT;
	tq_last_graph_storage_kind = HNSW_STORAGE_TURBOQUANT_FLAT;
}

static const char *
TqScanOrchestrationName(void)
{
	switch (tq_last_scan_orchestration)
	{
		case TQ_SCAN_ORCHESTRATION_GRAPH:
			return tq_last_graph_storage_kind == HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE ? "graph_native" : "graph_hnsw";
		case TQ_SCAN_ORCHESTRATION_FLAT:
			return "flat";
		case TQ_SCAN_ORCHESTRATION_NONE:
		default:
			return "none";
	}
}

static const char *
TqExactKernelName(int kernel)
{
	switch ((TqExactKernel) kernel)
	{
		case TQ_EXACT_KERNEL_SCALAR:
			return "scalar";
		case TQ_EXACT_KERNEL_AUTOVEC_FMA:
			return "autovec_fma";
		case TQ_EXACT_KERNEL_NEON:
			return "neon";
		case TQ_EXACT_KERNEL_AVX2:
			return "avx2";
		case TQ_EXACT_KERNEL_AVX512F:
			return "avx512f";
		default:
			return "unknown";
	}
}

static const char *
TqGraphAvx512WeightedModeName(int mode)
{
	switch ((TqGraphAvx512WeightedMode) mode)
	{
		case TQ_GRAPH_AVX512_WEIGHTED_ON:
			return "on";
		case TQ_GRAPH_AVX512_WEIGHTED_AUTO:
			return "auto";
		case TQ_GRAPH_AVX512_WEIGHTED_OFF:
		default:
			return "off";
	}
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tq_last_scan_stats);
Datum
tq_last_scan_stats(PG_FUNCTION_ARGS)
{
	StringInfoData json;

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"scan_orchestration\":\"%s\","
					 "\"graph_visited_nodes\":" INT64_FORMAT ","
					 "\"graph_scored_codes\":" INT64_FORMAT ","
					 "\"graph_batch_scored_codes\":" INT64_FORMAT ","
					 "\"graph_scalar_scored_codes\":" INT64_FORMAT ","
					 "\"graph_batch_kernel\":\"%s\","
					 "\"graph_batch_scoring\":\"%s\","
					 "\"graph_batch_size\":%d,"
					 "\"graph_candidate_count\":" INT64_FORMAT ","
					 "\"graph_rescore_count\":" INT64_FORMAT ","
					 "\"graph_rescore_pages\":" INT64_FORMAT ","
					 "\"graph_code_pages_read\":" INT64_FORMAT ","
					 "\"graph_adj_pages_read\":" INT64_FORMAT ","
					 "\"graph_entry_point_count\":" INT64_FORMAT ","
					 "\"graph_prepare_us\":" INT64_FORMAT ","
					 "\"graph_traverse_us\":" INT64_FORMAT ","
					 "\"graph_entry_us\":" INT64_FORMAT ","
					 "\"graph_base_us\":" INT64_FORMAT ","
					 "\"graph_batch_us\":" INT64_FORMAT ","
					 "\"graph_heap_us\":" INT64_FORMAT ","
					 "\"graph_fill_us\":" INT64_FORMAT ","
					 "\"graph_rescore_us\":" INT64_FORMAT ","
					 "\"graph_sort_us\":" INT64_FORMAT ","
					 "\"graph_total_us\":" INT64_FORMAT ","
					 "\"graph_dense_requested_k\":" INT64_FORMAT ","
					 "\"graph_effective_result_target\":" INT64_FORMAT ","
					 "\"graph_effective_search_ef\":" INT64_FORMAT ","
					 "\"graph_effective_rescore_band\":" INT64_FORMAT ","
					 "\"graph_highdim_widening_multiplier\":%.3f,"
					 "\"graph_widening_reason\":\"%s\","
					 "\"graph_dense_budget_policy\":\"%s\","
					 "\"graph_rescore_band_policy\":\"%s\","
					 "\"graph_scoring_kernel\":\"%s\","
					 "\"dense_simd_kernel\":\"%s\","
					 "\"dense_simd_force\":\"%s\","
					 "\"exact_simd_force\":\"%s\","
					 "\"graph_storage_kind\":\"%s\","
					 "\"graph_tuple_chasing\":%s,"
					 "\"graph_tq_bits\":%d,"
					 "\"graph_query_split_active\":%s}",
					 TqScanOrchestrationName(),
					 tq_last_graph_visited_nodes,
					 tq_last_graph_scored_codes,
					 tq_last_graph_batch_scored_codes,
					 tq_last_graph_scalar_scored_codes,
					 HnswTqScoringKernelName(tq_last_graph_batch_kernel),
					 hnsw_tq_graph_batch_scoring == TQ_GRAPH_BATCH_OFF ? "off" :
					 hnsw_tq_graph_batch_scoring == TQ_GRAPH_BATCH_ON ? "on" : "auto",
					 hnsw_tq_graph_batch_size,
					 tq_last_graph_candidate_count,
					 tq_last_graph_rescore_count,
					 tq_last_graph_rescore_pages,
					 tq_last_graph_code_pages_read,
					 tq_last_graph_adj_pages_read,
					 tq_last_graph_entry_point_count,
					 tq_last_graph_prepare_us,
					 tq_last_graph_traverse_us,
					 tq_last_graph_entry_us,
					 tq_last_graph_base_us,
					 tq_last_graph_batch_us,
					 tq_last_graph_heap_us,
					 tq_last_graph_fill_us,
					 tq_last_graph_rescore_us,
					 tq_last_graph_sort_us,
					 tq_last_graph_total_us,
					 tq_last_graph_dense_requested_k,
					 tq_last_graph_effective_result_target,
					 tq_last_graph_effective_search_ef,
					 tq_last_graph_effective_rescore_band,
					 tq_last_graph_highdim_widening_multiplier,
					 TqGraphDenseWideningReasonName(tq_last_graph_widening_reason),
					 TqGraphDenseBudgetPolicyNameExternal(tq_last_graph_dense_budget_policy),
					 TqGraphRescoreBandPolicyNameExternal(tq_last_graph_rescore_band_policy),
					 HnswTqScoringKernelName(tq_last_graph_scoring_kernel),
					 HnswTqScoringKernelName(tq_last_graph_scoring_kernel),
					 HnswTqSimdForceName(hnsw_tq_simd_force),
					 HnswTqExactSimdForceName(hnsw_tq_exact_simd_force),
					 HnswStorageKindName(tq_last_graph_storage_kind),
					 tq_last_graph_storage_kind == HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE ? "false" : "true",
					 tq_last_graph_tq_bits,
					 tq_last_graph_query_split_active ? "true" : "false");

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tq_simd_capabilities);
Datum
tq_simd_capabilities(PG_FUNCTION_ARGS)
{
	StringInfoData json;

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"architecture\":\"%s\","
					 "\"simd_build_disabled\":%s,"
					 "\"dense_force\":\"%s\","
					 "\"exact_force\":\"%s\","
					 "\"bm25_force\":\"%s\","
					 "\"compile_avx2\":%s,"
					 "\"compile_avx512vnni\":%s,"
					 "\"compile_avx512vpopcntdq\":%s,"
					 "\"compile_avx512_weighted\":%s,"
					 "\"compile_avxvnni\":%s,"
					 "\"compile_arm_dotprod\":%s,"
					 "\"compile_arm_i8mm\":%s,"
					 "\"avx512_weighted\":\"%s\"}",
#if defined(__aarch64__) || defined(_M_ARM64)
					 "arm64",
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
					 "amd64",
#else
					 "unknown",
#endif
#if defined(TQ_DISABLE_SIMD)
					 "true",
#else
					 "false",
#endif
					 HnswTqSimdForceName(hnsw_tq_simd_force),
					 HnswTqExactSimdForceName(hnsw_tq_exact_simd_force),
					 TqHybridBm25SimdForceName(tqhybrid_bm25_simd_force),
#if !defined(TQ_DISABLE_SIMD) && (defined(__AVX2__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
					 "true",
#else
					 "false",
#endif
#if !defined(TQ_DISABLE_SIMD) && (defined(__AVX512VNNI__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
					 "true",
#else
					 "false",
#endif
#if !defined(TQ_DISABLE_SIMD) && (defined(__AVX512VPOPCNTDQ__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
					 "true",
#else
					 "false",
#endif
#if !defined(TQ_DISABLE_SIMD) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
					 "true",
#else
					 "false",
#endif
#if !defined(TQ_DISABLE_SIMD) && (defined(__AVXVNNI__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
					 "true",
#else
					 "false",
#endif
#if !defined(TQ_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
					 "true",
					 "true",
#else
					 "false",
					 "false",
#endif
					 TqGraphAvx512WeightedModeName(hnsw_tq_graph_avx512_weighted)
		);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(tq_last_simd_stats);
Datum
tq_last_simd_stats(PG_FUNCTION_ARGS)
{
	StringInfoData json;

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"dense_graph_query_split_kernel\":\"%s\","
					 "\"dense_simd_kernel_counts\":{\"%s\":" INT64_FORMAT "},"
					 "\"graph_batch_scored_codes\":" INT64_FORMAT ","
					 "\"graph_scalar_scored_codes\":" INT64_FORMAT ","
					 "\"graph_batch_kernel\":\"%s\","
					 "\"exact_vector_kernel\":\"%s\","
					 "\"asymmetric_1bit_kernel\":\"scalar\","
					 "\"weighted_code_code_kernel\":\"%s\","
					 "\"avx512_weighted\":\"%s\","
					 "\"bm25_decode_kernel\":\"%s\","
					 "\"bm25_score_kernel\":\"%s\","
					 "\"bm25_simd_force\":\"%s\","
					 "\"bm25_simd_blocks\":" UINT64_FORMAT ","
					 "\"bm25_scalar_tail_postings\":" UINT64_FORMAT "}",
					 HnswTqScoringKernelName(tq_last_graph_scoring_kernel),
					 HnswTqScoringKernelName(tq_last_graph_scoring_kernel),
					 tq_last_graph_scored_codes,
					 tq_last_graph_batch_scored_codes,
					 tq_last_graph_scalar_scored_codes,
					 HnswTqScoringKernelName(tq_last_graph_batch_kernel),
					 TqExactKernelName(tq_last_exact_vector_kernel),
					 HnswTqScoringKernelName(tq_last_weighted_code_code_kernel),
					 TqGraphAvx512WeightedModeName(hnsw_tq_graph_avx512_weighted),
					 TqHybridBm25KernelName(tqhybrid_last_bm25_decode_kernel),
					 TqHybridBm25KernelName(tqhybrid_last_bm25_score_kernel),
					 TqHybridBm25SimdForceName(tqhybrid_bm25_simd_force),
					 tqhybrid_last_bm25_simd_blocks,
					 tqhybrid_last_bm25_scalar_tail_postings);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}
