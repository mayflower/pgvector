#include "postgres.h"

#include "fmgr.h"
#include "hnsw.h"
#include "lib/stringinfo.h"
#include "utils/fmgrprotos.h"

#define TQ_SCAN_ORCHESTRATION_NONE		0
#define TQ_SCAN_ORCHESTRATION_GRAPH		1
#define TQ_SCAN_ORCHESTRATION_FLAT		2

static int64 tq_last_graph_visited_nodes = 0;
static int64 tq_last_graph_scored_codes = 0;
static int64 tq_last_graph_candidate_count = 0;
static int64 tq_last_graph_rescore_count = 0;
static int64 tq_last_graph_rescore_pages = 0;
static int64 tq_last_graph_code_pages_read = 0;
static int64 tq_last_graph_adj_pages_read = 0;
static int64 tq_last_graph_entry_point_count = 0;
static int64 tq_last_graph_prepare_us = 0;
static int64 tq_last_graph_traverse_us = 0;
static int64 tq_last_graph_fill_us = 0;
static int64 tq_last_graph_rescore_us = 0;
static int64 tq_last_graph_sort_us = 0;
static int64 tq_last_graph_total_us = 0;
static int tq_last_graph_scoring_kernel = TQ_SCORING_SCALAR;
static int tq_last_graph_storage_kind = HNSW_STORAGE_HNSW;
static int tq_last_scan_orchestration = TQ_SCAN_ORCHESTRATION_NONE;
static int tq_last_graph_tq_bits = 0;
static bool tq_last_graph_query_split_active = false;

static const char *TqScanOrchestrationName(void);

void
HnswRecordGraphScanStats(HnswScanOpaque so)
{
	if (so == NULL || !so->turboquantGraphScan)
		return;

	tq_last_scan_orchestration = TQ_SCAN_ORCHESTRATION_GRAPH;
	tq_last_graph_visited_nodes = so->graphVisitedNodes;
	tq_last_graph_scored_codes = so->graphScoredCodes;
	tq_last_graph_candidate_count = so->graphCandidateCount;
	tq_last_graph_rescore_count = so->graphRescoreCount;
	tq_last_graph_rescore_pages = so->graphRescorePages;
	tq_last_graph_code_pages_read = so->graphCodePagesRead;
	tq_last_graph_adj_pages_read = so->graphAdjPagesRead;
	tq_last_graph_entry_point_count = so->graphEntryPointCount;
	tq_last_graph_prepare_us = so->graphPrepareUs;
	tq_last_graph_traverse_us = so->graphTraverseUs;
	tq_last_graph_fill_us = so->graphFillUs;
	tq_last_graph_rescore_us = so->graphRescoreUs;
	tq_last_graph_sort_us = so->graphSortUs;
	tq_last_graph_total_us = so->graphTotalUs;
	tq_last_graph_scoring_kernel = so->tq.enabled ? so->tq.scoringKernel : TQ_SCORING_SCALAR;
	tq_last_graph_storage_kind = so->graphStorageKind;
	tq_last_graph_tq_bits = so->tq.enabled ? so->tq.bits : 0;
#if defined(__aarch64__) || defined(_M_ARM64) || \
	defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
	tq_last_graph_query_split_active = so->tq.enabled && so->tq.querySplitEnabled;
#else
	tq_last_graph_query_split_active = false;
#endif
}

void
HnswRecordNonGraphScanStats(void)
{
	tq_last_scan_orchestration = TQ_SCAN_ORCHESTRATION_NONE;
	tq_last_graph_visited_nodes = 0;
	tq_last_graph_scored_codes = 0;
	tq_last_graph_candidate_count = 0;
	tq_last_graph_rescore_count = 0;
	tq_last_graph_rescore_pages = 0;
	tq_last_graph_code_pages_read = 0;
	tq_last_graph_adj_pages_read = 0;
	tq_last_graph_entry_point_count = 0;
	tq_last_graph_prepare_us = 0;
	tq_last_graph_traverse_us = 0;
	tq_last_graph_fill_us = 0;
	tq_last_graph_rescore_us = 0;
	tq_last_graph_sort_us = 0;
	tq_last_graph_total_us = 0;
	tq_last_graph_scoring_kernel = TQ_SCORING_SCALAR;
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
					 "\"graph_candidate_count\":" INT64_FORMAT ","
					 "\"graph_rescore_count\":" INT64_FORMAT ","
					 "\"graph_rescore_pages\":" INT64_FORMAT ","
					 "\"graph_code_pages_read\":" INT64_FORMAT ","
					 "\"graph_adj_pages_read\":" INT64_FORMAT ","
					 "\"graph_entry_point_count\":" INT64_FORMAT ","
					 "\"graph_prepare_us\":" INT64_FORMAT ","
					 "\"graph_traverse_us\":" INT64_FORMAT ","
					 "\"graph_fill_us\":" INT64_FORMAT ","
					 "\"graph_rescore_us\":" INT64_FORMAT ","
					 "\"graph_sort_us\":" INT64_FORMAT ","
					 "\"graph_total_us\":" INT64_FORMAT ","
					 "\"graph_scoring_kernel\":\"%s\","
					 "\"graph_storage_kind\":\"%s\","
					 "\"graph_tuple_chasing\":%s,"
					 "\"graph_tq_bits\":%d,"
					 "\"graph_query_split_active\":%s}",
					 TqScanOrchestrationName(),
					 tq_last_graph_visited_nodes,
					 tq_last_graph_scored_codes,
					 tq_last_graph_candidate_count,
					 tq_last_graph_rescore_count,
					 tq_last_graph_rescore_pages,
					 tq_last_graph_code_pages_read,
					 tq_last_graph_adj_pages_read,
					 tq_last_graph_entry_point_count,
					 tq_last_graph_prepare_us,
					 tq_last_graph_traverse_us,
					 tq_last_graph_fill_us,
					 tq_last_graph_rescore_us,
					 tq_last_graph_sort_us,
					 tq_last_graph_total_us,
					 HnswTqScoringKernelName(tq_last_graph_scoring_kernel),
					 HnswStorageKindName(tq_last_graph_storage_kind),
					 tq_last_graph_storage_kind == HNSW_STORAGE_TURBOQUANT_GRAPH_NATIVE ? "false" : "true",
					 tq_last_graph_tq_bits,
					 tq_last_graph_query_split_active ? "true" : "false");

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}
