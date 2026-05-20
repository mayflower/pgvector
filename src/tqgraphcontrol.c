#include "postgres.h"

#include <string.h>

#include "access/amapi.h"
#include "access/relation.h"
#include "access/relscan.h"
#include "catalog/pg_type_d.h"
#include "commands/explain.h"
#include "executor/executor.h"
#include "hnsw.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "nodes/nodeFuncs.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "pgstat.h"
#include "storage/lmgr.h"
#include "utils/fmgroids.h"
#include "tcop/tcopprot.h"
#include "tqgraph.h"
#include "utils/json.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"

typedef struct TqGraphExecWrapperState
{
	PlanState   *planstate;
	ExecProcNodeMtd original_exec_proc_node;
	LimitState  *limitstate;
}			TqGraphExecWrapperState;

static ExecutorStart_hook_type prev_ExecutorStart_hook = NULL;
static ExecutorEnd_hook_type prev_ExecutorEnd_hook = NULL;
static List *tqgraph_exec_wrapper_states = NIL;
static int64 tqgraph_active_limit_tuple_target = -1;
static double tqgraph_active_estimated_filter_selectivity = -1.0;
static bool tqgraph_active_payload_filter_valid = false;
static AttrNumber tqgraph_active_payload_filter_attno = InvalidAttrNumber;
static int32 tqgraph_active_payload_filter_value = 0;

static void TqGraphExecutorStartHook(QueryDesc *queryDesc, int eflags);
static void TqGraphExecutorEndHook(QueryDesc *queryDesc);
static TupleTableSlot *TqGraphExecIndexScanWithController(PlanState *planstate);
static void TqGraphWrapControlledIndexScans(PlanState *planstate, LimitState *limitstate);
static TqGraphExecWrapperState *TqGraphFindWrapperState(PlanState *planstate);
static bool TqGraphIsIndexScanState(PlanState *planstate);
static bool TqGraphIndexScanUsesTurboquant(IndexScanState *indexstate);
static int64 TqGraphGetLimitTupleTarget(LimitState *limitstate);
static double TqGraphEstimateFilterSelectivity(IndexScanState *indexstate);
static bool TqGraphExtractPayloadInt4Filter(IndexScanState *indexstate,
										 AttrNumber *heap_attno,
										 int32 *value);
static bool TqGraphExtractPayloadInt4FilterExpr(Node *node,
											 AttrNumber *heap_attno,
											 int32 *value);
int64
TqGraphGetActiveLimitTupleTarget(void)
{
	return tqgraph_active_limit_tuple_target;
}

double
TqGraphGetActiveEstimatedFilterSelectivity(void)
{
	return tqgraph_active_estimated_filter_selectivity;
}

bool
TqGraphGetActivePayloadInt4Filter(AttrNumber *heap_attno, int32 *value)
{
	if (!tqgraph_active_payload_filter_valid)
		return false;

	if (heap_attno != NULL)
		*heap_attno = tqgraph_active_payload_filter_attno;
	if (value != NULL)
		*value = tqgraph_active_payload_filter_value;

	return true;
}

void
TqGraphControlInit(void)
{
	prev_ExecutorStart_hook = ExecutorStart_hook;
	ExecutorStart_hook = TqGraphExecutorStartHook;

	prev_ExecutorEnd_hook = ExecutorEnd_hook;
	ExecutorEnd_hook = TqGraphExecutorEndHook;
}

static void
TqGraphExecutorStartHook(QueryDesc *queryDesc, int eflags)
{
	if (prev_ExecutorStart_hook)
		prev_ExecutorStart_hook(queryDesc, eflags);
	else
		standard_ExecutorStart(queryDesc, eflags);

	tqgraph_exec_wrapper_states = NIL;
	TqGraphWrapControlledIndexScans(queryDesc->planstate, NULL);
}

static void
TqGraphExecutorEndHook(QueryDesc *queryDesc)
{
	tqgraph_exec_wrapper_states = NIL;

	if (prev_ExecutorEnd_hook)
		prev_ExecutorEnd_hook(queryDesc);
	else
		standard_ExecutorEnd(queryDesc);
}

static TupleTableSlot *
TqGraphExecIndexScanWithController(PlanState *planstate)
{
	TqGraphExecWrapperState *wrapper_state = TqGraphFindWrapperState(planstate);
	IndexScanState *indexstate = castNode(IndexScanState, planstate);
	TupleTableSlot *slot;
	IndexScanDesc scan;
	HnswScanOpaque so;
	int64		prev_limit_tuple_target = tqgraph_active_limit_tuple_target;
	double		prev_estimated_filter_selectivity = tqgraph_active_estimated_filter_selectivity;
	bool		prev_payload_filter_valid = tqgraph_active_payload_filter_valid;
	AttrNumber	prev_payload_filter_attno = tqgraph_active_payload_filter_attno;
	int32		prev_payload_filter_value = tqgraph_active_payload_filter_value;
	int64		tuple_target = TqGraphGetLimitTupleTarget(wrapper_state->limitstate);
	double		estimated_filter_selectivity = TqGraphEstimateFilterSelectivity(indexstate);
	AttrNumber	payload_filter_attno = InvalidAttrNumber;
	int32		payload_filter_value = 0;
	bool		payload_filter_valid =
		TqGraphExtractPayloadInt4Filter(indexstate, &payload_filter_attno,
									 &payload_filter_value);

	tqgraph_active_limit_tuple_target = tuple_target;
	tqgraph_active_estimated_filter_selectivity = estimated_filter_selectivity;
	tqgraph_active_payload_filter_valid = payload_filter_valid;
	tqgraph_active_payload_filter_attno = payload_filter_attno;
	tqgraph_active_payload_filter_value = payload_filter_value;

	slot = wrapper_state->original_exec_proc_node(planstate);
	tqgraph_active_limit_tuple_target = prev_limit_tuple_target;
	tqgraph_active_estimated_filter_selectivity = prev_estimated_filter_selectivity;
	tqgraph_active_payload_filter_valid = prev_payload_filter_valid;
	tqgraph_active_payload_filter_attno = prev_payload_filter_attno;
	tqgraph_active_payload_filter_value = prev_payload_filter_value;

	scan = indexstate->iss_ScanDesc;
	if (scan == NULL || scan->opaque == NULL)
		return slot;

	so = (HnswScanOpaque) scan->opaque;
	TqGraphSeedScanContext(so, tuple_target, estimated_filter_selectivity);

	if (!TupIsNull(slot))
		so->returnedRows++;

	return slot;
}

static void
TqGraphWrapControlledIndexScans(PlanState *planstate, LimitState *limitstate)
{
	if (planstate == NULL)
		return;

	if (IsA(planstate, LimitState))
	{
		TqGraphWrapControlledIndexScans(outerPlanState(planstate),
									 castNode(LimitState, planstate));
		return;
	}

	if (TqGraphIsIndexScanState(planstate) &&
		TqGraphIndexScanUsesTurboquant(castNode(IndexScanState, planstate)))
	{
		TqGraphExecWrapperState *wrapper_state =
			palloc(sizeof(TqGraphExecWrapperState));

		wrapper_state->planstate = planstate;
		wrapper_state->original_exec_proc_node = planstate->ExecProcNodeReal;
		wrapper_state->limitstate = limitstate;
		tqgraph_exec_wrapper_states = lappend(tqgraph_exec_wrapper_states, wrapper_state);
		ExecSetExecProcNode(planstate, TqGraphExecIndexScanWithController);
	}

	if (IsA(planstate, AppendState))
	{
		AppendState *appendstate = castNode(AppendState, planstate);

		for (int i = 0; i < appendstate->as_nplans; i++)
			TqGraphWrapControlledIndexScans(appendstate->appendplans[i], limitstate);
		return;
	}

	if (IsA(planstate, MergeAppendState))
	{
		MergeAppendState *mergeappendstate = castNode(MergeAppendState, planstate);

		for (int i = 0; i < mergeappendstate->ms_nplans; i++)
			TqGraphWrapControlledIndexScans(mergeappendstate->mergeplans[i], limitstate);
		return;
	}

	if (IsA(planstate, ResultState))
	{
		ResultState *resultstate = castNode(ResultState, planstate);

		TqGraphWrapControlledIndexScans(outerPlanState(planstate),
									 resultstate->ps.qual == NULL ? limitstate : NULL);
		return;
	}

	if (IsA(planstate, SubqueryScanState))
	{
		SubqueryScanState *subquerystate = castNode(SubqueryScanState, planstate);

		TqGraphWrapControlledIndexScans(subquerystate->subplan,
									 subquerystate->ss.ps.qual == NULL ? limitstate : NULL);
		return;
	}

	if (IsA(planstate, GatherState) || IsA(planstate, GatherMergeState))
	{
		TqGraphWrapControlledIndexScans(outerPlanState(planstate), limitstate);
		return;
	}

	TqGraphWrapControlledIndexScans(outerPlanState(planstate), NULL);
	TqGraphWrapControlledIndexScans(innerPlanState(planstate), NULL);
}

static TqGraphExecWrapperState *
TqGraphFindWrapperState(PlanState *planstate)
{
	ListCell   *lc;

	foreach(lc, tqgraph_exec_wrapper_states)
	{
		TqGraphExecWrapperState *wrapper_state = lfirst(lc);

		if (wrapper_state->planstate == planstate)
			return wrapper_state;
	}

	elog(ERROR, "missing TurboQuant graph scan wrapper state");
	return NULL;
}

static bool
TqGraphIsIndexScanState(PlanState *planstate)
{
	return planstate != NULL && IsA(planstate, IndexScanState);
}

static bool
TqGraphIndexScanUsesTurboquant(IndexScanState *indexstate)
{
	return indexstate->iss_RelationDesc != NULL &&
		indexstate->iss_RelationDesc->rd_indam != NULL &&
		indexstate->iss_RelationDesc->rd_indam->amgettuple == turboquantgettuple;
}

static int64
TqGraphGetLimitTupleTarget(LimitState *limitstate)
{
	int64		tuple_target;

	if (limitstate == NULL || limitstate->noCount || limitstate->count < 0)
		return -1;

	tuple_target = limitstate->count;

	if (limitstate->offset > 0)
		tuple_target += limitstate->offset;

	return tuple_target;
}

static bool
TqGraphExtractPayloadInt4FilterExpr(Node *node, AttrNumber *heap_attno, int32 *value)
{
	OpExpr	   *op;
	Node	   *left;
	Node	   *right;
	Var		   *var = NULL;
	Const	   *constant = NULL;
	Oid			opfuncid;

	if (node == NULL || !IsA(node, OpExpr))
		return false;

	op = castNode(OpExpr, node);
	if (list_length(op->args) != 2)
		return false;

	opfuncid = op->opfuncid;
	if (!OidIsValid(opfuncid))
		opfuncid = get_opcode(op->opno);
	if (opfuncid != F_INT4EQ)
		return false;

	left = linitial(op->args);
	right = lsecond(op->args);

	if (IsA(left, Var) && IsA(right, Const))
	{
		var = castNode(Var, left);
		constant = castNode(Const, right);
	}
	else if (IsA(left, Const) && IsA(right, Var))
	{
		var = castNode(Var, right);
		constant = castNode(Const, left);
	}
	else
		return false;

	if (var->varattno <= 0 || var->vartype != INT4OID ||
		constant->consttype != INT4OID || constant->constisnull)
		return false;

	*heap_attno = var->varattno;
	*value = DatumGetInt32(constant->constvalue);
	return true;
}

static bool
TqGraphExtractPayloadInt4Filter(IndexScanState *indexstate, AttrNumber *heap_attno,
							 int32 *value)
{
	List	   *quals;
	ListCell   *lc;

	if (indexstate == NULL || indexstate->ss.ps.plan == NULL)
		return false;

	quals = indexstate->ss.ps.plan->qual;
	foreach(lc, quals)
	{
		if (TqGraphExtractPayloadInt4FilterExpr((Node *) lfirst(lc),
											 heap_attno, value))
			return true;
	}

	return false;
}

static double
TqGraphEstimateFilterSelectivity(IndexScanState *indexstate)
{
	Relation	heap_relation;
	double		reltuples;
	double		estimated_rows;
	Index		scanrelid;
	bool		close_heap_relation = false;
	RangeTblEntry *rte;

	if (indexstate == NULL)
		return -1.0;

	heap_relation = indexstate->ss.ss_currentRelation;
	if (heap_relation == NULL &&
		indexstate->ss.ps.state != NULL &&
		indexstate->ss.ps.plan != NULL)
	{
		scanrelid = castNode(Scan, indexstate->ss.ps.plan)->scanrelid;
		if (scanrelid > 0)
		{
			rte = exec_rt_fetch(scanrelid, indexstate->ss.ps.state);
			if (rte->relid != InvalidOid)
			{
				heap_relation = relation_open(rte->relid, AccessShareLock);
				close_heap_relation = true;
			}
		}
	}

	if (heap_relation == NULL || heap_relation->rd_rel == NULL)
		return -1.0;

	reltuples = heap_relation->rd_rel->reltuples;
	if (!(reltuples > 0))
		return -1.0;

	estimated_rows = indexstate->ss.ps.plan->plan_rows;
	if (estimated_rows < 0)
	{
		if (close_heap_relation)
			relation_close(heap_relation, AccessShareLock);
		return -1.0;
	}

	if (close_heap_relation)
		relation_close(heap_relation, AccessShareLock);

	return Max(Min(estimated_rows / reltuples, 1.0), 0.0);
}

static double
TqGraphClampEstimatedFilterSelectivity(double estimated_selectivity)
{
	double		min_selectivity;

	if (!(estimated_selectivity > 0))
		return 1.0;

	min_selectivity = 1.0 / (double) Max((int64) hnsw_max_scan_tuples, (int64) 1);

	return Max(Min(estimated_selectivity, 1.0), min_selectivity);
}

void
TqGraphSeedScanContext(HnswScanOpaque so, int64 tuple_target,
					double estimated_filter_selectivity)
{
	if (!so->hasTupleTargetRows && tuple_target >= 0)
	{
		so->hasTupleTargetRows = true;
		so->tupleTargetRows = tuple_target;
	}

	if (!so->hasEstimatedFilterSelectivity && estimated_filter_selectivity >= 0)
	{
		so->estimatedFilterSelectivity =
			TqGraphClampEstimatedFilterSelectivity(estimated_filter_selectivity);
		so->hasEstimatedFilterSelectivity = true;
	}

	if (!so->hasInitialEffectiveEfSearch)
	{
		so->initialEffectiveEfSearch = so->efSearch;
		so->hasInitialEffectiveEfSearch = true;
	}
}
