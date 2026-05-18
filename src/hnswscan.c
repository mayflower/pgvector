#include "postgres.h"

#include <math.h>
#include <limits.h>

#include "access/genam.h"
#include "access/relscan.h"
#include "fmgr.h"
#include "hnsw.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "nodes/pg_list.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/relcache.h"
#include "utils/snapmgr.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

/*
 * Get the initial iterative scan batch size
 */
static int
GetInitialIterativeBatchSize(HnswScanOpaque so)
{
	return so->efSearch;
}

static int
CompareSearchCandidateDistances(const ListCell *a, const ListCell *b)
{
	HnswSearchCandidate *sca = lfirst(a);
	HnswSearchCandidate *scb = lfirst(b);

	if (sca->distance < scb->distance)
		return 1;

	if (sca->distance > scb->distance)
		return -1;

	return 0;
}

static int
GetGraphAutoRescoreLimit(HnswScanOpaque so, List *items)
{
	/*
	 * The packed-code graph scorer is for traversal, not final ranking.
	 * Exact-rescore the complete candidate band returned by traversal so
	 * auto mode cannot drop true top-k rows due to code-domain ordering.
	 * This remains final-band-only: rescore reads only pages for returned
	 * candidates, not every visited page.
	 */
	return list_length(items);
}

static List *
SelectGraphRescoreBand(HnswScanOpaque so, List *items)
{
	List	   *band = NIL;
	int			rescoreLimit;
	int			skip;
	int			pos = 0;
	ListCell   *lc;

	if (so->graphRescoreBand == TQ_GRAPH_RESCORE_BAND_EXACT)
		return items;

	rescoreLimit = GetGraphAutoRescoreLimit(so, items);
	if (rescoreLimit >= list_length(items))
		return items;

	list_sort(items, CompareSearchCandidateDistances);
	skip = list_length(items) - rescoreLimit;

	foreach(lc, items)
	{
		if (pos++ >= skip)
			band = lappend(band, lfirst(lc));
	}

	return band;
}

static void
RescoreScanItems(IndexScanDesc scan, List *items)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	List	   *rescoreItems;

	if (!so->turboquantGraphScan ||
		so->graphRescoreBand == TQ_GRAPH_RESCORE_BAND_NONE ||
		items == NIL)
		return;

	rescoreItems = SelectGraphRescoreBand(so, items);
	so->graphRescorePages += HnswRescoreSearchCandidates(scan->indexRelation, &so->support, &so->q, rescoreItems);
	so->graphRescoreCount += list_length(rescoreItems);
	list_sort(items, CompareSearchCandidateDistances);
}

static void
RecordGraphScanBatchStats(HnswScanOpaque so, int64 beforeTuples, int64 afterTuples, List *items)
{
	if (!so->turboquantGraphScan)
		return;

	so->graphVisitedNodes += afterTuples - beforeTuples;
	so->graphCandidateCount += list_length(items);
	HnswRecordGraphScanStats(so);
}

static inline double
GetFlatDistance(Datum value, HnswElementTuple etup, HnswSupport *support)
{
	if (DatumGetPointer(value) == NULL)
		return 0;

	return DatumGetFloat8(FunctionCall2Coll(support->procinfo,
											support->collation,
											value,
											PointerGetDatum(&etup->data)));
}

/*
 * Exact flat scan over element pages for explicit turboquant routing=flat.
 */
static List *
GetFlatScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	BlockNumber blkno = HNSW_HEAD_BLKNO;
	List	   *items = NIL;
	char	   *base = NULL;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		BlockNumber nextblkno;
		OffsetNumber offno;
		OffsetNumber maxoffno;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoffno = PageGetMaxOffsetNumber(page);

		for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			HnswElementTuple etup;
			HnswElement element;
			HnswSearchCandidate *sc;

			if (!ItemIdIsUsed(iid))
				continue;

			etup = (HnswElementTuple) PageGetItem(page, iid);

			if (!HnswIsElementTuple(etup) || etup->deleted ||
				!ItemPointerIsValid(&etup->heaptids[0]))
				continue;

			element = HnswInitElementFromBlock(blkno, offno);
			HnswLoadElementFromTuple(element, etup, true, false);

			sc = palloc(sizeof(HnswSearchCandidate));
			HnswPtrStore(base, sc->element, element);
			sc->distance = GetFlatDistance(value, etup, &so->support);

			items = lappend(items, sc);
			so->tuples++;
		}

		nextblkno = HnswPageGetOpaque(page)->nextblkno;
		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
	}

	list_sort(items, CompareSearchCandidateDistances);

	return items;
}

/*
 * Algorithm 5 from paper
 */
static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	HnswSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	HnswElement entryPoint;
	char	   *base = NULL;
	HnswQuery  *q = &so->q;
	int			searchEf;
	int64		beforeTuples;
	int64		afterTuples;
	List	   *items;

	/* Get m and entry point */
	HnswGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;
	if (so->turboquantGraphScan)
		HnswPrepareTqQuery(index, support, value, &so->tq);

	if (entryPoint == NULL)
		return NIL;

	so->graphEntryPointCount = 1;
	ep = list_make1(HnswEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = HnswSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL, -1, so->turboquantGraphScan ? &so->graphScoredCodes : NULL, so->turboquantGraphScan ? &so->tq : NULL);
		ep = w;
	}

	searchEf = GetInitialIterativeBatchSize(so);
	if (so->turboquantGraphScan)
	{
		/*
		 * Keep graph_oversampling on traversal breadth as well as the final
		 * exact rescore band. Half-width traversal drops true neighbors on
		 * the smoke workload even with near-threshold exact refinement.
		 */
		searchEf = Min(searchEf * Max(so->graphOversampling, 1) * 2,
					   hnsw_max_scan_tuples);
	}

	beforeTuples = so->tuples;
	items = HnswSearchLayer(base, q, ep, searchEf, 0, index, support, m, false, NULL, &so->v, hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples, -1, so->turboquantGraphScan ? &so->graphScoredCodes : NULL, so->turboquantGraphScan ? &so->tq : NULL);
	afterTuples = so->tuples;
	RescoreScanItems(scan, items);

	RecordGraphScanBatchStats(so, beforeTuples, afterTuples, items);

	return items;
}

/*
 * Resume scan at ground level with discarded candidates
 */
static List *
ResumeScanItems(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	List	   *ep = NIL;
	char	   *base = NULL;
	int			batch_size = so->efSearch;
	int64		beforeTuples;
	int64		afterTuples;
	List	   *items;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		HnswSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));

		ep = lappend(ep, sc);
	}

	beforeTuples = so->tuples;
	items = HnswSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false, &so->tuples, -1, so->turboquantGraphScan ? &so->graphScoredCodes : NULL, so->turboquantGraphScan ? &so->tq : NULL);
	afterTuples = so->tuples;
	RescoreScanItems(scan, items);

	RecordGraphScanBatchStats(so, beforeTuples, afterTuples, items);

	return items;
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
			value = HnswNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

#if defined(HNSW_MEMORY)
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(HnswScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif

/*
 * Prepare for an index scan
 */
IndexScanDesc
hnswbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (HnswScanOpaque) palloc0(sizeof(HnswScanOpaqueData));
	so->typeInfo = HnswGetTypeInfo(index);

	/* Set support functions */
	HnswInitSupport(&so->support, index);

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Hnsw scan temporary context",
									   0, 8 * 1024, 256 * 1024);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * hnsw_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) (SIZE_MAX / 2));
	so->first = true;
	so->efSearch = HnswGetEfSearch(index);
	so->graphOversampling = HnswGetGraphOversampling(index);
	so->graphRescoreBand = HnswGetGraphRescoreBand(index);
	so->graphStorageKind = HnswGetMetaPageStorageKind(index);
	so->turboquantGraphScan = HnswUseTqGraph(index);
	so->turboquantFlatScan = HnswUseTqFlat(index);
	so->previousDistance = -get_float8_infinity();

	scan->opaque = so;

	return scan;
}

IndexScanDesc
turboquantbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	HnswScanOpaque so;

	if (HnswUseTqNativeGraph(index))
		return tqgraphbeginscan(index, nkeys, norderbys);

	if (!HnswUseTqFlat(index))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant requires a native graph-compatible opclass"),
				 errhint("Use the hnsw or ivfflat access method directly for this opclass.")));

	HnswSetForceTurboquantIndex(true);
	PG_TRY();
	{
		scan = hnswbeginscan(index, nkeys, norderbys);
	}
	PG_CATCH();
	{
		HnswSetForceTurboquantIndex(false);
		PG_RE_THROW();
	}
	PG_END_TRY();
	HnswSetForceTurboquantIndex(false);

	so = (HnswScanOpaque) scan->opaque;
	so->turboquantGraphScan = HnswUseTqGraph(index);
	so->turboquantFlatScan = HnswUseTqFlat(index);

	return scan;
}

/*
 * Start or restart an index scan
 */
void
hnswrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	so->first = true;
	/* v and discarded are allocated in tmpCtx */
	so->v.tids = NULL;
	so->discarded = NULL;
	so->hasTupleTargetRows = false;
	so->hasEstimatedFilterSelectivity = false;
	so->hasInitialEffectiveEfSearch = false;
	so->returnedRows = 0;
	so->tupleTargetRows = -1;
	so->estimatedFilterSelectivity = -1.0;
	so->efSearch = HnswGetEfSearch(scan->indexRelation);
	so->graphOversampling = HnswGetGraphOversampling(scan->indexRelation);
	so->graphRescoreBand = HnswGetGraphRescoreBand(scan->indexRelation);
	so->graphStorageKind = HnswGetMetaPageStorageKind(scan->indexRelation);
	so->turboquantGraphScan = HnswUseTqGraph(scan->indexRelation);
	so->turboquantFlatScan = HnswUseTqFlat(scan->indexRelation);
	so->initialEffectiveEfSearch = so->efSearch;
	so->tuples = 0;
	so->graphVisitedNodes = 0;
	so->graphScoredCodes = 0;
	so->graphCandidateCount = 0;
	so->graphRescoreCount = 0;
	so->graphRescorePages = 0;
	so->graphEntryPointCount = 0;
	memset(&so->tq, 0, sizeof(HnswTqQuery));
	so->previousDistance = -get_float8_infinity();
	MemoryContextReset(so->tmpCtx);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

void
turboquantrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	HnswScanOpaque so;

	if (HnswUseTqNativeGraph(scan->indexRelation))
	{
		tqgraphrescan(scan, keys, nkeys, orderbys, norderbys);
		return;
	}

	if (!HnswUseTqFlat(scan->indexRelation))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant requires a native graph-compatible opclass"),
				 errhint("Use the hnsw or ivfflat access method directly for this opclass.")));

	HnswSetForceTurboquantIndex(true);
	PG_TRY();
	{
		hnswrescan(scan, keys, nkeys, orderbys, norderbys);
	}
	PG_CATCH();
	{
		HnswSetForceTurboquantIndex(false);
		PG_RE_THROW();
	}
	PG_END_TRY();
	HnswSetForceTurboquantIndex(false);

	so = (HnswScanOpaque) scan->opaque;
	so->turboquantGraphScan = HnswUseTqGraph(scan->indexRelation);
	so->turboquantFlatScan = HnswUseTqFlat(scan->indexRelation);
}

/*
 * Fetch the next tuple in the given scan
 */
bool
hnswgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan hnsw index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with hnsw");

		/* Get scan value */
		value = GetScanValue(scan);
		if (so->turboquantFlatScan)
			HnswRecordFlatScanStats();
		else if (!so->turboquantGraphScan)
			HnswRecordNonGraphScanStats();

		/*
		 * Get a shared lock. This allows vacuum to ensure no in-flight scans
		 * before marking tuples as deleted.
		 */
		LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		if (so->turboquantFlatScan)
			so->w = GetFlatScanItems(scan, value);
		else
			so->w = GetScanItems(scan, value);

		/* Release shared lock */
		UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

		so->first = false;

#if defined(HNSW_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		HnswSearchCandidate *sc;
		HnswElement element;
		ItemPointer heaptid;

		if (list_length(so->w) == 0)
		{
			if (so->turboquantFlatScan)
				break;

			if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_OFF)
				break;

			/* Empty index */
			if (so->discarded == NULL)
				break;

			/* Reached max number of tuples or memory limit */
			if (so->tuples >= hnsw_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
			{
				if (pairingheap_is_empty(so->discarded))
					break;

				/* Return remaining tuples */
				so->w = lappend(so->w, HnswGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
			}
			else
			{
				/*
				 * Locking ensures when neighbors are read, the elements they
				 * reference will not be deleted (and replaced) during the
				 * iteration.
				 *
				 * Elements loaded into memory on previous iterations may have
				 * been deleted (and replaced), so when reading neighbors, the
				 * element version must be checked.
				 */
				LockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

				so->w = ResumeScanItems(scan);

				UnlockPage(scan->indexRelation, HNSW_SCAN_LOCK, ShareLock);

#if defined(HNSW_MEMORY)
				ShowMemoryUsage(so);
#endif
			}

			if (list_length(so->w) == 0)
				break;
		}

		sc = llast(so->w);
		element = HnswPtrAccess(base, sc->element);

		/* Move to next element if no valid heap TIDs */
		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);

			/* Mark memory as free for next iteration */
			if (hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF)
			{
				pfree(element);
				pfree(sc);
			}

			continue;
		}

		heaptid = &element->heaptids[--element->heaptidsLength];

		if (hnsw_iterative_scan == HNSW_ITERATIVE_SCAN_STRICT)
		{
			if (sc->distance < so->previousDistance)
				continue;

			so->previousDistance = sc->distance;
		}

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

bool
turboquantgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	bool		result;

	if (HnswUseTqNativeGraph(scan->indexRelation))
		return tqgraphgettuple(scan, dir);

	if (!HnswUseTqFlat(scan->indexRelation))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant requires a native graph-compatible opclass"),
				 errhint("Use the hnsw or ivfflat access method directly for this opclass.")));

	HnswSetForceTurboquantIndex(true);
	PG_TRY();
	{
		if (so != NULL && !so->turboquantGraphScan && HnswUseTqGraph(scan->indexRelation))
		{
			so->turboquantGraphScan = true;
			so->turboquantFlatScan = false;
			so->efSearch = HnswGetEfSearch(scan->indexRelation);
			so->graphOversampling = HnswGetGraphOversampling(scan->indexRelation);
			so->graphRescoreBand = HnswGetGraphRescoreBand(scan->indexRelation);
			so->graphStorageKind = HnswGetMetaPageStorageKind(scan->indexRelation);
			so->initialEffectiveEfSearch = so->efSearch;
		}
		else if (so != NULL)
			so->turboquantFlatScan = HnswUseTqFlat(scan->indexRelation);

		result = hnswgettuple(scan, dir);
	}
	PG_CATCH();
	{
		HnswSetForceTurboquantIndex(false);
		PG_RE_THROW();
	}
	PG_END_TRY();
	HnswSetForceTurboquantIndex(false);

	return result;
}

/*
 * End a scan and release resources
 */
void
hnswendscan(IndexScanDesc scan)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}

void
turboquantendscan(IndexScanDesc scan)
{
	if (HnswUseTqNativeGraph(scan->indexRelation))
	{
		tqgraphendscan(scan);
		return;
	}

	if (!HnswUseTqFlat(scan->indexRelation))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turboquant requires a native graph-compatible opclass"),
				 errhint("Use the hnsw or ivfflat access method directly for this opclass.")));

	hnswendscan(scan);
}
