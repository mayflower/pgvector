#ifndef TQHYBRID_H
#define TQHYBRID_H

#include "postgres.h"

#include "access/genam.h"
#include "nodes/plannodes.h"
#include "nodes/pathnodes.h"
#include "utils/rel.h"

typedef struct TqHybridOptions
{
	int32		vl_len_;
	int			m;
	int			efConstruction;
	int			routing;
	int			graphEfSearch;
	int			graphOversampling;
	int			graphRescoreBand;
	int			graphExactCache;
	int			graphReorder;
	int			tqBits;
	bool		tqWeighted;
	bool		tqQuantileFit;
	bool		tqRenorm;
	bool		tqExactStorage;
	float8		bm25K1;
	float8		bm25B;
	bool		bm25BlockMax;
	bool		bm25PrecomputeTfNorm;
	int			bm25DeltaCompactionThreshold;
	int			hybridDefaultFusion;
	int			hybridDefaultDenseK;
	int			hybridDefaultBm25K;
	int			hybridDefaultRrfK;
}			TqHybridOptions;

extern bool tqhybrid_enable_wand;
extern bool tqhybrid_debug_stats;
extern int	tqhybrid_max_union_candidates;
extern int	tqhybrid_default_dense_k;
extern int	tqhybrid_default_bm25_k;
extern int	tqhybrid_default_rrf_k;
extern int	tqhybrid_force_fusion;
extern int	tqhybrid_fusion_hash_threshold;
extern int	tqhybrid_debug_postings_chunk_size;
extern bool tqhybrid_enable_exact_rescore_for_bm25_only;
extern int	tqhybrid_bm25_cache_max_mb;
extern int	tqhybrid_bm25_simd_force;

typedef enum TqHybridBm25SimdForce
{
	TQHYBRID_BM25_SIMD_FORCE_AUTO,
	TQHYBRID_BM25_SIMD_FORCE_SCALAR,
	TQHYBRID_BM25_SIMD_FORCE_AVX2,
	TQHYBRID_BM25_SIMD_FORCE_NEON
}			TqHybridBm25SimdForce;

const char *TqHybridBm25SimdForceName(int force);

void		TqHybridInit(void);
PlannedStmt *TqHybridCurrentPlannedStmt(void);
Datum		turbohybridhandler(PG_FUNCTION_ARGS);

#endif
