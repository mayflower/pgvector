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
	bool		bm25ImpactHead;
	int			bm25ImpactMinDf;
	int			bm25ImpactHeadK;
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
extern bool tqhybrid_bm25_force_full_sort;
extern int	tqhybrid_bm25_accumulator_mode;
extern int	tqhybrid_bm25_dense_accumulator_threshold;
extern double tqhybrid_bm25_dense_accumulator_df_ratio;
extern int	tqhybrid_bm25_strategy;
extern bool tqhybrid_auto_budget;
extern int	tqhybrid_auto_budget_min_dense_k;
extern int	tqhybrid_auto_budget_min_bm25_k;
extern int	tqhybrid_auto_budget_limit_multiplier;
extern int	tqhybrid_auto_budget_quality_cap;

typedef enum TqHybridBm25SimdForce
{
	TQHYBRID_BM25_SIMD_FORCE_AUTO,
	TQHYBRID_BM25_SIMD_FORCE_SCALAR,
	TQHYBRID_BM25_SIMD_FORCE_AVX2,
	TQHYBRID_BM25_SIMD_FORCE_NEON
}			TqHybridBm25SimdForce;

typedef enum TqHybridBm25AccumulatorMode
{
	TQHYBRID_BM25_ACCUMULATOR_HASH,
	TQHYBRID_BM25_ACCUMULATOR_DENSE,
	TQHYBRID_BM25_ACCUMULATOR_AUTO
}			TqHybridBm25AccumulatorMode;

typedef enum TqHybridBm25Strategy
{
	TQHYBRID_BM25_STRATEGY_AUTO,
	TQHYBRID_BM25_STRATEGY_IMPACT,
	TQHYBRID_BM25_STRATEGY_WAND,
	TQHYBRID_BM25_STRATEGY_DAAT_SIMD,
	TQHYBRID_BM25_STRATEGY_DAAT_HASH
}			TqHybridBm25Strategy;

typedef enum TqHybridBm25RuntimeStrategy
{
	TQHYBRID_BM25_RUNTIME_NONE,
	TQHYBRID_BM25_RUNTIME_IMPACT_SINGLE,
	TQHYBRID_BM25_RUNTIME_IMPACT_SEEDED_WAND,
	TQHYBRID_BM25_RUNTIME_WAND,
	TQHYBRID_BM25_RUNTIME_DAAT_SIMD,
	TQHYBRID_BM25_RUNTIME_DAAT_HASH
}			TqHybridBm25RuntimeStrategy;

const char *TqHybridBm25SimdForceName(int force);
const char *TqHybridBm25AccumulatorModeName(int mode);
const char *TqHybridBm25StrategyName(int strategy);
const char *TqHybridBm25RuntimeStrategyName(int strategy);

void		TqHybridInit(void);
PlannedStmt *TqHybridCurrentPlannedStmt(void);
Datum		turbohybridhandler(PG_FUNCTION_ARGS);

#endif
