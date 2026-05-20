#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define TQ_GRAPH_X86 1
#else
#define TQ_GRAPH_X86 0
#endif

#define TQ_QUERY_SPLIT_HIGH_COEF 256
#define TQ_GRAPH_CODEBOOK_SCALE (127.0 / 2.733)
#define TQ_GRAPH_CODEBOOK2_SCALE (127.0 / 1.510)

#if !defined(TQ_DISABLE_SIMD) && \
	(defined(__AVX2__) || (TQ_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))))
#define TQ_GRAPH_COMPILE_AVX2 1
#else
#define TQ_GRAPH_COMPILE_AVX2 0
#endif

#if !defined(TQ_DISABLE_SIMD) && \
	(defined(__AVX512VNNI__) || (TQ_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))))
#define TQ_GRAPH_COMPILE_AVX512VNNI 1
#else
#define TQ_GRAPH_COMPILE_AVX512VNNI 0
#endif

/*
 * AVX-VNNI runtime detection requires __builtin_cpu_supports("avxvnni"), which
 * GCC has since 11.x and Clang since 18. Older Clang (e.g. 17, used for the
 * LLVM JIT bitcode build of this extension on Ubuntu 24.04) rejects that
 * feature string and breaks the build, so gate accordingly.
 */
#if !defined(TQ_DISABLE_SIMD) && \
	(defined(__AVXVNNI__) || \
	(TQ_GRAPH_X86 && defined(__clang__) && __clang_major__ >= 18) || \
	(TQ_GRAPH_X86 && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11))
#define TQ_GRAPH_COMPILE_AVXVNNI 1
#else
#define TQ_GRAPH_COMPILE_AVXVNNI 0
#endif

#if TQ_GRAPH_COMPILE_AVX2 && !defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
#define TQ_GRAPH_AVX2_TARGET __attribute__((target("avx2")))
#else
#define TQ_GRAPH_AVX2_TARGET
#endif

#if TQ_GRAPH_COMPILE_AVX512VNNI && \
	!(defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__)) && \
	(defined(__GNUC__) || defined(__clang__))
#define TQ_GRAPH_AVX512VNNI_TARGET __attribute__((target("avx512vnni,avx512vl,avx512bw,avx2")))
#else
#define TQ_GRAPH_AVX512VNNI_TARGET
#endif

/*
 * AVX-512 VPOPCNTDQ compile gates.  Available since
 * Ice Lake (client) / Sapphire Rapids (server).  Compile coverage
 * mirrors AVX-512 VNNI (broad target() attribute), runtime detection
 * is more specific because Skylake-X has VNNI but not VPOPCNTDQ.
 */
#if !defined(TQ_DISABLE_SIMD) && \
	(defined(__AVX512VPOPCNTDQ__) || (TQ_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))))
#define TQ_GRAPH_COMPILE_AVX512VPOPCNTDQ 1
#else
#define TQ_GRAPH_COMPILE_AVX512VPOPCNTDQ 0
#endif

#if TQ_GRAPH_COMPILE_AVX512VPOPCNTDQ && \
	!(defined(__AVX512VPOPCNTDQ__) && defined(__AVX512BW__)) && \
	(defined(__GNUC__) || defined(__clang__))
#define TQ_GRAPH_AVX512VPOPCNTDQ_TARGET __attribute__((target("avx512vpopcntdq,avx512bw,avx512f")))
#else
#define TQ_GRAPH_AVX512VPOPCNTDQ_TARGET
#endif

#if !defined(TQ_DISABLE_SIMD) && \
	(defined(__AVX512F__) || (TQ_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))))
#define TQ_GRAPH_COMPILE_AVX512_WEIGHTED 1
#else
#define TQ_GRAPH_COMPILE_AVX512_WEIGHTED 0
#endif

#if TQ_GRAPH_COMPILE_AVX512_WEIGHTED && \
	!(defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)) && \
	(defined(__GNUC__) || defined(__clang__))
#define TQ_GRAPH_AVX512_WEIGHTED_TARGET __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx2")))
#else
#define TQ_GRAPH_AVX512_WEIGHTED_TARGET
#endif

#if TQ_GRAPH_COMPILE_AVXVNNI && !defined(__AVXVNNI__) && (defined(__GNUC__) || defined(__clang__))
#define TQ_GRAPH_AVXVNNI_TARGET __attribute__((target("avxvnni,avx2")))
#else
#define TQ_GRAPH_AVXVNNI_TARGET
#endif

#if TQ_GRAPH_COMPILE_AVX2 || TQ_GRAPH_COMPILE_AVX512VNNI || \
	TQ_GRAPH_COMPILE_AVXVNNI || TQ_GRAPH_COMPILE_AVX512_WEIGHTED
#include <immintrin.h>
#endif

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#include "utils/fmgrprotos.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"

#include "tqgraph_score.h"

#if !defined(TQ_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
#if defined(__clang__)
#define TQ_GRAPH_ARM_DOT_TARGET __attribute__((target("dotprod")))
#define TQ_GRAPH_COMPILE_ARM_DOT 1
#define TQ_GRAPH_ARM_I8MM_TARGET __attribute__((target("i8mm")))
#define TQ_GRAPH_COMPILE_ARM_I8MM 1
#elif defined(__GNUC__)
#define TQ_GRAPH_ARM_DOT_TARGET __attribute__((target("+dotprod")))
#define TQ_GRAPH_COMPILE_ARM_DOT 1
#define TQ_GRAPH_ARM_I8MM_TARGET __attribute__((target("+i8mm")))
#define TQ_GRAPH_COMPILE_ARM_I8MM 1
#else
#define TQ_GRAPH_ARM_DOT_TARGET
#define TQ_GRAPH_COMPILE_ARM_DOT 0
#define TQ_GRAPH_ARM_I8MM_TARGET
#define TQ_GRAPH_COMPILE_ARM_I8MM 0
#endif
static const int8 TqGraphCodebookI8[TQ_LUT_WIDTH] = {
	-127, -96, -75, -58, -44, -31, -18, -6,
	6, 18, 31, 44, 58, 75, 96, 127
};
static const int8 TqGraphCodebook2I8[TQ_LUT_WIDTH] = {
	-127, -38, 38, 127, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0
};
#else
#define TQ_GRAPH_COMPILE_ARM_DOT 0
#define TQ_GRAPH_COMPILE_ARM_I8MM 0
#endif

#if TQ_GRAPH_COMPILE_AVX2 && !TQ_GRAPH_COMPILE_ARM_DOT
static const int8 TqGraphCodebookI8[TQ_LUT_WIDTH] = {
	-127, -96, -75, -58, -44, -31, -18, -6,
	6, 18, 31, 44, 58, 75, 96, 127
};
pg_attribute_unused()
static const int8 TqGraphCodebook2I8[TQ_LUT_WIDTH] = {
	-127, -38, 38, 127, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0
};
static const int8 TqGraphCodebook2PairEvenI8[16] = {
	-127, -38, 38, 127,
	-127, -38, 38, 127,
	-127, -38, 38, 127,
	-127, -38, 38, 127,
};
static const int8 TqGraphCodebook2PairOddI8[16] = {
	-127, -127, -127, -127,
	-38, -38, -38, -38,
	38, 38, 38, 38,
	127, 127, 127, 127,
};
#endif

#if TQ_GRAPH_COMPILE_ARM_DOT || TQ_GRAPH_COMPILE_AVX2 || \
	TQ_GRAPH_COMPILE_AVX512VNNI || TQ_GRAPH_COMPILE_AVXVNNI || \
	TQ_GRAPH_COMPILE_AVX512_WEIGHTED
#define TQ_GRAPH_COMPILE_QUERY_SPLIT 1
#else
#define TQ_GRAPH_COMPILE_QUERY_SPLIT 0
#endif


static inline float
TqGraphCodeCenter(int code, int bits)
{
	return TqGetCodeCenterBits(code, bits);
}

double
TqGraphExactDistance(HnswSupport *support, Datum a, Datum b)
{
	return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));
}

TqScoreMode
TqGraphGetScoreMode(HnswSupport *support)
{
	char	   *procname = get_func_name(support->procinfo->fn_oid);
	TqScoreMode mode = TQ_SCORE_L2;

	if (procname == NULL)
		return mode;

	if (strcmp(procname, "vector_negative_inner_product") == 0)
		mode = TQ_SCORE_IP;
	else if (strcmp(procname, "cosine_distance") == 0)
		mode = TQ_SCORE_COSINE;
	else if (strcmp(procname, "l1_distance") == 0)
		mode = TQ_SCORE_L1;

	pfree(procname);
	return mode;
}

float
TqGraphVectorNorm(Vector *vector)
{
	double		norm = 0;

	for (int i = 0; i < vector->dim; i++)
		norm += (double) vector->x[i] * vector->x[i];

	return (float) sqrt(norm);
}

float
TqGraphCodeNorm(const uint8 *code, int dimensions, int bits)
{
	float		norm = 0;

	for (int i = 0; i < dimensions; i++)
	{
		int			component = TqGetCodeComponentBits(code, i, bits);
		float		center = TqGraphCodeCenter(component, bits);

		norm += center * center;
	}

	return norm;
}


/*
 * Scalar reference for the TurboQuant+ symmetric (code-code) raw weighted
 * inner product, mirroring qdrant's score_symmetric_ec:
 *
 *     raw = Σ TqGraphCodeCenter(a[d]) · TqGraphCodeCenter(b[d]) · D'²[d]
 *
 * where D'²[d] = 1 / ecScale[d]², matching qdrant's convention
 * (ec.scale[d] is the per-coord *multiplier* applied during encoding —
 * `(rotated + shift) * scale` — and D' = 1/scale undoes it during
 * scoring).  The point of the weighting is exactly to cancel the
 * scaling done at encode time, so the raw weighted dot reconstructs
 * ⟨rotated_a + shift, rotated_b + shift⟩ in the original space (modulo
 * quantization noise).
 *
 * The full TQ+ score is then: raw + xm_a + xm_b - mm_const, with
 *   xm = ⟨rotated, -ecShift⟩ (computed at encode time inside
 *        TqEncodeVectorInternal; persisted per vector)
 *   mm_const = Σ ecShift[d]² (TqGraphMmConstScalar).
 *
 * Returns 0 when ecScale is NULL (legacy index, no weighted scoring).
 */
double
TqGraphCodeCodeWeightedRawScalar(const uint8 *a, const uint8 *b,
								  int dimensions, int bits,
								  const float *ecScale)
{
	double		acc = 0.0;

	if (ecScale == NULL)
		return 0.0;

	for (int i = 0; i < dimensions; i++)
	{
		int			ca = TqGetCodeComponentBits(a, i, bits);
		int			cb = TqGetCodeComponentBits(b, i, bits);
		double		centerA = (double) TqGraphCodeCenter(ca, bits);
		double		centerB = (double) TqGraphCodeCenter(cb, bits);
		double		s = (double) ecScale[i];
		double		w;

		if (fabs(s) <= FLT_EPSILON)
			continue;

		w = 1.0 / (s * s);
		acc += centerA * centerB * w;
	}

	return acc;
}

/*
 * Σ ecShift[d]² — the TQ+ "mm_const" scalar.  Computed once per scan
 * setup (or per build distance call) and reused.  Returns 0 when
 * ecShift is NULL.
 */
double
TqGraphMmConstScalar(const float *ecShift, int dimensions)
{
	double		acc = 0.0;

	if (ecShift == NULL)
		return 0.0;

	for (int i = 0; i < dimensions; i++)
		acc += (double) ecShift[i] * (double) ecShift[i];

	return acc;
}

float
TqGraphEncodeVector(TqGraphBuildState *state, Vector *vector, uint8 *code)
{
	if (state != NULL && state->ecShift != NULL && state->ecScale != NULL)
		return TqEncodeVectorWithCorrectionBits(vector, code, state->tqBits,
												state->ecShift, state->ecScale);

	return TqEncodeVectorBits(vector, code, state != NULL ? state->tqBits : TQ_DEFAULT_BITS);
}

/*
 * TurboQuant+ encode helper: emits both the code AND the qdrant-compatible
 * xm = ⟨rotated, -ecShift⟩ in one pass (no double rotation cost).  Used
 * by the build/insert paths when state->tqWeighted is set.
 *
 * Falls back to plain encoding (xm := 0) when ecShift/ecScale are not
 * available.
 */
float
TqGraphEncodeVectorWithXm(TqGraphBuildState *state, Vector *vector,
						   uint8 *code, float *xmOut)
{
	if (state != NULL && state->ecShift != NULL && state->ecScale != NULL)
		return TqEncodeVectorWithCorrectionAndXmBits(vector, code, state->tqBits,
													  state->ecShift, state->ecScale,
													  xmOut);

	if (xmOut != NULL)
		*xmOut = 0.0f;
	return TqEncodeVectorBits(vector, code, state != NULL ? state->tqBits : TQ_DEFAULT_BITS);
}

/*
 * Renormalizing encode helper: like TqGraphEncodeVectorWithXm but also returns
 * the renormalized scaling factor.  When ecShift/ecScale are present, the
 * return value is `length · sqrt(d) / centroid_norm` so existing scorers
 * that compute `node->scale · dot / dimSqrt` yield qdrant-style
 * `length · dot / centroid_norm` — the per-vector renormalization fold.
 *
 * `centroid_norm` is measured in EC-reverted (rescaled-pre-EC) space and
 * tracks ‖decoded_quantized‖.  For perfect quantization centroid_norm
 * equals sqrt(d) and the renorm is a no-op; for lossy quantization the
 * drift between centroid_norm and sqrt(d) is what the correction folds
 * back into scoring.
 *
 * When ecShift/ecScale are absent (L2 builds, plain quantization), the
 * renorm is not applicable and the helper falls back to plain length.
 * This matches qdrant: L2 stores `scaling_factor = l2_length` directly,
 * Dot/Cosine store `l2_length / centroid_norm`.
 */
float
TqGraphEncodeVectorWithXmRenorm(TqGraphBuildState *state, Vector *vector,
								uint8 *code, float *xmOut)
{
	float		length;
	float		centroidNorm;
	double		dimSqrt;

	if (state == NULL || state->ecShift == NULL || state->ecScale == NULL)
	{
		if (xmOut != NULL)
			*xmOut = 0.0f;
		return TqEncodeVectorBits(vector, code,
								  state != NULL ? state->tqBits : TQ_DEFAULT_BITS);
	}

	length = TqEncodeVectorWithCorrectionXmRenormBits(vector, code, state->tqBits,
													  state->ecShift, state->ecScale,
													  xmOut, &centroidNorm);

	if (length == 0.0f || centroidNorm <= 0.0f)
		return length;

	dimSqrt = sqrt((double) vector->dim);
	return (float) ((double) length * dimSqrt / (double) centroidNorm);
}

#if (defined(__aarch64__) || defined(_M_ARM64)) && TQ_GRAPH_ENABLE_SYMMETRIC_I8_DOT
static inline int32 TqGraphInt8Int8DotNeon(const int8 *query,
										   const int8 *components, int dim);
static inline void TqGraphInt8Int8Dot4Neon(const int8 *query,
										   const uint8 **valueCodes, int dim,
										   int32 *dots);
#endif
#if TQ_GRAPH_COMPILE_ARM_DOT
static bool TqGraphArmDotprodAvailable(void);
#if TQ_GRAPH_COMPILE_ARM_I8MM
static bool TqGraphArmI8mmAvailable(void);
static int32x4_t TQ_GRAPH_ARM_I8MM_TARGET
TqGraphDotI8x16ArmI8mm(int32x4_t acc, int8x16_t a, int8x16_t b);
#endif
static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphQuerySplitRawNeonSdot(const HnswTqQuery *tq, const uint8 *code);
static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphQuerySplit2RawNeonSdot(const HnswTqQuery *tq, const uint8 *code);
static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphCodeCodeRawNeonSdot(const uint8 *a, const uint8 *b, int dim,
						   int *sampleDims);
static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphCodeCode2RawNeonSdot(const uint8 *a, const uint8 *b, int dim,
							int *sampleDims);
static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphCodeCodeWeightedRawNeonSdot(const uint8 *a, const uint8 *b,
								   const int16 *weights, int dim);
static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphCodeCode2WeightedRawNeonSdot(const uint8 *a, const uint8 *b,
									const int16 *weights, int dim);
#endif
#if TQ_GRAPH_COMPILE_AVX2
static bool TqGraphAvx2Available(void);
static int64 TQ_GRAPH_AVX2_TARGET
TqGraphQuerySplitRawAvx2(const HnswTqQuery *tq, const uint8 *code);
static int64 TQ_GRAPH_AVX2_TARGET
TqGraphQuerySplit2RawAvx2(const HnswTqQuery *tq, const uint8 *code);
static int64 TQ_GRAPH_AVX2_TARGET
TqGraphCodeCodeRawAvx2(const uint8 *a, const uint8 *b, int dim,
					   int *sampleDims);
static int64 TQ_GRAPH_AVX2_TARGET
TqGraphCodeCode2RawAvx2(const uint8 *a, const uint8 *b, int dim,
						int *sampleDims);
static int64 TQ_GRAPH_AVX2_TARGET
TqGraphCodeCodeWeightedRawAvx2(const uint8 *a, const uint8 *b,
								const int16 *weights, int dim);
static int64 TQ_GRAPH_AVX2_TARGET
TqGraphCodeCode2WeightedRawAvx2(const uint8 *a, const uint8 *b,
								 const int16 *weights, int dim);
#endif
#if TQ_GRAPH_COMPILE_AVX512VNNI
static bool TqGraphAvx512VnniAvailable(void);
static int64 TQ_GRAPH_AVX512VNNI_TARGET
TqGraphQuerySplitRawAvx512Vnni(const HnswTqQuery *tq, const uint8 *code);
static int64 TQ_GRAPH_AVX512VNNI_TARGET
TqGraphQuerySplit2RawAvx512Vnni(const HnswTqQuery *tq, const uint8 *code);
static int64 TQ_GRAPH_AVX512VNNI_TARGET
TqGraphCodeCodeRawAvx512Vnni(const uint8 *a, const uint8 *b, int dim,
							 int *sampleDims);
static int64 TQ_GRAPH_AVX512VNNI_TARGET
TqGraphCodeCode2RawAvx512Vnni(const uint8 *a, const uint8 *b, int dim,
							  int *sampleDims);
#endif
#if TQ_GRAPH_COMPILE_AVX512_WEIGHTED
static bool TqGraphAvx512WeightedAvailable(void);
static int64 TQ_GRAPH_AVX512_WEIGHTED_TARGET
TqGraphCodeCodeWeightedRawAvx512(const uint8 *a, const uint8 *b,
								 const int16 *weights, int dim);
static int64 TQ_GRAPH_AVX512_WEIGHTED_TARGET
TqGraphCodeCode2WeightedRawAvx512(const uint8 *a, const uint8 *b,
								  const int16 *weights, int dim);
#endif
#if TQ_GRAPH_COMPILE_AVXVNNI
static bool TqGraphAvxVnniAvailable(void);
static int64 TQ_GRAPH_AVXVNNI_TARGET
TqGraphQuerySplitRawAvxVnni(const HnswTqQuery *tq, const uint8 *code);
static int64 TQ_GRAPH_AVXVNNI_TARGET
TqGraphQuerySplit2RawAvxVnni(const HnswTqQuery *tq, const uint8 *code);
static int64 TQ_GRAPH_AVXVNNI_TARGET
TqGraphCodeCodeRawAvxVnni(const uint8 *a, const uint8 *b, int dim,
						  int *sampleDims);
static int64 TQ_GRAPH_AVXVNNI_TARGET
TqGraphCodeCode2RawAvxVnni(const uint8 *a, const uint8 *b, int dim,
						   int *sampleDims);
#endif
#if TQ_GRAPH_ENABLE_SYMMETRIC_I8_DOT
static inline int32 TqGraphInt8Int8DotScalar(const int8 *query,
											 const int8 *components, int dim);
#endif

static double
TqGraphBuildCodeCodeRawScalarRange(const uint8 *a, const uint8 *b,
								   int startDim, int dimCount, int bits);

static double
TqGraphPackedDistance(const HnswTqQuery *tq, const uint8 *valueCode,
					  float valueScale, float valueCodeNorm, float valueNorm)
{
	(void) valueCodeNorm;
	(void) valueNorm;

	return TqCodeDistance(tq, valueCode, valueScale);
}

#if TQ_GRAPH_COMPILE_QUERY_SPLIT
static bool
TqGraphPackedDistanceQuerySplit4(const HnswTqQuery *tq, const uint8 *valueCode,
								 float valueScale, double *distance)
{
	double		dimSqrt;
	double		dot;
	int64		rawDot;
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;

	if (!tq->querySplitEnabled || tq->dimensions < 1024 ||
		tq->bits != TQ_DEFAULT_BITS || mode == TQ_SCORE_L1)
		return false;

#if TQ_GRAPH_COMPILE_ARM_DOT
	if (TqGraphArmDotprodAvailable())
		rawDot = TqGraphQuerySplitRawNeonSdot(tq, valueCode);
	else
#endif
#if TQ_GRAPH_COMPILE_AVX512VNNI
	if (TqGraphAvx512VnniAvailable())
		rawDot = TqGraphQuerySplitRawAvx512Vnni(tq, valueCode);
	else
#endif
#if TQ_GRAPH_COMPILE_AVXVNNI
	if (TqGraphAvxVnniAvailable())
		rawDot = TqGraphQuerySplitRawAvxVnni(tq, valueCode);
	else
#endif
#if TQ_GRAPH_COMPILE_AVX2
	if (TqGraphAvx2Available())
		rawDot = TqGraphQuerySplitRawAvx2(tq, valueCode);
	else
#endif
		return false;

	dimSqrt = sqrt((double) tq->dimensions);
	dot = (double) tq->querySplitPostprocessScale *
		(double) rawDot;
	dot += tq->ecCorrection;

	if (mode == TQ_SCORE_IP)
		*distance = -(valueScale * dot / dimSqrt);
	else if (mode == TQ_SCORE_COSINE)
	{
		if (tq->queryNorm == 0 || valueScale == 0)
			*distance = 1;
		else
			*distance = 1 - (dot / (sqrt(tq->queryNorm) * dimSqrt));
	}
	else
	{
		*distance = tq->queryNorm + ((double) valueScale * valueScale) -
			(2 * valueScale * dot / dimSqrt);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

static bool
TqGraphPackedDistanceQuerySplit2(const HnswTqQuery *tq, const uint8 *valueCode,
								 float valueScale, double *distance)
{
	double		dimSqrt;
	double		dot;
	int64		rawDot;
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;

	if (!tq->querySplitEnabled || tq->dimensions < 1024 ||
		tq->bits != 2 || mode == TQ_SCORE_L1)
		return false;

#if TQ_GRAPH_COMPILE_ARM_DOT
	if (TqGraphArmDotprodAvailable())
		rawDot = TqGraphQuerySplit2RawNeonSdot(tq, valueCode);
	else
#endif
#if TQ_GRAPH_COMPILE_AVX512VNNI
	if (TqGraphAvx512VnniAvailable())
		rawDot = TqGraphQuerySplit2RawAvx512Vnni(tq, valueCode);
	else
#endif
#if TQ_GRAPH_COMPILE_AVXVNNI
	if (TqGraphAvxVnniAvailable())
		rawDot = TqGraphQuerySplit2RawAvxVnni(tq, valueCode);
	else
#endif
#if TQ_GRAPH_COMPILE_AVX2
	if (TqGraphAvx2Available())
		rawDot = TqGraphQuerySplit2RawAvx2(tq, valueCode);
	else
#endif
		return false;

	dimSqrt = sqrt((double) tq->dimensions);
	dot = (double) tq->querySplitPostprocessScale *
		(double) rawDot;
	dot += tq->ecCorrection;

	if (mode == TQ_SCORE_IP)
		*distance = -(valueScale * dot / dimSqrt);
	else if (mode == TQ_SCORE_COSINE)
	{
		if (tq->queryNorm == 0 || valueScale == 0)
			*distance = 1;
		else
			*distance = 1 - (dot / (sqrt(tq->queryNorm) * dimSqrt));
	}
	else
	{
		*distance = tq->queryNorm + ((double) valueScale * valueScale) -
			(2 * valueScale * dot / dimSqrt);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

static bool
TqGraphBuildCodeCodeDistance4(TqGraphBuildState *state, uint32 a, uint32 b,
							  double *distance)
{
	TqGraphBuildNode *aNode;
	TqGraphBuildNode *bNode;
	double		dot;
	double		scale;
	double		codebookScaleSq;
	int			simdDimensions;
	int			tailDims;
	int			sampleDims;
	int64		rawDot;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if ((mode != TQ_SCORE_L2 && mode != TQ_SCORE_COSINE && mode != TQ_SCORE_IP) ||
		state->tqBits != TQ_DEFAULT_BITS)
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	tailDims = state->dimensions % 16;
	simdDimensions = state->dimensions - tailDims;

#if TQ_GRAPH_COMPILE_ARM_DOT
	if (TqGraphArmDotprodAvailable())
		rawDot = TqGraphCodeCodeRawNeonSdot(aNode->code, bNode->code,
											simdDimensions, &sampleDims);
	else
#endif
#if TQ_GRAPH_COMPILE_AVX512VNNI
	if (TqGraphAvx512VnniAvailable())
		rawDot = TqGraphCodeCodeRawAvx512Vnni(aNode->code, bNode->code,
											  simdDimensions, &sampleDims);
	else
#endif
#if TQ_GRAPH_COMPILE_AVXVNNI
	if (TqGraphAvxVnniAvailable())
		rawDot = TqGraphCodeCodeRawAvxVnni(aNode->code, bNode->code,
										   simdDimensions, &sampleDims);
	else
#endif
#if TQ_GRAPH_COMPILE_AVX2
	if (TqGraphAvx2Available())
		rawDot = TqGraphCodeCodeRawAvx2(aNode->code, bNode->code,
										simdDimensions, &sampleDims);
	else
#endif
		return false;

	codebookScaleSq = TQ_GRAPH_CODEBOOK_SCALE * TQ_GRAPH_CODEBOOK_SCALE;
	dot = (double) rawDot / codebookScaleSq;
	if (tailDims != 0 && sampleDims == simdDimensions)
	{
		dot += TqGraphBuildCodeCodeRawScalarRange(aNode->code, bNode->code,
												  simdDimensions, tailDims,
												  state->tqBits);
		sampleDims += tailDims;
	}
	if (sampleDims <= 0)
		return false;

	scale = (double) sampleDims;
	if (mode == TQ_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale *
					  dot / scale);
	else if (mode == TQ_SCORE_COSINE)
		*distance = 1 - (dot / scale);
	else
	{
		*distance = ((double) aNode->scale * aNode->scale) +
			((double) bNode->scale * bNode->scale) -
			(2 * (double) aNode->scale * (double) bNode->scale * dot / scale);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

static bool
TqGraphBuildCodeCodeDistance2(TqGraphBuildState *state, uint32 a, uint32 b,
							  double *distance)
{
	TqGraphBuildNode *aNode;
	TqGraphBuildNode *bNode;
	double		dot;
	double		scale;
	double		codebookScaleSq;
	int			simdDimensions;
	int			tailDims;
	int			sampleDims;
	int64		rawDot;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if ((mode != TQ_SCORE_L2 && mode != TQ_SCORE_COSINE && mode != TQ_SCORE_IP) ||
		state->tqBits != 2)
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	tailDims = state->dimensions % 16;
	simdDimensions = state->dimensions - tailDims;

#if TQ_GRAPH_COMPILE_ARM_DOT
	if (TqGraphArmDotprodAvailable())
		rawDot = TqGraphCodeCode2RawNeonSdot(aNode->code, bNode->code,
											 simdDimensions, &sampleDims);
	else
#endif
#if TQ_GRAPH_COMPILE_AVX512VNNI
	if (TqGraphAvx512VnniAvailable())
		rawDot = TqGraphCodeCode2RawAvx512Vnni(aNode->code, bNode->code,
											   simdDimensions, &sampleDims);
	else
#endif
#if TQ_GRAPH_COMPILE_AVXVNNI
	if (TqGraphAvxVnniAvailable())
		rawDot = TqGraphCodeCode2RawAvxVnni(aNode->code, bNode->code,
											simdDimensions, &sampleDims);
	else
#endif
#if TQ_GRAPH_COMPILE_AVX2
	if (TqGraphAvx2Available())
		rawDot = TqGraphCodeCode2RawAvx2(aNode->code, bNode->code,
										 simdDimensions, &sampleDims);
	else
#endif
		return false;

	codebookScaleSq = TQ_GRAPH_CODEBOOK2_SCALE * TQ_GRAPH_CODEBOOK2_SCALE;
	dot = (double) rawDot / codebookScaleSq;
	if (tailDims != 0 && sampleDims == simdDimensions)
	{
		dot += TqGraphBuildCodeCodeRawScalarRange(aNode->code, bNode->code,
												  simdDimensions, tailDims,
												  state->tqBits);
		sampleDims += tailDims;
	}
	if (sampleDims <= 0)
		return false;

	scale = (double) sampleDims;
	if (mode == TQ_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale *
					  dot / scale);
	else if (mode == TQ_SCORE_COSINE)
		*distance = 1 - (dot / scale);
	else
	{
		*distance = ((double) aNode->scale * aNode->scale) +
			((double) bNode->scale * bNode->scale) -
			(2 * (double) aNode->scale * (double) bNode->scale * dot / scale);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}
#endif

static double
TqGraphBuildCodeCodeRawScalarRange(const uint8 *a, const uint8 *b,
								   int startDim, int dimCount, int bits)
{
	double		raw = 0.0;

	for (int dimIdx = startDim; dimIdx < startDim + dimCount; dimIdx++)
	{
		int			ca = TqGetCodeComponentBits(a, dimIdx, bits);
		int			cb = TqGetCodeComponentBits(b, dimIdx, bits);

		raw += (double) TqGraphCodeCenter(ca, bits) *
			(double) TqGraphCodeCenter(cb, bits);
	}

	return raw;
}

static double
TqGraphBuildCodeCodeRawScalar(const uint8 *a, const uint8 *b, int dim,
							  int bits, int *sampleDims)
{
	double		raw = 0.0;
	int			chunkDims = 16;
	int			chunks = dim / chunkDims;
	int			tailDims = dim - chunks * chunkDims;
	int			scoredChunks = chunks;

	if (chunks > TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * chunkDims;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		int			start = chunk * chunkDims;

		raw += TqGraphBuildCodeCodeRawScalarRange(a, b, start, chunkDims,
												  bits);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		int			start = chunks * chunkDims;

		raw += TqGraphBuildCodeCodeRawScalarRange(a, b, start, tailDims,
												  bits);
		*sampleDims += tailDims;
	}

	return raw;
}

static bool
TqGraphBuildCodeCodeDistanceScalar(TqGraphBuildState *state, uint32 a, uint32 b,
								   double *distance)
{
	TqGraphBuildNode *aNode;
	TqGraphBuildNode *bNode;
	double		dot;
	double		scale;
	int			sampleDims;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if ((mode != TQ_SCORE_L2 && mode != TQ_SCORE_COSINE && mode != TQ_SCORE_IP) ||
		(state->tqBits != TQ_DEFAULT_BITS && state->tqBits != 2))
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	dot = TqGraphBuildCodeCodeRawScalar(aNode->code, bNode->code,
										state->dimensions, state->tqBits,
										&sampleDims);
	if (sampleDims <= 0)
		return false;

	scale = (double) sampleDims;
	if (mode == TQ_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale *
					  dot / scale);
	else if (mode == TQ_SCORE_COSINE)
		*distance = 1 - (dot / scale);
	else
	{
		*distance = ((double) aNode->scale * aNode->scale) +
			((double) bNode->scale * bNode->scale) -
			(2 * (double) aNode->scale * (double) bNode->scale * dot / scale);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

/*
 * TurboQuant+ symmetric (code-code) build distance.
 *
 * The qdrant TQ+ formula reconstructs ⟨rotated_a, rotated_b⟩ from the
 * quantized representation:
 *
 *    sim = (Σ X+_a · X+_b · D'²) + xm_a + xm_b - mm_const
 *        = ⟨rotated_a, rotated_b⟩    (modulo quantization noise)
 *
 * TqPreprocessVector rescales each rotated vector to length √dim, so
 * ⟨rot_a, rot_b⟩ / dim = cos(angle).
 *
 * pgvector's cosine_ops opclass uses vector_negative_inner_product as
 * FUNCTION 1, so its scoreMode is TQ_SCORE_IP — the "cosine-via-IP"
 * convention of the rest of the codebase.  We mirror the existing
 * unweighted scorer's distance shape so the value is comparable across
 * legacy and weighted paths within the same build:
 *    IP:     dist = -aScale · bScale · sim / dim
 *    COSINE: dist = 1 - sim / dim
 *
 * Activates only when state->tqWeighted is set, ecShift+ecScale are
 * present, the metric is IP or COSINE, the user-set hnsw.tq_weighted
 * GUC is on (kill-switch), and the bit width is 2 or 4.  Falls
 * through to the unweighted scorer otherwise.
 *
 * No SIMD yet — this is the scalar reference.  Slices ζ-θ wire the
 * VNNI / NEON kernels.  Build is not the perf hot path so the scalar
 * version is acceptable for now.
 */
static bool
TqGraphBuildCodeCodeWeighted(TqGraphBuildState *state, uint32 a, uint32 b,
							  double *distance)
{
	TqGraphBuildNode *aNode;
	TqGraphBuildNode *bNode;
	double		raw;
	double		sim;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if (!state->tqWeighted ||
		(mode != TQ_SCORE_COSINE && mode != TQ_SCORE_IP) ||
		state->ecShift == NULL || state->ecScale == NULL ||
		!hnsw_tq_weighted)
		return false;

	if (state->tqBits != TQ_DEFAULT_BITS && state->tqBits != 2)
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	raw = 0.0;
	if (state->dPrimeSqI16 != NULL && state->weightScale > FLT_EPSILON)
	{
		int64		rawI64 = 0;
		double		codebookScaleSq;
		bool		simdRan = false;

		if (state->tqBits == TQ_DEFAULT_BITS)
			codebookScaleSq = TQ_GRAPH_CODEBOOK_SCALE * TQ_GRAPH_CODEBOOK_SCALE;
		else
			codebookScaleSq = TQ_GRAPH_CODEBOOK2_SCALE * TQ_GRAPH_CODEBOOK2_SCALE;

#if TQ_GRAPH_COMPILE_AVX512_WEIGHTED
		if (!simdRan && TqGraphAvx512WeightedAvailable())
		{
			if (state->tqBits == TQ_DEFAULT_BITS)
				rawI64 = TqGraphCodeCodeWeightedRawAvx512(aNode->code, bNode->code,
														  state->dPrimeSqI16,
														  state->dimensions);
			else
				rawI64 = TqGraphCodeCode2WeightedRawAvx512(aNode->code, bNode->code,
														   state->dPrimeSqI16,
														   state->dimensions);
			HnswRecordWeightedCodeCodeKernel(TQ_SCORING_AVX512BW_DQ);
			simdRan = true;
		}
#endif
#if TQ_GRAPH_COMPILE_ARM_DOT
		if (!simdRan && TqGraphArmDotprodAvailable())
		{
			if (state->tqBits == TQ_DEFAULT_BITS)
				rawI64 = TqGraphCodeCodeWeightedRawNeonSdot(aNode->code, bNode->code,
															 state->dPrimeSqI16,
															 state->dimensions);
			else
				rawI64 = TqGraphCodeCode2WeightedRawNeonSdot(aNode->code, bNode->code,
															  state->dPrimeSqI16,
															  state->dimensions);
			HnswRecordWeightedCodeCodeKernel(TQ_SCORING_NEON);
			simdRan = true;
		}
#endif
#if TQ_GRAPH_COMPILE_AVX2
		if (!simdRan && TqGraphAvx2Available())
		{
			if (state->tqBits == TQ_DEFAULT_BITS)
				rawI64 = TqGraphCodeCodeWeightedRawAvx2(aNode->code, bNode->code,
														 state->dPrimeSqI16,
														 state->dimensions);
			else
				rawI64 = TqGraphCodeCode2WeightedRawAvx2(aNode->code, bNode->code,
														  state->dPrimeSqI16,
														  state->dimensions);
			HnswRecordWeightedCodeCodeKernel(TQ_SCORING_AVX2);
			simdRan = true;
		}
#endif

		if (simdRan)
			raw = (double) rawI64 / ((double) state->weightScale * codebookScaleSq);
		else
		{
			HnswRecordWeightedCodeCodeKernel(TQ_SCORING_SCALAR);
			raw = TqGraphCodeCodeWeightedRawScalar(aNode->code, bNode->code,
												   state->dimensions, state->tqBits,
												   state->ecScale);
		}
	}
	else
	{
		HnswRecordWeightedCodeCodeKernel(TQ_SCORING_SCALAR);
		raw = TqGraphCodeCodeWeightedRawScalar(aNode->code, bNode->code,
											   state->dimensions, state->tqBits,
											   state->ecScale);
	}

	sim = raw + (double) aNode->ecCorrection + (double) bNode->ecCorrection -
		state->mmConst;

	if (mode == TQ_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale *
					  sim / (double) state->dimensions);
	else						/* TQ_SCORE_COSINE */
		*distance = 1.0 - sim / (double) state->dimensions;

	return true;
}

#if TQ_GRAPH_ENABLE_SYMMETRIC_I8_DOT
static inline int32
TqGraphInt8Int8DotScalar(const int8 *query, const int8 *components, int dim)
{
	int32		dot = 0;

	for (int i = 0; i < dim; i++)
		dot += (int32) query[i] * (int32) components[i];

	return dot;
}
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if TQ_GRAPH_ENABLE_SYMMETRIC_I8_DOT
static inline int32
TqGraphInt8Int8DotNeon(const int8 *query, const int8 *components, int dim)
{
	int32		dot;
	int			i = 0;
#if defined(__ARM_FEATURE_DOTPROD)
	int32x4_t	acc = vdupq_n_s32(0);

	for (; i + 16 <= dim; i += 16)
	{
		int8x16_t	q = vld1q_s8(query + i);
		int8x16_t	c = vld1q_s8(components + i);

		acc = vdotq_s32(acc, q, c);
	}

	dot = vaddvq_s32(acc);
#else
	int32x4_t	acc0 = vdupq_n_s32(0);
	int32x4_t	acc1 = vdupq_n_s32(0);

	for (; i + 16 <= dim; i += 16)
	{
		int16x8_t	lo = vmull_s8(vld1_s8(query + i),
								   vld1_s8(components + i));
		int16x8_t	hi = vmull_s8(vld1_s8(query + i + 8),
								   vld1_s8(components + i + 8));

		acc0 = vpadalq_s16(acc0, lo);
		acc1 = vpadalq_s16(acc1, hi);
	}

	acc0 = vaddq_s32(acc0, acc1);
	dot = vaddvq_s32(acc0);
#endif
	for (; i < dim; i++)
		dot += (int32) query[i] * (int32) components[i];

	return dot;
}

static inline void
TqGraphInt8Int8Dot4Neon(const int8 *query, const uint8 **valueCodes, int dim,
						int32 *dots)
{
	int			i = 0;
	const int8 *c0 = (const int8 *) valueCodes[0];
	const int8 *c1 = (const int8 *) valueCodes[1];
	const int8 *c2 = (const int8 *) valueCodes[2];
	const int8 *c3 = (const int8 *) valueCodes[3];
#if defined(__ARM_FEATURE_DOTPROD)
	int32x4_t	acc0 = vdupq_n_s32(0);
	int32x4_t	acc1 = vdupq_n_s32(0);
	int32x4_t	acc2 = vdupq_n_s32(0);
	int32x4_t	acc3 = vdupq_n_s32(0);

	for (; i + 16 <= dim; i += 16)
	{
		int8x16_t	q = vld1q_s8(query + i);

		acc0 = vdotq_s32(acc0, q, vld1q_s8(c0 + i));
		acc1 = vdotq_s32(acc1, q, vld1q_s8(c1 + i));
		acc2 = vdotq_s32(acc2, q, vld1q_s8(c2 + i));
		acc3 = vdotq_s32(acc3, q, vld1q_s8(c3 + i));
	}

	dots[0] = vaddvq_s32(acc0);
	dots[1] = vaddvq_s32(acc1);
	dots[2] = vaddvq_s32(acc2);
	dots[3] = vaddvq_s32(acc3);
#else
	dots[0] = dots[1] = dots[2] = dots[3] = 0;
	for (; i + 16 <= dim; i += 16)
	{
		int8x8_t	qlo = vld1_s8(query + i);
		int8x8_t	qhi = vld1_s8(query + i + 8);
		int16x8_t	m0lo = vmull_s8(qlo, vld1_s8(c0 + i));
		int16x8_t	m0hi = vmull_s8(qhi, vld1_s8(c0 + i + 8));
		int16x8_t	m1lo = vmull_s8(qlo, vld1_s8(c1 + i));
		int16x8_t	m1hi = vmull_s8(qhi, vld1_s8(c1 + i + 8));
		int16x8_t	m2lo = vmull_s8(qlo, vld1_s8(c2 + i));
		int16x8_t	m2hi = vmull_s8(qhi, vld1_s8(c2 + i + 8));
		int16x8_t	m3lo = vmull_s8(qlo, vld1_s8(c3 + i));
		int16x8_t	m3hi = vmull_s8(qhi, vld1_s8(c3 + i + 8));

		dots[0] += vaddvq_s16(m0lo) + vaddvq_s16(m0hi);
		dots[1] += vaddvq_s16(m1lo) + vaddvq_s16(m1hi);
		dots[2] += vaddvq_s16(m2lo) + vaddvq_s16(m2hi);
		dots[3] += vaddvq_s16(m3lo) + vaddvq_s16(m3hi);
	}
#endif
	for (; i < dim; i++)
	{
		int32		q = query[i];

		dots[0] += q * (int32) c0[i];
		dots[1] += q * (int32) c1[i];
		dots[2] += q * (int32) c2[i];
		dots[3] += q * (int32) c3[i];
	}
}
#endif

#endif

#if TQ_GRAPH_COMPILE_AVX2
static bool
TqGraphAvx2Available(void)
{
	static int	available = -1;

	if (hnsw_tq_simd_force != TQ_SIMD_FORCE_AUTO &&
		hnsw_tq_simd_force != TQ_SIMD_FORCE_AVX2)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVX2__)
	available = 1;
#elif TQ_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))
	available = __builtin_cpu_supports("avx2") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}

static inline int64 TQ_GRAPH_AVX2_TARGET
TqGraphHorizontalSumI32Avx2(__m256i v)
{
	int32		s[8];

	_mm256_storeu_si256((__m256i *) s, v);
	return (int64) s[0] + s[1] + s[2] + s[3] +
		s[4] + s[5] + s[6] + s[7];
}

static inline double TQ_GRAPH_AVX2_TARGET
TqGraphHorizontalSumF32Avx2(__m256 v)
{
	float		s[8];

	_mm256_storeu_ps(s, v);
	return (double) s[0] + s[1] + s[2] + s[3] +
		s[4] + s[5] + s[6] + s[7];
}

static inline __m128i TQ_GRAPH_AVX2_TARGET
TqGraphExpandPacked4Avx2(const uint8 *code)
{
	const __m128i mask = _mm_set1_epi8(0x0f);
	const __m128i codebook = _mm_loadu_si128((const __m128i *) TqGraphCodebookI8);
	__m128i	packed = _mm_loadl_epi64((const __m128i *) code);
	__m128i	lo = _mm_and_si128(packed, mask);
	__m128i	hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
	__m128i	idx = _mm_unpacklo_epi8(lo, hi);

	return _mm_shuffle_epi8(codebook, idx);
}

static inline __m128i TQ_GRAPH_AVX2_TARGET
TqGraphExpandPacked2Avx2(const uint8 *code)
{
	int32		codeI32;
	__m128i		data;
	__m128i		loMask = _mm_set1_epi8(0x0F);
	__m128i		loNibs;
	__m128i		hiNibs;
	__m128i		pairIdx;
	__m128i		tEven;
	__m128i		tOdd;
	__m128i		cEven;
	__m128i		cOdd;

	memcpy(&codeI32, code, sizeof(codeI32));
	data = _mm_cvtsi32_si128(codeI32);
	loNibs = _mm_and_si128(data, loMask);
	hiNibs = _mm_and_si128(_mm_srli_epi16(data, 4), loMask);
	pairIdx = _mm_unpacklo_epi8(loNibs, hiNibs);

	tEven = _mm_loadu_si128((const __m128i *) TqGraphCodebook2PairEvenI8);
	tOdd = _mm_loadu_si128((const __m128i *) TqGraphCodebook2PairOddI8);
	cEven = _mm_shuffle_epi8(tEven, pairIdx);
	cOdd = _mm_shuffle_epi8(tOdd, pairIdx);

	return _mm_unpacklo_epi8(cEven, cOdd);
}

static inline __m256i TQ_GRAPH_AVX2_TARGET
TqGraphDotI8x16Avx2(__m128i a, __m128i b)
{
	__m256i	a16 = _mm256_cvtepi8_epi16(a);
	__m256i	b16 = _mm256_cvtepi8_epi16(b);

	return _mm256_madd_epi16(a16, b16);
}

static int64 TQ_GRAPH_AVX2_TARGET
TqGraphQuerySplitRawAvx2(const HnswTqQuery *tq, const uint8 *code)
{
	__m256i	accLow = _mm256_setzero_si256();
	__m256i	accHigh = _mm256_setzero_si256();

	for (int chunk = 0; chunk < tq->querySplitChunks; chunk++)
	{
		__m128i	c = TqGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i	low = _mm_loadu_si128((const __m128i *) (tq->querySplitLow + chunk * 16));
		__m128i	high = _mm_loadu_si128((const __m128i *) (tq->querySplitHigh + chunk * 16));

		accLow = _mm256_add_epi32(accLow, TqGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, TqGraphDotI8x16Avx2(high, c));
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		__m128i	c;
		__m128i	low;
		__m128i	high;

		memcpy(scratch, code + tq->querySplitChunks * 8, tailBytes);
		c = TqGraphExpandPacked4Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->querySplitTailLow);
		high = _mm_loadu_si128((const __m128i *) tq->querySplitTailHigh);
		accLow = _mm256_add_epi32(accLow, TqGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, TqGraphDotI8x16Avx2(high, c));
	}

	return TqGraphHorizontalSumI32Avx2(accLow) +
		(int64) TQ_QUERY_SPLIT_HIGH_COEF * TqGraphHorizontalSumI32Avx2(accHigh);
}

static int64 TQ_GRAPH_AVX2_TARGET
TqGraphQuerySplit2RawAvx2(const HnswTqQuery *tq, const uint8 *code)
{
	__m256i	accLow = _mm256_setzero_si256();
	__m256i	accHigh = _mm256_setzero_si256();

	for (int chunk = 0; chunk < tq->querySplitChunks; chunk++)
	{
		__m128i	c = TqGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i	low = _mm_loadu_si128((const __m128i *) (tq->querySplitLow + chunk * 16));
		__m128i	high = _mm_loadu_si128((const __m128i *) (tq->querySplitHigh + chunk * 16));

		accLow = _mm256_add_epi32(accLow, TqGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, TqGraphDotI8x16Avx2(high, c));
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[4] = {0};
		int			tailBytes = (tq->querySplitTailDims + 3) / 4;
		__m128i	c;
		__m128i	low;
		__m128i	high;

		memcpy(scratch, code + tq->querySplitChunks * 4, tailBytes);
		c = TqGraphExpandPacked2Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->querySplitTailLow);
		high = _mm_loadu_si128((const __m128i *) tq->querySplitTailHigh);
		accLow = _mm256_add_epi32(accLow, TqGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, TqGraphDotI8x16Avx2(high, c));
	}

	return TqGraphHorizontalSumI32Avx2(accLow) +
		(int64) TQ_QUERY_SPLIT_HIGH_COEF * TqGraphHorizontalSumI32Avx2(accHigh);
}

/*
 * AVX2 SIMD weighted symmetric (code-code) kernels for TQ+.
 *
 * For each chunk of 16 coords (4-bit unpack via shuffle, 2-bit unpack
 * via the pair-table), expand a/b to i8x16, widen to i16x16, then:
 *
 *    prod = c_a · c_b              (i16, fits since |c| ≤ 127)
 *    pw   = madd(prod, weights)    (Σ pairs of i16 → i32, 8 lanes)
 *    acc += widen i32 → i64
 *
 * Returns Σ c_a[d] · c_b[d] · D'²_i16[d] as i64.  Caller divides by
 * `weight_scale · CODEBOOK_SCALE²` to recover the f32 weighted dot.
 *
 * No pruning (unlike the unweighted kernel) — every coord must
 * contribute its weight, otherwise the formula degenerates.  At
 * dim=1536 this is ~3× the unweighted kernel's per-pair cost (32
 * chunks vs 96 chunks scanned, plus the widen-mul-madd overhead);
 * still ~50× faster than the scalar fallback.
 */
static inline int64 TQ_GRAPH_AVX2_TARGET
TqGraphHorizontalSumI64Avx2(__m256i v)
{
	int64		s[4];

	_mm256_storeu_si256((__m256i *) s, v);
	return s[0] + s[1] + s[2] + s[3];
}

static inline __m256i TQ_GRAPH_AVX2_TARGET
TqGraphWeightedDotI8x16Avx2(__m128i ca, __m128i cb,
							 const int16 *weightsAt)
{
	__m256i		caI16 = _mm256_cvtepi8_epi16(ca);
	__m256i		cbI16 = _mm256_cvtepi8_epi16(cb);
	__m256i		prod = _mm256_mullo_epi16(caI16, cbI16);
	__m256i		w = _mm256_loadu_si256((const __m256i *) weightsAt);
	__m256i		pw = _mm256_madd_epi16(prod, w);
	__m256i		pwLoI64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(pw));
	__m256i		pwHiI64 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(pw, 1));

	return _mm256_add_epi64(pwLoI64, pwHiI64);
}

static int64 TQ_GRAPH_AVX2_TARGET
TqGraphCodeCodeWeightedRawAvx2(const uint8 *a, const uint8 *b,
								const int16 *weights, int dim)
{
	__m256i		acc = _mm256_setzero_si256();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m128i		ca = TqGraphExpandPacked4Avx2(a + chunk * 8);
		__m128i		cb = TqGraphExpandPacked4Avx2(b + chunk * 8);

		acc = _mm256_add_epi64(acc,
								TqGraphWeightedDotI8x16Avx2(ca, cb, weights + chunk * 16));
	}

	if (tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int16		scratchW[16] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		memcpy(scratchW, weights + chunks * 16, sizeof(int16) * tailDims);
		ca = TqGraphExpandPacked4Avx2(scratchA);
		cb = TqGraphExpandPacked4Avx2(scratchB);
		acc = _mm256_add_epi64(acc, TqGraphWeightedDotI8x16Avx2(ca, cb, scratchW));
	}

	return TqGraphHorizontalSumI64Avx2(acc);
}

static int64 TQ_GRAPH_AVX2_TARGET
TqGraphCodeCode2WeightedRawAvx2(const uint8 *a, const uint8 *b,
								 const int16 *weights, int dim)
{
	__m256i		acc = _mm256_setzero_si256();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m128i		ca = TqGraphExpandPacked2Avx2(a + chunk * 4);
		__m128i		cb = TqGraphExpandPacked2Avx2(b + chunk * 4);

		acc = _mm256_add_epi64(acc,
								TqGraphWeightedDotI8x16Avx2(ca, cb, weights + chunk * 16));
	}

	if (tailDims != 0)
	{
		uint8		scratchA[4] = {0};
		uint8		scratchB[4] = {0};
		int16		scratchW[16] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 4, tailBytes);
		memcpy(scratchB, b + chunks * 4, tailBytes);
		memcpy(scratchW, weights + chunks * 16, sizeof(int16) * tailDims);
		ca = TqGraphExpandPacked2Avx2(scratchA);
		cb = TqGraphExpandPacked2Avx2(scratchB);
		acc = _mm256_add_epi64(acc, TqGraphWeightedDotI8x16Avx2(ca, cb, scratchW));
	}

	return TqGraphHorizontalSumI64Avx2(acc);
}

#if TQ_GRAPH_COMPILE_AVX512_WEIGHTED
static bool
TqGraphAvx512WeightedAvailable(void)
{
	static int	available = -1;

	if (hnsw_tq_graph_avx512_weighted == TQ_GRAPH_AVX512_WEIGHTED_OFF ||
		hnsw_tq_simd_force == TQ_SIMD_FORCE_SCALAR)
		return false;

	if (hnsw_tq_simd_force != TQ_SIMD_FORCE_AUTO &&
		hnsw_tq_simd_force != TQ_SIMD_FORCE_AVX512VNNI)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
	available = 1;
#elif TQ_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))
	available = __builtin_cpu_supports("avx512f") &&
		__builtin_cpu_supports("avx512bw") &&
		__builtin_cpu_supports("avx512dq") &&
		__builtin_cpu_supports("avx512vl") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}

static inline int64 TQ_GRAPH_AVX512_WEIGHTED_TARGET
TqGraphHorizontalSumI32x16Avx512(__m512i v)
{
	int32		s[16];
	int64		sum = 0;

	_mm512_storeu_si512((__m512i *) s, v);
	for (int i = 0; i < 16; i++)
		sum += s[i];

	return sum;
}

static inline __m256i TQ_GRAPH_AVX512_WEIGHTED_TARGET
TqGraphExpandPacked4x32Avx512(const uint8 *code)
{
	__m128i		lo = TqGraphExpandPacked4Avx2(code);
	__m128i		hi = TqGraphExpandPacked4Avx2(code + 8);

	return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline __m256i TQ_GRAPH_AVX512_WEIGHTED_TARGET
TqGraphExpandPacked2x32Avx512(const uint8 *code)
{
	__m128i		lo = TqGraphExpandPacked2Avx2(code);
	__m128i		hi = TqGraphExpandPacked2Avx2(code + 4);

	return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline int64 TQ_GRAPH_AVX512_WEIGHTED_TARGET
TqGraphWeightedDotI8x32Avx512(__m256i ca, __m256i cb,
							  const int16 *weightsAt)
{
	__m512i		caI16 = _mm512_cvtepi8_epi16(ca);
	__m512i		cbI16 = _mm512_cvtepi8_epi16(cb);
	__m512i		prod = _mm512_mullo_epi16(caI16, cbI16);
	__m512i		w = _mm512_loadu_si512((const __m512i *) weightsAt);
	__m512i		pw = _mm512_madd_epi16(prod, w);

	return TqGraphHorizontalSumI32x16Avx512(pw);
}

static int64 TQ_GRAPH_AVX512_WEIGHTED_TARGET
TqGraphCodeCodeWeightedRawAvx512(const uint8 *a, const uint8 *b,
								 const int16 *weights, int dim)
{
	int64		acc = 0;
	int			chunks = dim / 32;
	int			tailDims = dim - chunks * 32;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m256i		ca = TqGraphExpandPacked4x32Avx512(a + chunk * 16);
		__m256i		cb = TqGraphExpandPacked4x32Avx512(b + chunk * 16);

		acc += TqGraphWeightedDotI8x32Avx512(ca, cb, weights + chunk * 32);
	}

	if (tailDims != 0)
	{
		uint8		scratchA[16] = {0};
		uint8		scratchB[16] = {0};
		int16		scratchW[32] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m256i		ca;
		__m256i		cb;

		memcpy(scratchA, a + chunks * 16, tailBytes);
		memcpy(scratchB, b + chunks * 16, tailBytes);
		memcpy(scratchW, weights + chunks * 32, sizeof(int16) * tailDims);
		ca = TqGraphExpandPacked4x32Avx512(scratchA);
		cb = TqGraphExpandPacked4x32Avx512(scratchB);
		acc += TqGraphWeightedDotI8x32Avx512(ca, cb, scratchW);
	}

	return acc;
}

static int64 TQ_GRAPH_AVX512_WEIGHTED_TARGET
TqGraphCodeCode2WeightedRawAvx512(const uint8 *a, const uint8 *b,
								  const int16 *weights, int dim)
{
	int64		acc = 0;
	int			chunks = dim / 32;
	int			tailDims = dim - chunks * 32;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m256i		ca = TqGraphExpandPacked2x32Avx512(a + chunk * 8);
		__m256i		cb = TqGraphExpandPacked2x32Avx512(b + chunk * 8);

		acc += TqGraphWeightedDotI8x32Avx512(ca, cb, weights + chunk * 32);
	}

	if (tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int16		scratchW[32] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m256i		ca;
		__m256i		cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		memcpy(scratchW, weights + chunks * 32, sizeof(int16) * tailDims);
		ca = TqGraphExpandPacked2x32Avx512(scratchA);
		cb = TqGraphExpandPacked2x32Avx512(scratchB);
		acc += TqGraphWeightedDotI8x32Avx512(ca, cb, scratchW);
	}

	return acc;
}
#endif

static int64 TQ_GRAPH_AVX2_TARGET
TqGraphCodeCodeRawAvx2(const uint8 *a, const uint8 *b, int dim,
					   int *sampleDims)
{
	__m256i	acc = _mm256_setzero_si256();
	int		chunks = dim / 16;
	int		tailDims = dim - chunks * 16;
	int		scoredChunks = chunks;

	if (chunks > TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i	ca = TqGraphExpandPacked4Avx2(a + chunk * 8);
		__m128i	cb = TqGraphExpandPacked4Avx2(b + chunk * 8);

		acc = _mm256_add_epi32(acc, TqGraphDotI8x16Avx2(ca, cb));
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m128i	ca;
		__m128i	cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		ca = TqGraphExpandPacked4Avx2(scratchA);
		cb = TqGraphExpandPacked4Avx2(scratchB);
		acc = _mm256_add_epi32(acc, TqGraphDotI8x16Avx2(ca, cb));
		*sampleDims += tailDims;
	}

	return TqGraphHorizontalSumI32Avx2(acc);
}

static int64 TQ_GRAPH_AVX2_TARGET
TqGraphCodeCode2RawAvx2(const uint8 *a, const uint8 *b, int dim,
						int *sampleDims)
{
	__m256i	acc = _mm256_setzero_si256();
	int		chunks = dim / 16;
	int		tailDims = dim - chunks * 16;
	int		scoredChunks = chunks;

	if (chunks > TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i	ca = TqGraphExpandPacked2Avx2(a + chunk * 4);
		__m128i	cb = TqGraphExpandPacked2Avx2(b + chunk * 4);

		acc = _mm256_add_epi32(acc, TqGraphDotI8x16Avx2(ca, cb));
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[4] = {0};
		uint8		scratchB[4] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m128i	ca;
		__m128i	cb;

		memcpy(scratchA, a + chunks * 4, tailBytes);
		memcpy(scratchB, b + chunks * 4, tailBytes);
		ca = TqGraphExpandPacked2Avx2(scratchA);
		cb = TqGraphExpandPacked2Avx2(scratchB);
		acc = _mm256_add_epi32(acc, TqGraphDotI8x16Avx2(ca, cb));
		*sampleDims += tailDims;
	}

	return TqGraphHorizontalSumI32Avx2(acc);
}

#if TQ_GRAPH_COMPILE_AVX512VNNI
static bool
TqGraphAvx512VnniAvailable(void)
{
	static int	available = -1;

	/*
	 * The hnsw.tq_graph_avx512vnni GUC is consulted on every call so a
	 * downclock or dispatch-fallback measurement can flip without
	 * restarting the backend.  The CPU-feature probe is still memoised.
	 */
	if (!hnsw_tq_graph_avx512vnni)
		return false;
	if (hnsw_tq_simd_force != TQ_SIMD_FORCE_AUTO &&
		hnsw_tq_simd_force != TQ_SIMD_FORCE_AVX512VNNI)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__)
	available = 1;
#elif TQ_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))
	available = __builtin_cpu_supports("avx512vnni") &&
		__builtin_cpu_supports("avx512vl") &&
		__builtin_cpu_supports("avx512bw") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}

/*
 * AVX-512 VNNI ZMM scoring kernels.
 *
 * Process 64 dims per iteration via _mm512_dpbusd_epi32 on ZMM, with
 * 32-dim YMM and 16-dim XMM tails for residuals. Same XOR-0x80 trick as
 * the AVX-VNNI tier; the -128 * sum(c) correction is hoisted into a third
 * VNNI accumulator that runs in parallel.
 *
 * AVX-512 dispatch should be capped at hosts that don't downclock heavily
 * under wide vectors (Ice Lake server, Sapphire Rapids, Zen 4 server, and
 * newer). Skylake-X / Cascade Lake suffer measurable drops; the 256-bit
 * AVX-VNNI tier is preferred there.
 */

static inline __m512i TQ_GRAPH_AVX512VNNI_TARGET
TqGraphExpandPacked4Avx512(const uint8 *code)
{
	__m128i		c0 = TqGraphExpandPacked4Avx2(code + 0);
	__m128i		c1 = TqGraphExpandPacked4Avx2(code + 8);
	__m128i		c2 = TqGraphExpandPacked4Avx2(code + 16);
	__m128i		c3 = TqGraphExpandPacked4Avx2(code + 24);
	__m256i		lo = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
	__m256i		hi = _mm256_inserti128_si256(_mm256_castsi128_si256(c2), c3, 1);

	return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

static inline __m512i TQ_GRAPH_AVX512VNNI_TARGET
TqGraphExpandPacked2Avx512(const uint8 *code)
{
	__m128i		c0 = TqGraphExpandPacked2Avx2(code + 0);
	__m128i		c1 = TqGraphExpandPacked2Avx2(code + 4);
	__m128i		c2 = TqGraphExpandPacked2Avx2(code + 8);
	__m128i		c3 = TqGraphExpandPacked2Avx2(code + 12);
	__m256i		lo = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
	__m256i		hi = _mm256_inserti128_si256(_mm256_castsi128_si256(c2), c3, 1);

	return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

static inline int64 TQ_GRAPH_AVX512VNNI_TARGET
TqGraphHorizontalSumI32x4Avx512Vnni(__m128i v)
{
	int32		s[4];

	_mm_storeu_si128((__m128i *) s, v);
	return (int64) s[0] + s[1] + s[2] + s[3];
}

static int64 TQ_GRAPH_AVX512VNNI_TARGET
TqGraphQuerySplitRawAvx512Vnni(const HnswTqQuery *tq, const uint8 *code)
{
	const __m512i ones512 = _mm512_set1_epi8(1);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m512i		accLow512 = _mm512_setzero_si512();
	__m512i		accHigh512 = _mm512_setzero_si512();
	__m512i		accCSum512 = _mm512_setzero_si512();
	__m256i		accLow256 = _mm256_setzero_si256();
	__m256i		accHigh256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLow128 = _mm_setzero_si128();
	__m128i		accHigh128 = _mm_setzero_si128();
	__m128i		accCSum128 = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	/* Quad-stepped main loop: 64 dims per iteration on ZMM. */
	for (; chunk + 4 <= tq->querySplitChunks; chunk += 4)
	{
		__m512i		c = TqGraphExpandPacked4Avx512(code + chunk * 8);
		__m512i		low = _mm512_loadu_si512((const __m512i *) (tq->querySplitLowU8 + chunk * 16));
		__m512i		high = _mm512_loadu_si512((const __m512i *) (tq->querySplitHighU8 + chunk * 16));

		accLow512 = _mm512_dpbusd_epi32(accLow512, low, c);
		accHigh512 = _mm512_dpbusd_epi32(accHigh512, high, c);
		accCSum512 = _mm512_dpbusd_epi32(accCSum512, ones512, c);
	}

	/* Pair-step trailing 32 dims on YMM. */
	if (chunk + 2 <= tq->querySplitChunks)
	{
		__m128i		c0 = TqGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i		c1 = TqGraphExpandPacked4Avx2(code + (chunk + 1) * 8);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->querySplitLowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->querySplitHighU8 + chunk * 16));

		accLow256 = _mm256_dpbusd_epi32(accLow256, low, c);
		accHigh256 = _mm256_dpbusd_epi32(accHigh256, high, c);
		accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, c);
		chunk += 2;
	}

	/* Trailing 16-dim chunk on XMM. */
	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = TqGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->querySplitLowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->querySplitHighU8 + chunk * 16));

		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	/* Sub-chunk tail dims. */
	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		__m128i		c;
		__m128i		low;
		__m128i		high;

		memcpy(scratch, code + tq->querySplitChunks * 8, tailBytes);
		c = TqGraphExpandPacked4Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->querySplitTailLowU8);
		high = _mm_loadu_si128((const __m128i *) tq->querySplitTailHighU8);
		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	dotLow = (int64) _mm512_reduce_add_epi32(accLow512) +
		TqGraphHorizontalSumI32Avx2(accLow256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(accLow128);
	dotHigh = (int64) _mm512_reduce_add_epi32(accHigh512) +
		TqGraphHorizontalSumI32Avx2(accHigh256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(accHigh128);
	cSum = (int64) _mm512_reduce_add_epi32(accCSum512) +
		TqGraphHorizontalSumI32Avx2(accCSum256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(accCSum128);

	return (dotLow - 128 * cSum) +
		(int64) TQ_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}

static int64 TQ_GRAPH_AVX512VNNI_TARGET
TqGraphQuerySplit2RawAvx512Vnni(const HnswTqQuery *tq, const uint8 *code)
{
	const __m512i ones512 = _mm512_set1_epi8(1);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m512i		accLow512 = _mm512_setzero_si512();
	__m512i		accHigh512 = _mm512_setzero_si512();
	__m512i		accCSum512 = _mm512_setzero_si512();
	__m256i		accLow256 = _mm256_setzero_si256();
	__m256i		accHigh256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLow128 = _mm_setzero_si128();
	__m128i		accHigh128 = _mm_setzero_si128();
	__m128i		accCSum128 = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	for (; chunk + 4 <= tq->querySplitChunks; chunk += 4)
	{
		__m512i		c = TqGraphExpandPacked2Avx512(code + chunk * 4);
		__m512i		low = _mm512_loadu_si512((const __m512i *) (tq->querySplitLowU8 + chunk * 16));
		__m512i		high = _mm512_loadu_si512((const __m512i *) (tq->querySplitHighU8 + chunk * 16));

		accLow512 = _mm512_dpbusd_epi32(accLow512, low, c);
		accHigh512 = _mm512_dpbusd_epi32(accHigh512, high, c);
		accCSum512 = _mm512_dpbusd_epi32(accCSum512, ones512, c);
	}

	if (chunk + 2 <= tq->querySplitChunks)
	{
		__m128i		c0 = TqGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i		c1 = TqGraphExpandPacked2Avx2(code + (chunk + 1) * 4);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->querySplitLowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->querySplitHighU8 + chunk * 16));

		accLow256 = _mm256_dpbusd_epi32(accLow256, low, c);
		accHigh256 = _mm256_dpbusd_epi32(accHigh256, high, c);
		accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, c);
		chunk += 2;
	}

	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = TqGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->querySplitLowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->querySplitHighU8 + chunk * 16));

		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[4] = {0};
		int			tailBytes = (tq->querySplitTailDims + 3) / 4;
		__m128i		c;
		__m128i		low;
		__m128i		high;

		memcpy(scratch, code + tq->querySplitChunks * 4, tailBytes);
		c = TqGraphExpandPacked2Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->querySplitTailLowU8);
		high = _mm_loadu_si128((const __m128i *) tq->querySplitTailHighU8);
		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	dotLow = (int64) _mm512_reduce_add_epi32(accLow512) +
		TqGraphHorizontalSumI32Avx2(accLow256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(accLow128);
	dotHigh = (int64) _mm512_reduce_add_epi32(accHigh512) +
		TqGraphHorizontalSumI32Avx2(accHigh256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(accHigh128);
	cSum = (int64) _mm512_reduce_add_epi32(accCSum512) +
		TqGraphHorizontalSumI32Avx2(accCSum256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(accCSum128);

	return (dotLow - 128 * cSum) +
		(int64) TQ_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}

static int64 TQ_GRAPH_AVX512VNNI_TARGET
TqGraphCodeCodeRawAvx512Vnni(const uint8 *a, const uint8 *b, int dim,
							 int *sampleDims)
{
	const __m512i signFlip512 = _mm512_set1_epi8((char) 0x80);
	const __m512i ones512 = _mm512_set1_epi8(1);
	const __m256i signFlip256 = _mm256_set1_epi8((char) 0x80);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i signFlip128 = _mm_set1_epi8((char) 0x80);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m512i		acc512 = _mm512_setzero_si512();
	__m512i		accCSum512 = _mm512_setzero_si512();
	__m256i		acc256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		acc128 = _mm_setzero_si128();
	__m128i		accCSum128 = _mm_setzero_si128();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
	int			scored = 0;
	int64		dot;
	int64		cSum;

	if (chunks > TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	if (scoredChunks == chunks)
	{
		/*
		 * Contiguous full traversal — drive ZMM/YMM strides directly across
		 * the input.
		 */
		for (; scored + 4 <= scoredChunks; scored += 4)
		{
			__m512i		ca = TqGraphExpandPacked4Avx512(a + scored * 8);
			__m512i		cb = TqGraphExpandPacked4Avx512(b + scored * 8);

			acc512 = _mm512_dpbusd_epi32(acc512,
										 _mm512_xor_si512(ca, signFlip512), cb);
			accCSum512 = _mm512_dpbusd_epi32(accCSum512, ones512, cb);
		}
		if (scored + 2 <= scoredChunks)
		{
			__m128i		ca0 = TqGraphExpandPacked4Avx2(a + scored * 8);
			__m128i		cb0 = TqGraphExpandPacked4Avx2(b + scored * 8);
			__m128i		ca1 = TqGraphExpandPacked4Avx2(a + (scored + 1) * 8);
			__m128i		cb1 = TqGraphExpandPacked4Avx2(b + (scored + 1) * 8);
			__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
			__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

			acc256 = _mm256_dpbusd_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
			accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, cb);
			scored += 2;
		}
	}
	else
	{
		/* Stride-sampled scoring; can still pair adjacent samples. */
		for (; scored + 2 <= scoredChunks; scored += 2)
		{
			int			chunkA = (int) (((int64) scored * chunks) / scoredChunks);
			int			chunkB = (int) (((int64) (scored + 1) * chunks) / scoredChunks);
			__m128i		ca0 = TqGraphExpandPacked4Avx2(a + chunkA * 8);
			__m128i		cb0 = TqGraphExpandPacked4Avx2(b + chunkA * 8);
			__m128i		ca1 = TqGraphExpandPacked4Avx2(a + chunkB * 8);
			__m128i		cb1 = TqGraphExpandPacked4Avx2(b + chunkB * 8);
			__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
			__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

			acc256 = _mm256_dpbusd_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
			accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, cb);
		}
	}

	if (scored < scoredChunks)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i		ca = TqGraphExpandPacked4Avx2(a + chunk * 8);
		__m128i		cb = TqGraphExpandPacked4Avx2(b + chunk * 8);

		acc128 = _mm_dpbusd_epi32(acc128,
								  _mm_xor_si128(ca, signFlip128), cb);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		ca = TqGraphExpandPacked4Avx2(scratchA);
		cb = TqGraphExpandPacked4Avx2(scratchB);
		acc128 = _mm_dpbusd_epi32(acc128,
								  _mm_xor_si128(ca, signFlip128), cb);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, cb);
		*sampleDims += tailDims;
	}

	dot = (int64) _mm512_reduce_add_epi32(acc512) +
		TqGraphHorizontalSumI32Avx2(acc256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(acc128);
	cSum = (int64) _mm512_reduce_add_epi32(accCSum512) +
		TqGraphHorizontalSumI32Avx2(accCSum256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(accCSum128);

	return dot - 128 * cSum;
}

static int64 TQ_GRAPH_AVX512VNNI_TARGET
TqGraphCodeCode2RawAvx512Vnni(const uint8 *a, const uint8 *b, int dim,
							  int *sampleDims)
{
	const __m512i signFlip512 = _mm512_set1_epi8((char) 0x80);
	const __m512i ones512 = _mm512_set1_epi8(1);
	const __m256i signFlip256 = _mm256_set1_epi8((char) 0x80);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i signFlip128 = _mm_set1_epi8((char) 0x80);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m512i		acc512 = _mm512_setzero_si512();
	__m512i		accCSum512 = _mm512_setzero_si512();
	__m256i		acc256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		acc128 = _mm_setzero_si128();
	__m128i		accCSum128 = _mm_setzero_si128();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
	int			scored = 0;
	int64		dot;
	int64		cSum;

	if (chunks > TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	if (scoredChunks == chunks)
	{
		for (; scored + 4 <= scoredChunks; scored += 4)
		{
			__m512i		ca = TqGraphExpandPacked2Avx512(a + scored * 4);
			__m512i		cb = TqGraphExpandPacked2Avx512(b + scored * 4);

			acc512 = _mm512_dpbusd_epi32(acc512,
										 _mm512_xor_si512(ca, signFlip512), cb);
			accCSum512 = _mm512_dpbusd_epi32(accCSum512, ones512, cb);
		}
		if (scored + 2 <= scoredChunks)
		{
			__m128i		ca0 = TqGraphExpandPacked2Avx2(a + scored * 4);
			__m128i		cb0 = TqGraphExpandPacked2Avx2(b + scored * 4);
			__m128i		ca1 = TqGraphExpandPacked2Avx2(a + (scored + 1) * 4);
			__m128i		cb1 = TqGraphExpandPacked2Avx2(b + (scored + 1) * 4);
			__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
			__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

			acc256 = _mm256_dpbusd_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
			accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, cb);
			scored += 2;
		}
	}
	else
	{
		for (; scored + 2 <= scoredChunks; scored += 2)
		{
			int			chunkA = (int) (((int64) scored * chunks) / scoredChunks);
			int			chunkB = (int) (((int64) (scored + 1) * chunks) / scoredChunks);
			__m128i		ca0 = TqGraphExpandPacked2Avx2(a + chunkA * 4);
			__m128i		cb0 = TqGraphExpandPacked2Avx2(b + chunkA * 4);
			__m128i		ca1 = TqGraphExpandPacked2Avx2(a + chunkB * 4);
			__m128i		cb1 = TqGraphExpandPacked2Avx2(b + chunkB * 4);
			__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
			__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

			acc256 = _mm256_dpbusd_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
			accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, cb);
		}
	}

	if (scored < scoredChunks)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i		ca = TqGraphExpandPacked2Avx2(a + chunk * 4);
		__m128i		cb = TqGraphExpandPacked2Avx2(b + chunk * 4);

		acc128 = _mm_dpbusd_epi32(acc128,
								  _mm_xor_si128(ca, signFlip128), cb);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[4] = {0};
		uint8		scratchB[4] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 4, tailBytes);
		memcpy(scratchB, b + chunks * 4, tailBytes);
		ca = TqGraphExpandPacked2Avx2(scratchA);
		cb = TqGraphExpandPacked2Avx2(scratchB);
		acc128 = _mm_dpbusd_epi32(acc128,
								  _mm_xor_si128(ca, signFlip128), cb);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, cb);
		*sampleDims += tailDims;
	}

	dot = (int64) _mm512_reduce_add_epi32(acc512) +
		TqGraphHorizontalSumI32Avx2(acc256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(acc128);
	cSum = (int64) _mm512_reduce_add_epi32(accCSum512) +
		TqGraphHorizontalSumI32Avx2(accCSum256) +
		TqGraphHorizontalSumI32x4Avx512Vnni(accCSum128);

	return dot - 128 * cSum;
}
#endif

#if TQ_GRAPH_COMPILE_AVXVNNI
static bool
TqGraphAvxVnniAvailable(void)
{
	static int	available = -1;

	/* GUC kill-switch — see hnsw.tq_graph_avxvnni in hnsw.c. */
	if (!hnsw_tq_graph_avxvnni)
		return false;
	if (hnsw_tq_simd_force != TQ_SIMD_FORCE_AUTO &&
		hnsw_tq_simd_force != TQ_SIMD_FORCE_AVXVNNI)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVXVNNI__)
	available = 1;
#elif TQ_GRAPH_X86 && \
	(defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || \
	(defined(__clang__) && __clang_major__ >= 18)
	available = __builtin_cpu_supports("avxvnni") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}

/*
 * AVX-VNNI YMM scoring kernels.
 *
 * Each iteration consumes 32 i8 dims via vpdpbusd on a 256-bit register
 * (one fused uop on Alder Lake / Zen 4 desktop, three or four on plain
 * AVX2 via vpmaddubsw + vpmaddwd + vpaddd). Signed-vs-signed dot product
 * is synthesised from VNNI's u8 * i8 form with the standard XOR-0x80
 * trick:
 *
 *   sum(a_signed * b_signed)
 *     = sum((a_signed XOR 0x80) * b_signed) - 128 * sum(b_signed)
 *
 * The right-hand correction is hoisted out of the inner loop with a third
 * VNNI accumulator running `dpbusd(accCSum, ones_u8, b_signed)`, which
 * produces sum(b_signed) once at the end and replaces the per-chunk
 * cvtepi8_epi16 / madd / hsum chain that the XMM version performed.
 */

static inline int64 TQ_GRAPH_AVXVNNI_TARGET
TqGraphHorizontalSumI32x4AvxVnni(__m128i v)
{
	int32		s[4];

	_mm_storeu_si128((__m128i *) s, v);
	return (int64) s[0] + s[1] + s[2] + s[3];
}

static int64 TQ_GRAPH_AVXVNNI_TARGET
TqGraphQuerySplitRawAvxVnni(const HnswTqQuery *tq, const uint8 *code)
{
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		accLow = _mm256_setzero_si256();
	__m256i		accHigh = _mm256_setzero_si256();
	__m256i		accCSum = _mm256_setzero_si256();
	__m128i		accLowLo = _mm_setzero_si128();
	__m128i		accHighLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	/* Pair-stepped main loop: 32 dims per iteration on YMM. */
	for (; chunk + 2 <= tq->querySplitChunks; chunk += 2)
	{
		__m128i		c0 = TqGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i		c1 = TqGraphExpandPacked4Avx2(code + (chunk + 1) * 8);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->querySplitLowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->querySplitHighU8 + chunk * 16));

		accLow = _mm256_dpbusd_avx_epi32(accLow, low, c);
		accHigh = _mm256_dpbusd_avx_epi32(accHigh, high, c);
		accCSum = _mm256_dpbusd_avx_epi32(accCSum, ones256, c);
	}

	/* Trailing 16-dim chunk (if querySplitChunks is odd). */
	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = TqGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->querySplitLowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->querySplitHighU8 + chunk * 16));

		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	/* Final sub-chunk tail dims. */
	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		__m128i		c;
		__m128i		low;
		__m128i		high;

		memcpy(scratch, code + tq->querySplitChunks * 8, tailBytes);
		c = TqGraphExpandPacked4Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->querySplitTailLowU8);
		high = _mm_loadu_si128((const __m128i *) tq->querySplitTailHighU8);
		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	dotLow = TqGraphHorizontalSumI32Avx2(accLow) +
		TqGraphHorizontalSumI32x4AvxVnni(accLowLo);
	dotHigh = TqGraphHorizontalSumI32Avx2(accHigh) +
		TqGraphHorizontalSumI32x4AvxVnni(accHighLo);
	cSum = TqGraphHorizontalSumI32Avx2(accCSum) +
		TqGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return (dotLow - 128 * cSum) +
		(int64) TQ_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}

static int64 TQ_GRAPH_AVXVNNI_TARGET
TqGraphQuerySplit2RawAvxVnni(const HnswTqQuery *tq, const uint8 *code)
{
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		accLow = _mm256_setzero_si256();
	__m256i		accHigh = _mm256_setzero_si256();
	__m256i		accCSum = _mm256_setzero_si256();
	__m128i		accLowLo = _mm_setzero_si128();
	__m128i		accHighLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	for (; chunk + 2 <= tq->querySplitChunks; chunk += 2)
	{
		__m128i		c0 = TqGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i		c1 = TqGraphExpandPacked2Avx2(code + (chunk + 1) * 4);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->querySplitLowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->querySplitHighU8 + chunk * 16));

		accLow = _mm256_dpbusd_avx_epi32(accLow, low, c);
		accHigh = _mm256_dpbusd_avx_epi32(accHigh, high, c);
		accCSum = _mm256_dpbusd_avx_epi32(accCSum, ones256, c);
	}

	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = TqGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->querySplitLowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->querySplitHighU8 + chunk * 16));

		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[4] = {0};
		int			tailBytes = (tq->querySplitTailDims + 3) / 4;
		__m128i		c;
		__m128i		low;
		__m128i		high;

		memcpy(scratch, code + tq->querySplitChunks * 4, tailBytes);
		c = TqGraphExpandPacked2Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->querySplitTailLowU8);
		high = _mm_loadu_si128((const __m128i *) tq->querySplitTailHighU8);
		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	dotLow = TqGraphHorizontalSumI32Avx2(accLow) +
		TqGraphHorizontalSumI32x4AvxVnni(accLowLo);
	dotHigh = TqGraphHorizontalSumI32Avx2(accHigh) +
		TqGraphHorizontalSumI32x4AvxVnni(accHighLo);
	cSum = TqGraphHorizontalSumI32Avx2(accCSum) +
		TqGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return (dotLow - 128 * cSum) +
		(int64) TQ_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}

static int64 TQ_GRAPH_AVXVNNI_TARGET
TqGraphCodeCodeRawAvxVnni(const uint8 *a, const uint8 *b, int dim,
						  int *sampleDims)
{
	const __m256i signFlip256 = _mm256_set1_epi8((char) 0x80);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i signFlip128 = _mm_set1_epi8((char) 0x80);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		acc256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
	int			scored = 0;
	int64		dot;
	int64		cSum;

	if (chunks > TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (; scored + 2 <= scoredChunks; scored += 2)
	{
		int			chunkA = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		int			chunkB = scoredChunks == chunks ? scored + 1 :
			(int) (((int64) (scored + 1) * chunks) / scoredChunks);
		__m128i		ca0 = TqGraphExpandPacked4Avx2(a + chunkA * 8);
		__m128i		cb0 = TqGraphExpandPacked4Avx2(b + chunkA * 8);
		__m128i		ca1 = TqGraphExpandPacked4Avx2(a + chunkB * 8);
		__m128i		cb1 = TqGraphExpandPacked4Avx2(b + chunkB * 8);
		__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
		__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

		acc256 = _mm256_dpbusd_avx_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
		accCSum256 = _mm256_dpbusd_avx_epi32(accCSum256, ones256, cb);
	}

	if (scored < scoredChunks)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i		ca = TqGraphExpandPacked4Avx2(a + chunk * 8);
		__m128i		cb = TqGraphExpandPacked4Avx2(b + chunk * 8);

		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		ca = TqGraphExpandPacked4Avx2(scratchA);
		cb = TqGraphExpandPacked4Avx2(scratchB);
		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
		*sampleDims += tailDims;
	}

	dot = TqGraphHorizontalSumI32Avx2(acc256) +
		TqGraphHorizontalSumI32x4AvxVnni(accLo);
	cSum = TqGraphHorizontalSumI32Avx2(accCSum256) +
		TqGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return dot - 128 * cSum;
}

static int64 TQ_GRAPH_AVXVNNI_TARGET
TqGraphCodeCode2RawAvxVnni(const uint8 *a, const uint8 *b, int dim,
						   int *sampleDims)
{
	const __m256i signFlip256 = _mm256_set1_epi8((char) 0x80);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i signFlip128 = _mm_set1_epi8((char) 0x80);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		acc256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
	int			scored = 0;
	int64		dot;
	int64		cSum;

	if (chunks > TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (; scored + 2 <= scoredChunks; scored += 2)
	{
		int			chunkA = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		int			chunkB = scoredChunks == chunks ? scored + 1 :
			(int) (((int64) (scored + 1) * chunks) / scoredChunks);
		__m128i		ca0 = TqGraphExpandPacked2Avx2(a + chunkA * 4);
		__m128i		cb0 = TqGraphExpandPacked2Avx2(b + chunkA * 4);
		__m128i		ca1 = TqGraphExpandPacked2Avx2(a + chunkB * 4);
		__m128i		cb1 = TqGraphExpandPacked2Avx2(b + chunkB * 4);
		__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
		__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

		acc256 = _mm256_dpbusd_avx_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
		accCSum256 = _mm256_dpbusd_avx_epi32(accCSum256, ones256, cb);
	}

	if (scored < scoredChunks)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i		ca = TqGraphExpandPacked2Avx2(a + chunk * 4);
		__m128i		cb = TqGraphExpandPacked2Avx2(b + chunk * 4);

		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[4] = {0};
		uint8		scratchB[4] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 4, tailBytes);
		memcpy(scratchB, b + chunks * 4, tailBytes);
		ca = TqGraphExpandPacked2Avx2(scratchA);
		cb = TqGraphExpandPacked2Avx2(scratchB);
		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
		*sampleDims += tailDims;
	}

	dot = TqGraphHorizontalSumI32Avx2(acc256) +
		TqGraphHorizontalSumI32x4AvxVnni(accLo);
	cSum = TqGraphHorizontalSumI32Avx2(accCSum256) +
		TqGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return dot - 128 * cSum;
}
#endif

static bool TQ_GRAPH_AVX2_TARGET
TqGraphExactVectorDistanceAvx2(HnswScanOpaque so, Vector *queryVector,
							   Vector *valueVector, double *result)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	int			i = 0;

	if (mode == TQ_SCORE_L2)
	{
		__m256		acc0 = _mm256_setzero_ps();
		__m256		acc1 = _mm256_setzero_ps();
		__m256		acc2 = _mm256_setzero_ps();
		__m256		acc3 = _mm256_setzero_ps();

		for (; i + 32 <= queryVector->dim; i += 32)
		{
			__m256	q0 = _mm256_loadu_ps(&queryVector->x[i]);
			__m256	q1 = _mm256_loadu_ps(&queryVector->x[i + 8]);
			__m256	q2 = _mm256_loadu_ps(&queryVector->x[i + 16]);
			__m256	q3 = _mm256_loadu_ps(&queryVector->x[i + 24]);
			__m256	v0 = _mm256_loadu_ps(&valueVector->x[i]);
			__m256	v1 = _mm256_loadu_ps(&valueVector->x[i + 8]);
			__m256	v2 = _mm256_loadu_ps(&valueVector->x[i + 16]);
			__m256	v3 = _mm256_loadu_ps(&valueVector->x[i + 24]);
			__m256	d0 = _mm256_sub_ps(q0, v0);
			__m256	d1 = _mm256_sub_ps(q1, v1);
			__m256	d2 = _mm256_sub_ps(q2, v2);
			__m256	d3 = _mm256_sub_ps(q3, v3);

			acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
			acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
			acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
			acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
		}

		acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
							 _mm256_add_ps(acc2, acc3));
		*result = TqGraphHorizontalSumF32Avx2(acc0);
		for (; i < queryVector->dim; i++)
		{
			double		diff = (double) queryVector->x[i] - valueVector->x[i];

			*result += diff * diff;
		}
		return true;
	}

	if (mode == TQ_SCORE_IP || mode == TQ_SCORE_COSINE)
	{
		__m256		dotAcc = _mm256_setzero_ps();
		__m256		normAcc = _mm256_setzero_ps();
		double		dot;
		double		valueNorm = 0;

		for (; i + 8 <= queryVector->dim; i += 8)
		{
			__m256	qv = _mm256_loadu_ps(&queryVector->x[i]);
			__m256	vv = _mm256_loadu_ps(&valueVector->x[i]);

			dotAcc = _mm256_add_ps(dotAcc, _mm256_mul_ps(qv, vv));
			if (mode == TQ_SCORE_COSINE)
				normAcc = _mm256_add_ps(normAcc, _mm256_mul_ps(vv, vv));
		}

		dot = TqGraphHorizontalSumF32Avx2(dotAcc);
		if (mode == TQ_SCORE_COSINE)
			valueNorm = TqGraphHorizontalSumF32Avx2(normAcc);

		for (; i < queryVector->dim; i++)
		{
			double		qv = queryVector->x[i];
			double		vv = valueVector->x[i];

			dot += qv * vv;
			if (mode == TQ_SCORE_COSINE)
				valueNorm += vv * vv;
		}

		if (mode == TQ_SCORE_IP)
			*result = -dot;
		else if (so->tq.queryNorm == 0 || valueNorm == 0)
			*result = 1;
		else
			*result = 1 - (dot / sqrt(so->tq.queryNorm * valueNorm));

		return true;
	}

	return false;
}

static bool TQ_GRAPH_AVX2_TARGET
TqGraphBuildExactDistanceAvx2(TqGraphBuildState *state, Vector *av,
							  Vector *bv, double *result)
{
	TqScoreMode mode = (TqScoreMode) state->scoreMode;
	int			i = 0;

	if (mode == TQ_SCORE_L2)
	{
		__m256		acc0 = _mm256_setzero_ps();
		__m256		acc1 = _mm256_setzero_ps();
		__m256		acc2 = _mm256_setzero_ps();
		__m256		acc3 = _mm256_setzero_ps();

		for (; i + 32 <= av->dim; i += 32)
		{
			__m256	a0 = _mm256_loadu_ps(&av->x[i]);
			__m256	a1 = _mm256_loadu_ps(&av->x[i + 8]);
			__m256	a2 = _mm256_loadu_ps(&av->x[i + 16]);
			__m256	a3 = _mm256_loadu_ps(&av->x[i + 24]);
			__m256	b0 = _mm256_loadu_ps(&bv->x[i]);
			__m256	b1 = _mm256_loadu_ps(&bv->x[i + 8]);
			__m256	b2 = _mm256_loadu_ps(&bv->x[i + 16]);
			__m256	b3 = _mm256_loadu_ps(&bv->x[i + 24]);
			__m256	d0 = _mm256_sub_ps(a0, b0);
			__m256	d1 = _mm256_sub_ps(a1, b1);
			__m256	d2 = _mm256_sub_ps(a2, b2);
			__m256	d3 = _mm256_sub_ps(a3, b3);

			acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
			acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
			acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
			acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
		}

		acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
							 _mm256_add_ps(acc2, acc3));
		*result = TqGraphHorizontalSumF32Avx2(acc0);
		for (; i < av->dim; i++)
		{
			double		diff = (double) av->x[i] - bv->x[i];

			*result += diff * diff;
		}
		return true;
	}

	if (mode == TQ_SCORE_IP || mode == TQ_SCORE_COSINE)
	{
		__m256		dotAcc = _mm256_setzero_ps();
		__m256		aNormAcc = _mm256_setzero_ps();
		__m256		bNormAcc = _mm256_setzero_ps();
		double		dot;
		double		aNorm = 0;
		double		bNorm = 0;

		for (; i + 8 <= av->dim; i += 8)
		{
			__m256	avv = _mm256_loadu_ps(&av->x[i]);
			__m256	bvv = _mm256_loadu_ps(&bv->x[i]);

			dotAcc = _mm256_add_ps(dotAcc, _mm256_mul_ps(avv, bvv));
			if (mode == TQ_SCORE_COSINE)
			{
				aNormAcc = _mm256_add_ps(aNormAcc, _mm256_mul_ps(avv, avv));
				bNormAcc = _mm256_add_ps(bNormAcc, _mm256_mul_ps(bvv, bvv));
			}
		}

		dot = TqGraphHorizontalSumF32Avx2(dotAcc);
		if (mode == TQ_SCORE_COSINE)
		{
			aNorm = TqGraphHorizontalSumF32Avx2(aNormAcc);
			bNorm = TqGraphHorizontalSumF32Avx2(bNormAcc);
		}

		for (; i < av->dim; i++)
		{
			double		aval = av->x[i];
			double		bval = bv->x[i];

			dot += aval * bval;
			if (mode == TQ_SCORE_COSINE)
			{
				aNorm += aval * aval;
				bNorm += bval * bval;
			}
		}

		if (mode == TQ_SCORE_IP)
			*result = -dot;
		else if (aNorm == 0 || bNorm == 0)
			*result = 1;
		else
			*result = 1 - (dot / sqrt(aNorm * bNorm));

		return true;
	}

	return false;
}

#endif

double
TqGraphExactVectorDistance(HnswScanOpaque so, Datum query, char *valuePtr)
{
	Vector	   *queryVector = (Vector *) DatumGetPointer(query);
	Vector	   *valueVector = (Vector *) valuePtr;
	double		distance = 0;
	double		dot = 0;
	double		valueNorm = 0;
	TqScoreMode mode;

	if (queryVector == NULL || valueVector == NULL ||
		queryVector->dim != valueVector->dim || !so->tq.enabled)
	{
		if (so->support.procinfo == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("turboquant exact distance support function is missing")));
		HnswRecordExactVectorKernel(hnsw_tq_exact_simd_force == TQ_EXACT_SIMD_FORCE_SCALAR ?
									TQ_EXACT_KERNEL_SCALAR :
									TQ_EXACT_KERNEL_AUTOVEC_FMA);
		return TqGraphExactDistance(&so->support, query, PointerGetDatum(valuePtr));
	}

	mode = (TqScoreMode) so->tq.scoreMode;
#if TQ_GRAPH_COMPILE_AVX2
	if (hnsw_tq_exact_simd_force != TQ_EXACT_SIMD_FORCE_SCALAR &&
		TqGraphAvx2Available() &&
		TqGraphExactVectorDistanceAvx2(so, queryVector, valueVector, &distance))
	{
		HnswRecordExactVectorKernel(TQ_EXACT_KERNEL_AVX2);
		return distance;
	}
#endif
#if !defined(TQ_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
	if (hnsw_tq_exact_simd_force == TQ_EXACT_SIMD_FORCE_SCALAR ||
		hnsw_tq_exact_simd_force == TQ_EXACT_SIMD_FORCE_AVX512F)
		goto scalar_exact_distance;

	if (mode == TQ_SCORE_L2)
	{
		float32x4_t acc0 = vdupq_n_f32(0);
		float32x4_t acc1 = vdupq_n_f32(0);
		float32x4_t acc2 = vdupq_n_f32(0);
		float32x4_t acc3 = vdupq_n_f32(0);
		int			i = 0;

		for (; i + 16 <= queryVector->dim; i += 16)
		{
			float32x4_t qv0 = vld1q_f32(&queryVector->x[i]);
			float32x4_t qv1 = vld1q_f32(&queryVector->x[i + 4]);
			float32x4_t qv2 = vld1q_f32(&queryVector->x[i + 8]);
			float32x4_t qv3 = vld1q_f32(&queryVector->x[i + 12]);
			float32x4_t vv0 = vld1q_f32(&valueVector->x[i]);
			float32x4_t vv1 = vld1q_f32(&valueVector->x[i + 4]);
			float32x4_t vv2 = vld1q_f32(&valueVector->x[i + 8]);
			float32x4_t vv3 = vld1q_f32(&valueVector->x[i + 12]);
			float32x4_t diff0 = vsubq_f32(qv0, vv0);
			float32x4_t diff1 = vsubq_f32(qv1, vv1);
			float32x4_t diff2 = vsubq_f32(qv2, vv2);
			float32x4_t diff3 = vsubq_f32(qv3, vv3);

			acc0 = vfmaq_f32(acc0, diff0, diff0);
			acc1 = vfmaq_f32(acc1, diff1, diff1);
			acc2 = vfmaq_f32(acc2, diff2, diff2);
			acc3 = vfmaq_f32(acc3, diff3, diff3);
		}

		acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
		for (; i + 4 <= queryVector->dim; i += 4)
		{
			float32x4_t qv = vld1q_f32(&queryVector->x[i]);
			float32x4_t vv = vld1q_f32(&valueVector->x[i]);
			float32x4_t diff = vsubq_f32(qv, vv);

			acc0 = vfmaq_f32(acc0, diff, diff);
		}

		distance = (double) vaddvq_f32(acc0);
		for (; i < queryVector->dim; i++)
		{
			double		diff = (double) queryVector->x[i] - valueVector->x[i];

			distance += diff * diff;
		}

		HnswRecordExactVectorKernel(TQ_EXACT_KERNEL_NEON);
		return distance;
	}

	if (mode == TQ_SCORE_IP || mode == TQ_SCORE_COSINE)
	{
		float32x4_t dotAcc0 = vdupq_n_f32(0);
		float32x4_t dotAcc1 = vdupq_n_f32(0);
		float32x4_t dotAcc2 = vdupq_n_f32(0);
		float32x4_t dotAcc3 = vdupq_n_f32(0);
		float32x4_t normAcc0 = vdupq_n_f32(0);
		float32x4_t normAcc1 = vdupq_n_f32(0);
		float32x4_t normAcc2 = vdupq_n_f32(0);
		float32x4_t normAcc3 = vdupq_n_f32(0);
		int			i = 0;

		for (; i + 16 <= queryVector->dim; i += 16)
		{
			float32x4_t qv0 = vld1q_f32(&queryVector->x[i]);
			float32x4_t qv1 = vld1q_f32(&queryVector->x[i + 4]);
			float32x4_t qv2 = vld1q_f32(&queryVector->x[i + 8]);
			float32x4_t qv3 = vld1q_f32(&queryVector->x[i + 12]);
			float32x4_t vv0 = vld1q_f32(&valueVector->x[i]);
			float32x4_t vv1 = vld1q_f32(&valueVector->x[i + 4]);
			float32x4_t vv2 = vld1q_f32(&valueVector->x[i + 8]);
			float32x4_t vv3 = vld1q_f32(&valueVector->x[i + 12]);

			dotAcc0 = vfmaq_f32(dotAcc0, qv0, vv0);
			dotAcc1 = vfmaq_f32(dotAcc1, qv1, vv1);
			dotAcc2 = vfmaq_f32(dotAcc2, qv2, vv2);
			dotAcc3 = vfmaq_f32(dotAcc3, qv3, vv3);
			if (mode == TQ_SCORE_COSINE)
			{
				normAcc0 = vfmaq_f32(normAcc0, vv0, vv0);
				normAcc1 = vfmaq_f32(normAcc1, vv1, vv1);
				normAcc2 = vfmaq_f32(normAcc2, vv2, vv2);
				normAcc3 = vfmaq_f32(normAcc3, vv3, vv3);
			}
		}

		dotAcc0 = vaddq_f32(vaddq_f32(dotAcc0, dotAcc1),
							vaddq_f32(dotAcc2, dotAcc3));
		if (mode == TQ_SCORE_COSINE)
			normAcc0 = vaddq_f32(vaddq_f32(normAcc0, normAcc1),
								 vaddq_f32(normAcc2, normAcc3));

		for (; i + 4 <= queryVector->dim; i += 4)
		{
			float32x4_t qv = vld1q_f32(&queryVector->x[i]);
			float32x4_t vv = vld1q_f32(&valueVector->x[i]);

			dotAcc0 = vfmaq_f32(dotAcc0, qv, vv);
			if (mode == TQ_SCORE_COSINE)
				normAcc0 = vfmaq_f32(normAcc0, vv, vv);
		}

		dot = (double) vaddvq_f32(dotAcc0);
		if (mode == TQ_SCORE_COSINE)
			valueNorm = (double) vaddvq_f32(normAcc0);

		for (; i < queryVector->dim; i++)
		{
			double		qv = queryVector->x[i];
			double		vv = valueVector->x[i];

			dot += qv * vv;
			if (mode == TQ_SCORE_COSINE)
				valueNorm += vv * vv;
		}

		if (mode == TQ_SCORE_IP)
			return -dot;

		HnswRecordExactVectorKernel(TQ_EXACT_KERNEL_NEON);
		if (so->tq.queryNorm == 0 || valueNorm == 0)
			return 1;

		return 1 - (dot / sqrt(so->tq.queryNorm * valueNorm));
	}
#endif

#if !defined(TQ_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
scalar_exact_distance:
#endif
	for (int i = 0; i < queryVector->dim; i++)
	{
		double		qv = queryVector->x[i];
		double		vv = valueVector->x[i];

		if (mode == TQ_SCORE_L1)
			distance += fabs(qv - vv);
		else if (mode == TQ_SCORE_L2)
		{
			double		diff = qv - vv;

			distance += diff * diff;
		}
		else
		{
			dot += qv * vv;
			if (mode == TQ_SCORE_COSINE)
				valueNorm += vv * vv;
		}
	}

	if (mode == TQ_SCORE_L1 || mode == TQ_SCORE_L2)
	{
		HnswRecordExactVectorKernel(TQ_EXACT_KERNEL_SCALAR);
		return distance;
	}

	if (mode == TQ_SCORE_IP)
	{
		HnswRecordExactVectorKernel(TQ_EXACT_KERNEL_SCALAR);
		return -dot;
	}

	if (mode == TQ_SCORE_COSINE)
	{
		HnswRecordExactVectorKernel(TQ_EXACT_KERNEL_SCALAR);
		if (so->tq.queryNorm == 0 || valueNorm == 0)
			return 1;

		return 1 - (dot / sqrt(so->tq.queryNorm * valueNorm));
	}

	HnswRecordExactVectorKernel(hnsw_tq_exact_simd_force == TQ_EXACT_SIMD_FORCE_SCALAR ?
								TQ_EXACT_KERNEL_SCALAR :
								TQ_EXACT_KERNEL_AUTOVEC_FMA);
	return TqGraphExactDistance(&so->support, query, PointerGetDatum(valuePtr));
}

static bool
TqGraphUseExactLowBitRouting(HnswScanOpaque so, Datum query)
{
	return DatumGetPointer(query) != NULL &&
		so->tq.enabled &&
		(TqScoreMode) so->tq.scoreMode == TQ_SCORE_L2 &&
		so->tq.bits < TQ_DEFAULT_BITS &&
		so->tq.dimensions >= 1024;
}

bool
TqGraphCachedExactNodeDistance(HnswScanOpaque so, Datum query,
							   TqGraphScanNode *node, double *distance)
{
	if (!TqGraphUseExactLowBitRouting(so, query) || node->exactVector == NULL)
		return false;

	*distance = TqGraphExactVectorDistance(so, query, node->exactVector);
	return true;
}

static bool
TqGraphUseExactHighdimL2Entry(HnswScanOpaque so, Datum query)
{
	return DatumGetPointer(query) != NULL &&
		so->tq.enabled &&
		(TqScoreMode) so->tq.scoreMode == TQ_SCORE_L2 &&
		so->tq.bits == TQ_DEFAULT_BITS &&
		so->tq.dimensions >= 1024;
}

bool
TqGraphExactHighdimEntryDistance(HnswScanOpaque so, Datum query,
								 TqGraphScanNode *node, double *distance)
{
	if (!TqGraphUseExactHighdimL2Entry(so, query) || node->exactVector == NULL)
		return false;

	*distance = TqGraphExactVectorDistance(so, query, node->exactVector);
	return true;
}

static int
TqGraphBit1PopcntRawCodes(const uint8 *a, const uint8 *b, int dim)
{
	int			fullBytes = dim / 8;
	int			tailBits = dim & 7;
	int			same = 0;

	for (int i = 0; i < fullBytes; i++)
		same += pg_number_of_ones[(uint8) ~(a[i] ^ b[i])];

	if (tailBits != 0)
	{
		uint8		mask = (uint8) ((1U << tailBits) - 1U);

		same += pg_number_of_ones[(uint8) (~(a[fullBytes] ^ b[fullBytes]) & mask)];
	}

	return (2 * same) - dim;
}

/*
 * Scalar asymmetric 1-bit query-vs-code scorer.
 *
 * Returns the centroid-space dot product computed from a bit-plane
 * decomposed 8-bit signed query (tq->queryPlanes, populated by
 * TqPrepareQueryAsymBit1) against a 1-bit packed code.  Reduction:
 *
 *   v_dot_q   = Σ_b w_b · popcount(code AND plane_b)
 *               with w_b = 2^b for b<7, -128 for b=7 (sign plane)
 *   signed_dot = 2 · v_dot_q − Σ q_signed
 *   score      = (c / q_scale) · signed_dot
 *
 * The trailing partial block (tail_bytes < 16) is processed via the
 * same scalar inner loop after copying the tail data bytes into a
 * zero-padded 16-byte scratch buffer.  Plane bytes beyond tail_bytes
 * are already zero from palloc0, so AND-popcount with the padding
 * lanes contributes 0.
 *
 * SIMD parity reference — the AVX2 / AVX-512 / NEON variants must match this byte-for-byte on equivalent inputs.
 */
#define TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL 16

static float
TqGraphAsymBit1ScalarRawScore(const HnswTqQuery *tq, const uint8 *code)
{
	const uint8 *planes = tq->queryPlanes;
	int			numFullBlocks = tq->queryAsymNumFullBlocks;
	int			tailBytes = tq->queryAsymTailBytes;
	int			BITS = tq->queryAsymBits;
	int64		vDotQ = 0;

	for (int block = 0; block < numFullBlocks; block++)
	{
		const uint8 *dataBlock = code + (Size) block * TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL;
		const uint8 *blockPlanes = planes +
			(Size) block * BITS * TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL;

		for (int b = 0; b < BITS; b++)
		{
			const uint8 *plane = blockPlanes + b * TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL;
			int			pop = 0;

			for (int i = 0; i < TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL; i++)
				pop += pg_number_of_ones[(uint8) (dataBlock[i] & plane[i])];

			if (b == BITS - 1)
				vDotQ -= (int64) (1 << (BITS - 1)) * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	if (tailBytes > 0)
	{
		uint8		dataScratch[TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL] = {0};
		const uint8 *blockPlanes = planes +
			(Size) numFullBlocks * BITS * TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL;
		const uint8 *tailSrc = code + (Size) numFullBlocks * TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL;

		memcpy(dataScratch, tailSrc, tailBytes);

		for (int b = 0; b < BITS; b++)
		{
			const uint8 *plane = blockPlanes + b * TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL;
			int			pop = 0;

			for (int i = 0; i < TQ_QUERY_ASYM_BLOCK_BYTES_LOCAL; i++)
				pop += pg_number_of_ones[(uint8) (dataScratch[i] & plane[i])];

			if (b == BITS - 1)
				vDotQ -= (int64) (1 << (BITS - 1)) * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	{
		int64		signedDot = 2 * vDotQ - tq->queryAsymSumSigned;

		return tq->queryAsymScale * (float) signedDot;
	}
}

/*
 * SIMD asymmetric 1-bit scorer.
 *
 * AVX2 path uses the nibble-lookup popcount trick (pshufb): split each
 * byte into low/high nibbles, look up popcount per nibble from a
 * 16-entry table, add.  16-byte block processed per inner-loop chunk;
 * the per-block work is `BITS=8` AND + pshufb-popcount sequences.
 *
 * NEON path uses vcntq_u8 (per-byte popcount) directly, then horizontal
 * sum via vaddvq_u8.
 *
 * Both produce bit-identical scalar output (modulo the final `float`
 * cast of the i64 signed_dot).  Verified by the parity test.
 */

#if TQ_GRAPH_COMPILE_AVX2
static float TQ_GRAPH_AVX2_TARGET
TqGraphAsymBit1Avx2RawScore(const HnswTqQuery *tq, const uint8 *code)
{
	const uint8 *planes = tq->queryPlanes;
	int			numFullBlocks = tq->queryAsymNumFullBlocks;
	int			tailBytes = tq->queryAsymTailBytes;
	int			BITS = tq->queryAsymBits;
	int64		signWeight = (int64) 1 << (BITS - 1);
	int64		vDotQ = 0;
	const __m128i nibblePopLut = _mm_setr_epi8(
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
	const __m128i nibbleMask = _mm_set1_epi8(0x0F);

	for (int block = 0; block < numFullBlocks; block++)
	{
		const uint8 *dataBlock = code + (Size) block * 16;
		const uint8 *blockPlanes = planes + (Size) block * BITS * 16;
		__m128i		data = _mm_loadu_si128((const __m128i *) dataBlock);

		for (int b = 0; b < BITS; b++)
		{
			__m128i		plane = _mm_loadu_si128((const __m128i *) (blockPlanes + b * 16));
			__m128i		v = _mm_and_si128(data, plane);
			__m128i		lo = _mm_and_si128(v, nibbleMask);
			__m128i		hi = _mm_and_si128(_mm_srli_epi16(v, 4), nibbleMask);
			__m128i		popLo = _mm_shuffle_epi8(nibblePopLut, lo);
			__m128i		popHi = _mm_shuffle_epi8(nibblePopLut, hi);
			__m128i		popByte = _mm_add_epi8(popLo, popHi);
			__m128i		popSad = _mm_sad_epu8(popByte, _mm_setzero_si128());
			int			pop = (int) (_mm_cvtsi128_si32(popSad) +
									 _mm_cvtsi128_si32(_mm_srli_si128(popSad, 8)));

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	if (tailBytes > 0)
	{
		uint8		dataScratch[16] = {0};
		const uint8 *blockPlanes = planes + (Size) numFullBlocks * BITS * 16;
		const uint8 *tailSrc = code + (Size) numFullBlocks * 16;
		__m128i		data;

		memcpy(dataScratch, tailSrc, tailBytes);
		data = _mm_loadu_si128((const __m128i *) dataScratch);

		for (int b = 0; b < BITS; b++)
		{
			__m128i		plane = _mm_loadu_si128((const __m128i *) (blockPlanes + b * 16));
			__m128i		v = _mm_and_si128(data, plane);
			__m128i		lo = _mm_and_si128(v, nibbleMask);
			__m128i		hi = _mm_and_si128(_mm_srli_epi16(v, 4), nibbleMask);
			__m128i		popLo = _mm_shuffle_epi8(nibblePopLut, lo);
			__m128i		popHi = _mm_shuffle_epi8(nibblePopLut, hi);
			__m128i		popByte = _mm_add_epi8(popLo, popHi);
			__m128i		popSad = _mm_sad_epu8(popByte, _mm_setzero_si128());
			int			pop = (int) (_mm_cvtsi128_si32(popSad) +
									 _mm_cvtsi128_si32(_mm_srli_si128(popSad, 8)));

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	{
		int64		signedDot = 2 * vDotQ - tq->queryAsymSumSigned;

		return tq->queryAsymScale * (float) signedDot;
	}
}
#endif		/* TQ_GRAPH_COMPILE_AVX2 */

#if TQ_GRAPH_COMPILE_AVX512VPOPCNTDQ
/*
 * AVX-512 VPOPCNTDQ kernel — native `_mm512_popcnt_epi8`
 * processes 64 bytes per cycle (~1 cycle throughput on Ice Lake / SPR).
 *
 * Layout: one block is 16 bytes of data and `BITS` × 16 bytes of plane.
 * For BITS = 8 the planes fit in one 128-byte chunk = two ZMM loads;
 * we process each plane as a single 16-byte load broadcast / AND with
 * data, popcount, accumulate.  This is the same per-plane work as the
 * AVX2 path but `_mm512_popcnt_epi8` replaces the pshufb nibble-lookup
 * popcount with a single uop.
 *
 * Hardware validation: compile + scalar parity only on this i7-12800H
 * (Alder Lake doesn't have AVX-512).  Runtime exercise needs an
 * Ice Lake / SPR / Tiger Lake-H35 host.
 */
static float TQ_GRAPH_AVX512VPOPCNTDQ_TARGET
TqGraphAsymBit1Avx512VpopcntdqRawScore(const HnswTqQuery *tq, const uint8 *code)
{
	const uint8 *planes = tq->queryPlanes;
	int			numFullBlocks = tq->queryAsymNumFullBlocks;
	int			tailBytes = tq->queryAsymTailBytes;
	int			BITS = tq->queryAsymBits;
	int64		signWeight = (int64) 1 << (BITS - 1);
	int64		vDotQ = 0;

	for (int block = 0; block < numFullBlocks; block++)
	{
		const uint8 *dataBlock = code + (Size) block * 16;
		const uint8 *blockPlanes = planes + (Size) block * BITS * 16;
		/* Broadcast 16-byte block into the low XMM lane of a ZMM */
		__m512i		data = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *) dataBlock));

		for (int b = 0; b < BITS; b++)
		{
			__m512i		plane = _mm512_castsi128_si512(
				_mm_loadu_si128((const __m128i *) (blockPlanes + b * 16)));
			__m512i		v = _mm512_and_si512(data, plane);
			/*
			 * VPOPCNTDQ gives per-qword popcount.  Our AND result has 16
			 * meaningful bytes (lanes 0..1 of the ZMM, lanes 2..7 are
			 * zero from the cast), so popcount(lane0) + popcount(lane1)
			 * equals the popcount of the 16-byte AND.  reduce_add_epi64
			 * sums all 8 lanes; the zero ones contribute 0.
			 */
			__m512i		popLanes = _mm512_popcnt_epi64(v);
			int			pop = (int) _mm512_reduce_add_epi64(popLanes);

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	if (tailBytes > 0)
	{
		uint8		dataScratch[16] = {0};
		const uint8 *blockPlanes = planes + (Size) numFullBlocks * BITS * 16;
		const uint8 *tailSrc = code + (Size) numFullBlocks * 16;
		__m512i		data;

		memcpy(dataScratch, tailSrc, tailBytes);
		data = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *) dataScratch));

		for (int b = 0; b < BITS; b++)
		{
			__m512i		plane = _mm512_castsi128_si512(
				_mm_loadu_si128((const __m128i *) (blockPlanes + b * 16)));
			__m512i		v = _mm512_and_si512(data, plane);
			/*
			 * VPOPCNTDQ gives per-qword popcount.  Our AND result has 16
			 * meaningful bytes (lanes 0..1 of the ZMM, lanes 2..7 are
			 * zero from the cast), so popcount(lane0) + popcount(lane1)
			 * equals the popcount of the 16-byte AND.  reduce_add_epi64
			 * sums all 8 lanes; the zero ones contribute 0.
			 */
			__m512i		popLanes = _mm512_popcnt_epi64(v);
			int			pop = (int) _mm512_reduce_add_epi64(popLanes);

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	{
		int64		signedDot = 2 * vDotQ - tq->queryAsymSumSigned;

		return tq->queryAsymScale * (float) signedDot;
	}
}

static bool
TqGraphAvx512VpopcntdqAvailable(void)
{
	static int	available = -1;

	/*
	 * GUC kill-switch lets users force the dispatcher
	 * to fall through to the AVX2 kernel even on hosts that have
	 * VPOPCNTDQ.  Useful for parity testing and downclock measurement.
	 * Checked first so the runtime feature probe still memoizes only
	 * the hardware capability — flipping the GUC doesn't require
	 * resetting `available`.
	 */
	if (!hnsw_tq_graph_avx512vpopcntdq)
		return false;
	if (hnsw_tq_simd_force != TQ_SIMD_FORCE_AUTO &&
		hnsw_tq_simd_force != TQ_SIMD_FORCE_AVX512VNNI)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVX512VPOPCNTDQ__)
	available = 1;
#elif TQ_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))
	/*
	 * __builtin_cpu_supports("avx512vpopcntdq") needs GCC 8+/Clang 8+;
	 * both gates the parent COMPILE flag already requires.
	 */
	available = __builtin_cpu_supports("avx512vpopcntdq") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}
#endif		/* TQ_GRAPH_COMPILE_AVX512VPOPCNTDQ */

#if defined(__aarch64__) || defined(_M_ARM64)
static float
TqGraphAsymBit1NeonRawScore(const HnswTqQuery *tq, const uint8 *code)
{
	const uint8 *planes = tq->queryPlanes;
	int			numFullBlocks = tq->queryAsymNumFullBlocks;
	int			tailBytes = tq->queryAsymTailBytes;
	int			BITS = tq->queryAsymBits;
	int64		signWeight = (int64) 1 << (BITS - 1);
	int64		vDotQ = 0;

	for (int block = 0; block < numFullBlocks; block++)
	{
		const uint8 *dataBlock = code + (Size) block * 16;
		const uint8 *blockPlanes = planes + (Size) block * BITS * 16;
		uint8x16_t	data = vld1q_u8(dataBlock);

		for (int b = 0; b < BITS; b++)
		{
			uint8x16_t	plane = vld1q_u8(blockPlanes + b * 16);
			uint8x16_t	v = vandq_u8(data, plane);
			uint8x16_t	popByte = vcntq_u8(v);
			int			pop = (int) vaddvq_u8(popByte);

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	if (tailBytes > 0)
	{
		uint8		dataScratch[16] = {0};
		const uint8 *blockPlanes = planes + (Size) numFullBlocks * BITS * 16;
		const uint8 *tailSrc = code + (Size) numFullBlocks * 16;
		uint8x16_t	data;

		memcpy(dataScratch, tailSrc, tailBytes);
		data = vld1q_u8(dataScratch);

		for (int b = 0; b < BITS; b++)
		{
			uint8x16_t	plane = vld1q_u8(blockPlanes + b * 16);
			uint8x16_t	v = vandq_u8(data, plane);
			uint8x16_t	popByte = vcntq_u8(v);
			int			pop = (int) vaddvq_u8(popByte);

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	{
		int64		signedDot = 2 * vDotQ - tq->queryAsymSumSigned;

		return tq->queryAsymScale * (float) signedDot;
	}
}
#endif		/* aarch64 */

/*
 * Asymmetric 1-bit dispatch: pick the fastest available kernel.  AVX2 on amd64
 * (runtime feature gated alongside the rest of the AMD64 SIMD knobs),
 * NEON on aarch64, scalar reference fallback elsewhere.
 *
 * The runtime AVX2 gate uses __builtin_cpu_supports("avx2") same as
 * the existing 1-bit + 2-bit fast paths.  When AVX2 is disabled
 * via the existing hnsw.tq_graph_i8mm / similar paths, we still take
 * the AVX2 popcount kernel — those GUCs gate VNNI / SDOT, not basic
 * SIMD popcount.
 */
static float
TqGraphAsymBit1Score(const HnswTqQuery *tq, const uint8 *code)
{
#if TQ_GRAPH_COMPILE_AVX512VPOPCNTDQ
	if (TqGraphAvx512VpopcntdqAvailable())
		return TqGraphAsymBit1Avx512VpopcntdqRawScore(tq, code);
#endif
#if TQ_GRAPH_COMPILE_AVX2
	if (TqGraphAvx2Available())
		return TqGraphAsymBit1Avx2RawScore(tq, code);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
	return TqGraphAsymBit1NeonRawScore(tq, code);
#endif
	return TqGraphAsymBit1ScalarRawScore(tq, code);
}

static bool
TqGraphBuildQueryBit1PopcntDistance(TqGraphBuildState *state,
									TqGraphBuildNode *node,
									double *distance)
{
	HnswTqQuery *tq = &state->buildTq;
	double		dimSqrt;
	double		dot;
	double		center;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if (!hnsw_tq_graph_lowbit_popcnt || !state->buildTqValid ||
		tq->bits != 1 || tq->querySignBits == NULL || node->code == NULL ||
		tq->dimensions <= 0 || mode == TQ_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) tq->dimensions);
	center = TqGraphCodeCenter(1, 1);
	dot = tq->ecCorrection +
		(center * (double) TqGraphBit1PopcntRawCodes(tq->querySignBits,
													 node->code,
													 tq->dimensions));

	if (mode == TQ_SCORE_IP)
		*distance = -(node->scale * dot / dimSqrt);
	else if (mode == TQ_SCORE_COSINE)
	{
		if (tq->queryNorm == 0 || node->scale == 0)
			*distance = 1;
		else
			*distance = 1 - (dot / (sqrt(tq->queryNorm) * dimSqrt));
	}
	else
	{
		*distance = tq->queryNorm + ((double) node->scale * node->scale) -
			(2 * node->scale * dot / dimSqrt);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

static bool
TqGraphBuildCodeCodeDistanceBits1Popcnt(TqGraphBuildState *state,
										uint32 a, uint32 b,
										double *distance)
{
	TqGraphBuildNode *aNode;
	TqGraphBuildNode *bNode;
	double		dot;
	double		avg;
	double		center;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if (!hnsw_tq_graph_lowbit_popcnt || state->tqBits != 1 ||
		state->dimensions <= 0 || mode == TQ_SCORE_L1)
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	center = TqGraphCodeCenter(1, 1);
	dot = center * center *
		(double) TqGraphBit1PopcntRawCodes(aNode->code, bNode->code,
										   state->dimensions);
	avg = dot / (double) state->dimensions;

	if (mode == TQ_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale * avg);
	else if (mode == TQ_SCORE_COSINE)
		*distance = 1 - (avg / (center * center));
	else
	{
		*distance = ((double) aNode->scale * aNode->scale) +
			((double) bNode->scale * bNode->scale) -
			(2 * (double) aNode->scale * (double) bNode->scale * avg);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

static bool
TqGraphTryBuildQueryDistance(TqGraphBuildState *state, uint32 a, uint32 b,
							 double *distance)
{
	if (!state->buildTqValid || a != state->buildQueryNodeId ||
		state->nodes[b].code == NULL)
		return false;

	if (TqGraphBuildQueryBit1PopcntDistance(state, &state->nodes[b], distance))
	{
		state->buildDistanceQuerySplit++;
		return true;
	}
#if TQ_GRAPH_COMPILE_QUERY_SPLIT
	if (TqGraphPackedDistanceQuerySplit4(&state->buildTq, state->nodes[b].code,
									 state->nodes[b].scale, distance))
	{
		state->buildDistanceQuerySplit++;
		return true;
	}
	if (TqGraphPackedDistanceQuerySplit2(&state->buildTq, state->nodes[b].code,
									 state->nodes[b].scale, distance))
	{
		state->buildDistanceQuerySplit++;
		return true;
	}
#endif

	state->buildDistancePacked++;
	*distance = TqGraphPackedDistance(&state->buildTq, state->nodes[b].code,
									 state->nodes[b].scale,
									 state->nodes[b].correction,
									 state->nodes[b].norm);
	return true;
}

static bool
TqGraphTryBuildCodeCodeDistance(TqGraphBuildState *state, uint32 a, uint32 b,
								 double *distance)
{
	/*
	 * TQ+ weighted symmetric scoring takes precedence over the unweighted
	 * code-code path when the index is built with tq_weighted = on AND the
	 * user-set GUC permits it.  Cosine only for now; L2/IP land in a follow-up.
	 */
	if (TqGraphBuildCodeCodeWeighted(state, a, b, distance))
	{
		state->buildDistanceWeighted++;
		return true;
	}

#if TQ_GRAPH_COMPILE_QUERY_SPLIT
	if (TqGraphBuildCodeCodeDistance4(state, a, b, distance))
	{
		state->buildDistanceCodeCode++;
		return true;
	}
	if (TqGraphBuildCodeCodeDistance2(state, a, b, distance))
	{
		state->buildDistanceCodeCode++;
		return true;
	}
#endif
	if (TqGraphBuildCodeCodeDistanceBits1Popcnt(state, a, b, distance))
	{
		state->buildDistanceCodeCode++;
		return true;
	}
	if (TqGraphBuildCodeCodeDistanceScalar(state, a, b, distance))
	{
		state->buildDistanceCodeCode++;
		return true;
	}

	return false;
}

bool
TqGraphCodeCodeDistance(HnswScanOpaque so, HnswMetaPageData *meta,
						TqGraphScanNode *aNode, TqGraphScanNode *bNode,
						double *distance)
{
	TqGraphBuildState state;
	TqGraphBuildNode nodes[2];

	if (so == NULL || !so->tq.enabled || meta == NULL ||
		aNode == NULL || bNode == NULL ||
		aNode->code == NULL || bNode->code == NULL)
		return false;

	memset(&state, 0, sizeof(state));
	memset(nodes, 0, sizeof(nodes));
	state.nodes = nodes;
	state.nodeCount = 2;
	state.dimensions = meta->dimensions;
	state.tqBits = meta->tqBits;
	state.tqWeighted = (meta->tqFlags & TQ_GRAPH_TQ_WEIGHTED) != 0;
	state.scoreMode = so->tq.scoreMode;
	state.ecShift = so->tq.ecShift;
	state.ecScale = so->tq.ecScale;
	if (state.ecShift != NULL)
		state.mmConst = TqGraphMmConstScalar(state.ecShift, state.dimensions);

	nodes[0].code = aNode->code;
	nodes[0].scale = aNode->scale;
	nodes[0].norm = aNode->norm;
	nodes[0].correction = aNode->codeNorm;
	nodes[0].ecCorrection = aNode->ecCorrection;
	nodes[1].code = bNode->code;
	nodes[1].scale = bNode->scale;
	nodes[1].norm = bNode->norm;
	nodes[1].correction = bNode->codeNorm;
	nodes[1].ecCorrection = bNode->ecCorrection;

	return TqGraphTryBuildCodeCodeDistance(&state, 0, 1, distance);
}

static double
TqGraphBuildExactVectorDistance(TqGraphBuildState *state, uint32 a, uint32 b)
{
	Vector	   *av = state->nodes[a].vector;
	Vector	   *bv = state->nodes[b].vector;
	TqScoreMode mode;
	double		distance = 0;
	double		dot = 0;
	double		aNorm = 0;
	double		bNorm = 0;

	if (av == NULL || bv == NULL || av->dim != bv->dim)
	{
		state->buildDistanceFallback++;
		return TqGraphExactDistance(&state->support,
								PointerGetDatum(av),
								PointerGetDatum(bv));
	}

	state->buildDistanceExact++;

	mode = (TqScoreMode) state->scoreMode;
#if TQ_GRAPH_COMPILE_AVX2
	if (TqGraphAvx2Available() &&
		TqGraphBuildExactDistanceAvx2(state, av, bv, &distance))
		return distance;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
	if (mode == TQ_SCORE_L2)
	{
		float32x4_t acc0 = vdupq_n_f32(0);
		float32x4_t acc1 = vdupq_n_f32(0);
		float32x4_t acc2 = vdupq_n_f32(0);
		float32x4_t acc3 = vdupq_n_f32(0);
		int			i = 0;

		for (; i + 16 <= av->dim; i += 16)
		{
			float32x4_t av0 = vld1q_f32(&av->x[i]);
			float32x4_t av1 = vld1q_f32(&av->x[i + 4]);
			float32x4_t av2 = vld1q_f32(&av->x[i + 8]);
			float32x4_t av3 = vld1q_f32(&av->x[i + 12]);
			float32x4_t bv0 = vld1q_f32(&bv->x[i]);
			float32x4_t bv1 = vld1q_f32(&bv->x[i + 4]);
			float32x4_t bv2 = vld1q_f32(&bv->x[i + 8]);
			float32x4_t bv3 = vld1q_f32(&bv->x[i + 12]);
			float32x4_t diff0 = vsubq_f32(av0, bv0);
			float32x4_t diff1 = vsubq_f32(av1, bv1);
			float32x4_t diff2 = vsubq_f32(av2, bv2);
			float32x4_t diff3 = vsubq_f32(av3, bv3);

			acc0 = vfmaq_f32(acc0, diff0, diff0);
			acc1 = vfmaq_f32(acc1, diff1, diff1);
			acc2 = vfmaq_f32(acc2, diff2, diff2);
			acc3 = vfmaq_f32(acc3, diff3, diff3);
		}

		acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
		for (; i + 4 <= av->dim; i += 4)
		{
			float32x4_t avv = vld1q_f32(&av->x[i]);
			float32x4_t bvv = vld1q_f32(&bv->x[i]);
			float32x4_t diff = vsubq_f32(avv, bvv);

			acc0 = vfmaq_f32(acc0, diff, diff);
		}

		distance = (double) vaddvq_f32(acc0);
		for (; i < av->dim; i++)
		{
			double		diff = (double) av->x[i] - bv->x[i];

			distance += diff * diff;
		}

		return distance;
	}

	if (mode == TQ_SCORE_IP || mode == TQ_SCORE_COSINE)
	{
		float32x4_t dotAcc0 = vdupq_n_f32(0);
		float32x4_t dotAcc1 = vdupq_n_f32(0);
		float32x4_t dotAcc2 = vdupq_n_f32(0);
		float32x4_t dotAcc3 = vdupq_n_f32(0);
		float32x4_t aNormAcc0 = vdupq_n_f32(0);
		float32x4_t aNormAcc1 = vdupq_n_f32(0);
		float32x4_t aNormAcc2 = vdupq_n_f32(0);
		float32x4_t aNormAcc3 = vdupq_n_f32(0);
		float32x4_t bNormAcc0 = vdupq_n_f32(0);
		float32x4_t bNormAcc1 = vdupq_n_f32(0);
		float32x4_t bNormAcc2 = vdupq_n_f32(0);
		float32x4_t bNormAcc3 = vdupq_n_f32(0);
		int			i = 0;

		for (; i + 16 <= av->dim; i += 16)
		{
			float32x4_t av0 = vld1q_f32(&av->x[i]);
			float32x4_t av1 = vld1q_f32(&av->x[i + 4]);
			float32x4_t av2 = vld1q_f32(&av->x[i + 8]);
			float32x4_t av3 = vld1q_f32(&av->x[i + 12]);
			float32x4_t bv0 = vld1q_f32(&bv->x[i]);
			float32x4_t bv1 = vld1q_f32(&bv->x[i + 4]);
			float32x4_t bv2 = vld1q_f32(&bv->x[i + 8]);
			float32x4_t bv3 = vld1q_f32(&bv->x[i + 12]);

			dotAcc0 = vfmaq_f32(dotAcc0, av0, bv0);
			dotAcc1 = vfmaq_f32(dotAcc1, av1, bv1);
			dotAcc2 = vfmaq_f32(dotAcc2, av2, bv2);
			dotAcc3 = vfmaq_f32(dotAcc3, av3, bv3);
			if (mode == TQ_SCORE_COSINE)
			{
				aNormAcc0 = vfmaq_f32(aNormAcc0, av0, av0);
				aNormAcc1 = vfmaq_f32(aNormAcc1, av1, av1);
				aNormAcc2 = vfmaq_f32(aNormAcc2, av2, av2);
				aNormAcc3 = vfmaq_f32(aNormAcc3, av3, av3);
				bNormAcc0 = vfmaq_f32(bNormAcc0, bv0, bv0);
				bNormAcc1 = vfmaq_f32(bNormAcc1, bv1, bv1);
				bNormAcc2 = vfmaq_f32(bNormAcc2, bv2, bv2);
				bNormAcc3 = vfmaq_f32(bNormAcc3, bv3, bv3);
			}
		}

		dotAcc0 = vaddq_f32(vaddq_f32(dotAcc0, dotAcc1),
							vaddq_f32(dotAcc2, dotAcc3));
		if (mode == TQ_SCORE_COSINE)
		{
			aNormAcc0 = vaddq_f32(vaddq_f32(aNormAcc0, aNormAcc1),
								  vaddq_f32(aNormAcc2, aNormAcc3));
			bNormAcc0 = vaddq_f32(vaddq_f32(bNormAcc0, bNormAcc1),
								  vaddq_f32(bNormAcc2, bNormAcc3));
		}

		for (; i + 4 <= av->dim; i += 4)
		{
			float32x4_t avv = vld1q_f32(&av->x[i]);
			float32x4_t bvv = vld1q_f32(&bv->x[i]);

			dotAcc0 = vfmaq_f32(dotAcc0, avv, bvv);
			if (mode == TQ_SCORE_COSINE)
			{
				aNormAcc0 = vfmaq_f32(aNormAcc0, avv, avv);
				bNormAcc0 = vfmaq_f32(bNormAcc0, bvv, bvv);
			}
		}

		dot = (double) vaddvq_f32(dotAcc0);
		if (mode == TQ_SCORE_COSINE)
		{
			aNorm = (double) vaddvq_f32(aNormAcc0);
			bNorm = (double) vaddvq_f32(bNormAcc0);
		}

		for (; i < av->dim; i++)
		{
			double		aval = av->x[i];
			double		bval = bv->x[i];

			dot += aval * bval;
			if (mode == TQ_SCORE_COSINE)
			{
				aNorm += aval * aval;
				bNorm += bval * bval;
			}
		}

		if (mode == TQ_SCORE_IP)
			return -dot;
		if (aNorm == 0 || bNorm == 0)
			return 1;

		return 1 - (dot / sqrt(aNorm * bNorm));
	}
#endif

	for (int i = 0; i < av->dim; i++)
	{
		double		aval = av->x[i];
		double		bval = bv->x[i];

		if (mode == TQ_SCORE_L1)
			distance += fabs(aval - bval);
		else if (mode == TQ_SCORE_L2)
		{
			double		diff = aval - bval;

			distance += diff * diff;
		}
		else
		{
			dot += aval * bval;
			if (mode == TQ_SCORE_COSINE)
			{
				aNorm += aval * aval;
				bNorm += bval * bval;
			}
		}
	}

	if (mode == TQ_SCORE_L1 || mode == TQ_SCORE_L2)
		return distance;
	if (mode == TQ_SCORE_IP)
		return -dot;
	if (mode == TQ_SCORE_COSINE)
	{
		if (aNorm == 0 || bNorm == 0)
			return 1;

		return 1 - (dot / sqrt(aNorm * bNorm));
	}

	return TqGraphExactDistance(&state->support,
							PointerGetDatum(av),
							PointerGetDatum(bv));
}

double
TqGraphBuildDistance(TqGraphBuildState *state, uint32 a, uint32 b)
{
	double		distance;

	state->buildDistanceCalls++;
	if (a >= state->nodeCount || b >= state->nodeCount)
		return DBL_MAX;

	/*
	 * When requested, short-circuit the quantized fast paths and route every
	 * build-time pruning call through exact f32 distance.  This locks in a
	 * graph topology built with perfect distances; scan-time scoring still
	 * uses the packed codes.
	 */
	if (state->buildExactDistances && state->nodes[a].vector != NULL &&
		state->nodes[b].vector != NULL &&
		state->nodes[a].vector->dim == state->nodes[b].vector->dim)
		return TqGraphBuildExactVectorDistance(state, a, b);

	if (TqGraphTryBuildQueryDistance(state, a, b, &distance))
		return distance;
	if (TqGraphTryBuildCodeCodeDistance(state, a, b, &distance))
		return distance;

	return TqGraphBuildExactVectorDistance(state, a, b);
}

void
TqGraphPrepareBuildQuery(TqGraphBuildState *state, uint32 nodeId)
{
	MemoryContext oldctx;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	state->buildTqValid = false;
	if ((mode != TQ_SCORE_L2 && mode != TQ_SCORE_COSINE && mode != TQ_SCORE_IP) ||
		(state->tqBits != 1 && state->tqBits != 2 &&
		 state->tqBits != TQ_DEFAULT_BITS) ||
		nodeId >= state->nodeCount ||
		state->nodes[nodeId].vector == NULL || state->buildQueryCtx == NULL)
		return;

	MemoryContextReset(state->buildQueryCtx);
	oldctx = MemoryContextSwitchTo(state->buildQueryCtx);
	if (state->ecShift != NULL && state->ecScale != NULL)
		HnswPrepareTqBuildQueryWithCorrection(state->index, &state->support,
											  PointerGetDatum(state->nodes[nodeId].vector),
											  &state->buildTq,
											  state->ecShift,
											  state->ecScale);
	else
		HnswPrepareTqBuildQuery(state->index, &state->support,
								PointerGetDatum(state->nodes[nodeId].vector),
								&state->buildTq);
	MemoryContextSwitchTo(oldctx);

	state->buildQueryNodeId = nodeId;
	state->buildTqValid = state->buildTq.enabled &&
		state->buildTq.scoreMode == state->scoreMode;
}
double
TqGraphScoreNode(HnswScanOpaque so, TqGraphScanNode *node)
{
	if (!so->tq.enabled || node->code == NULL)
		return 0;

	so->graphScoredCodes++;
	so->graphScalarScoredCodes++;

	/*
	 * Asymmetric 1-bit single-node fast path.  Mirrors the
	 * batch dispatch slot above so tail nodes (< 4 left over) get the
	 * same scoring math as the batch-of-4 path.  Falls through to
	 * TqGraphPackedDistance when the GUC is off, queryPlanes is NULL,
	 * the bit-width isn't 1, the score mode is L1, or the node lacks a
	 * packed code.
	 */
	if (hnsw_tq_query_1bit_asymmetric && so->tq.bits == 1 &&
		so->tq.queryPlanes != NULL &&
		(TqScoreMode) so->tq.scoreMode != TQ_SCORE_L1)
	{
		TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
		double		dimSqrt = sqrt((double) so->tq.dimensions);
		double		dot;

		dot = so->tq.ecCorrection +
			(double) TqGraphAsymBit1Score(&so->tq, node->code);

		if (mode == TQ_SCORE_IP)
			return -(node->scale * dot / dimSqrt);
		if (mode == TQ_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				return 1;
			return 1 - (dot / (sqrt(so->tq.queryNorm) * dimSqrt));
		}
		/* TQ_SCORE_L2 */
		{
			double		distance = so->tq.queryNorm +
				((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);

			return distance < 0 ? 0 : distance;
		}
	}

	return TqGraphPackedDistance(&so->tq, node->code, node->scale,
								 node->codeNorm, node->norm);
}

#if TQ_GRAPH_COMPILE_ARM_DOT
static bool
TqGraphArmDotprodAvailable(void)
{
	static int	available = -1;

	if (hnsw_tq_simd_force != TQ_SIMD_FORCE_AUTO &&
		hnsw_tq_simd_force != TQ_SIMD_FORCE_ARM_SDOT &&
		hnsw_tq_simd_force != TQ_SIMD_FORCE_NEON)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__APPLE__)
	{
		int			value = 0;
		size_t		len = sizeof(value);

		if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &len,
						 NULL, 0) == 0)
		{
			available = value != 0;
			return available != 0;
		}
	}
	available = 1;
#elif defined(__linux__) && defined(HWCAP_ASIMDDP)
	available = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
	available = 0;
#endif
	return available != 0;
}

#if TQ_GRAPH_COMPILE_ARM_I8MM
static bool
TqGraphArmI8mmAvailable(void)
{
	static int	available = -1;

	if (!hnsw_tq_graph_i8mm)
		return false;
	if (hnsw_tq_simd_force != TQ_SIMD_FORCE_AUTO &&
		hnsw_tq_simd_force != TQ_SIMD_FORCE_ARM_I8MM)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__APPLE__)
	{
		int			value = 0;
		size_t		len = sizeof(value);

		if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &value, &len,
						 NULL, 0) == 0)
		{
			available = value != 0;
			return available != 0;
		}
	}
	available = 0;
#elif defined(__linux__) && defined(HWCAP2_I8MM)
	available = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
	available = 0;
#endif
	return available != 0;
}

static int32x4_t TQ_GRAPH_ARM_I8MM_TARGET
TqGraphDotI8x16ArmI8mm(int32x4_t acc, int8x16_t a, int8x16_t b)
{
	uint8x16_t	aUnsigned = vreinterpretq_u8_s8(veorq_s8(a, vdupq_n_s8((int8) 0x80)));
	int16x8_t	pairSums = vpaddlq_s8(b);
	int32x4_t	groupSums = vpaddlq_s16(pairSums);

	acc = vusdotq_s32(acc, aUnsigned, b);
	return vsubq_s32(acc, vmulq_n_s32(groupSums, 128));
}
#define TQ_GRAPH_ARM_DOT(acc, a, b) \
	(useI8mm ? TqGraphDotI8x16ArmI8mm((acc), (a), (b)) : vdotq_s32((acc), (a), (b)))
#else
#define TQ_GRAPH_ARM_DOT(acc, a, b) vdotq_s32((acc), (a), (b))
#endif

static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphQuerySplitRawNeonSdot(const HnswTqQuery *tq, const uint8 *code)
{
	int32x4_t	accLow0 = vdupq_n_s32(0);
	int32x4_t	accLow1 = vdupq_n_s32(0);
	int32x4_t	accHigh0 = vdupq_n_s32(0);
	int32x4_t	accHigh1 = vdupq_n_s32(0);
	uint8x16_t	mask = vdupq_n_u8(0x0f);
	int8x16_t	codebook = vld1q_s8((const int8_t *) TqGraphCodebookI8);
	int			pairs = tq->querySplitChunks / 2;
#if TQ_GRAPH_COMPILE_ARM_I8MM
	bool		useI8mm = TqGraphArmI8mmAvailable();
#endif

	for (int pair = 0; pair < pairs; pair++)
	{
		uint8x16_t packed = vld1q_u8(code + pair * 16);
		uint8x16_t loNibbles = vandq_u8(packed, mask);
		uint8x16_t hiNibbles = vshrq_n_u8(packed, 4);
		uint8x16_t idx0 = vzip1q_u8(loNibbles, hiNibbles);
		uint8x16_t idx1 = vzip2q_u8(loNibbles, hiNibbles);
		int8x16_t	c0 = vqtbl1q_s8(codebook, idx0);
		int8x16_t	c1 = vqtbl1q_s8(codebook, idx1);
		const int8 *low = tq->querySplitLow + pair * 32;
		const int8 *high = tq->querySplitHigh + pair * 32;

		accLow0 = TQ_GRAPH_ARM_DOT(accLow0, vld1q_s8((const int8_t *) low), c0);
		accLow1 = TQ_GRAPH_ARM_DOT(accLow1, vld1q_s8((const int8_t *) (low + 16)), c1);
		accHigh0 = TQ_GRAPH_ARM_DOT(accHigh0, vld1q_s8((const int8_t *) high), c0);
		accHigh1 = TQ_GRAPH_ARM_DOT(accHigh1, vld1q_s8((const int8_t *) (high + 16)), c1);
	}

	if ((tq->querySplitChunks & 1) != 0)
	{
		int			chunk = tq->querySplitChunks - 1;
		uint8x8_t	packed = vld1_u8(code + chunk * 8);
		uint8x8_t	loNibbles = vand_u8(packed, vdup_n_u8(0x0f));
		uint8x8_t	hiNibbles = vshr_n_u8(packed, 4);
		uint8x16_t	idx = vcombine_u8(vzip1_u8(loNibbles, hiNibbles),
									   vzip2_u8(loNibbles, hiNibbles));
		int8x16_t	c = vqtbl1q_s8(codebook, idx);
		const int8 *low = tq->querySplitLow + chunk * 16;
		const int8 *high = tq->querySplitHigh + chunk * 16;

		accLow0 = TQ_GRAPH_ARM_DOT(accLow0, vld1q_s8((const int8_t *) low), c);
		accHigh0 = TQ_GRAPH_ARM_DOT(accHigh0, vld1q_s8((const int8_t *) high), c);
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		const uint8 *tail = code + tq->querySplitChunks * 8;
		uint8x8_t	packed;
		uint8x8_t	loNibbles;
		uint8x8_t	hiNibbles;
		uint8x16_t	idx;
		int8x16_t	c;

		memcpy(scratch, tail, tailBytes);
		packed = vld1_u8(scratch);
		loNibbles = vand_u8(packed, vdup_n_u8(0x0f));
		hiNibbles = vshr_n_u8(packed, 4);
		idx = vcombine_u8(vzip1_u8(loNibbles, hiNibbles),
						  vzip2_u8(loNibbles, hiNibbles));
		c = vqtbl1q_s8(codebook, idx);

		accLow0 = TQ_GRAPH_ARM_DOT(accLow0,
								   vld1q_s8((const int8_t *) tq->querySplitTailLow), c);
		accHigh0 = TQ_GRAPH_ARM_DOT(accHigh0,
									vld1q_s8((const int8_t *) tq->querySplitTailHigh), c);
	}

	accLow0 = vaddq_s32(accLow0, accLow1);
	accHigh0 = vaddq_s32(accHigh0, accHigh1);
	return (int64) vaddvq_s32(accLow0) +
		(int64) TQ_QUERY_SPLIT_HIGH_COEF * (int64) vaddvq_s32(accHigh0);
}

static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphQuerySplit2RawNeonSdot(const HnswTqQuery *tq, const uint8 *code)
{
	int32x4_t	accLow = vdupq_n_s32(0);
	int32x4_t	accHigh = vdupq_n_s32(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) TqGraphCodebook2I8);
	uint8x16_t	mask = vdupq_n_u8(0x03);
	int8x16_t	shifts = {
		0, -2, -4, -6, 0, -2, -4, -6,
		0, -2, -4, -6, 0, -2, -4, -6
	};
#if TQ_GRAPH_COMPILE_ARM_I8MM
	bool		useI8mm = TqGraphArmI8mmAvailable();
#endif

	for (int chunk = 0; chunk < tq->querySplitChunks; chunk++)
	{
		uint8		repeatedBytes[16];
		const uint8 *bytes = code + chunk * 4;
		uint8x16_t	repeated;
		uint8x16_t	idx;
		int8x16_t	c;
		const int8 *low = tq->querySplitLow + chunk * 16;
		const int8 *high = tq->querySplitHigh + chunk * 16;

		for (int i = 0; i < 16; i++)
			repeatedBytes[i] = bytes[i / 4];

		repeated = vld1q_u8(repeatedBytes);
		idx = vandq_u8(vshlq_u8(repeated, shifts), mask);
		c = vqtbl1q_s8(codebook, idx);

		accLow = TQ_GRAPH_ARM_DOT(accLow, vld1q_s8((const int8_t *) low), c);
		accHigh = TQ_GRAPH_ARM_DOT(accHigh, vld1q_s8((const int8_t *) high), c);
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		repeatedBytes[16] = {0};
		const uint8 *bytes = code + tq->querySplitChunks * 4;
		uint8x16_t	repeated;
		uint8x16_t	idx;
		int8x16_t	c;

		for (int i = 0; i < tq->querySplitTailDims; i++)
			repeatedBytes[i] = bytes[i / 4];

		repeated = vld1q_u8(repeatedBytes);
		idx = vandq_u8(vshlq_u8(repeated, shifts), mask);
		c = vqtbl1q_s8(codebook, idx);

		accLow = TQ_GRAPH_ARM_DOT(accLow,
								  vld1q_s8((const int8_t *) tq->querySplitTailLow), c);
		accHigh = TQ_GRAPH_ARM_DOT(accHigh,
								   vld1q_s8((const int8_t *) tq->querySplitTailHigh), c);
	}

	return (int64) vaddvq_s32(accLow) +
		(int64) TQ_QUERY_SPLIT_HIGH_COEF * (int64) vaddvq_s32(accHigh);
}

static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphCodeCodeRawNeonSdot(const uint8 *a, const uint8 *b, int dim,
						   int *sampleDims)
{
	int32x4_t	acc0 = vdupq_n_s32(0);
	int32x4_t	acc1 = vdupq_n_s32(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) TqGraphCodebookI8);
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
#if TQ_GRAPH_COMPILE_ARM_I8MM
	bool		useI8mm = TqGraphArmI8mmAvailable();
#endif

	if (chunks > TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		uint8x8_t	pa = vld1_u8(a + chunk * 8);
		uint8x8_t	pb = vld1_u8(b + chunk * 8);
		uint8x8_t	loA = vand_u8(pa, vdup_n_u8(0x0f));
		uint8x8_t	hiA = vshr_n_u8(pa, 4);
		uint8x8_t	loB = vand_u8(pb, vdup_n_u8(0x0f));
		uint8x8_t	hiB = vshr_n_u8(pb, 4);
		uint8x16_t	idxA = vcombine_u8(vzip1_u8(loA, hiA),
										vzip2_u8(loA, hiA));
		uint8x16_t	idxB = vcombine_u8(vzip1_u8(loB, hiB),
										vzip2_u8(loB, hiB));
		int8x16_t	ca = vqtbl1q_s8(codebook, idxA);
		int8x16_t	cb = vqtbl1q_s8(codebook, idxB);

		acc0 = TQ_GRAPH_ARM_DOT(acc0, ca, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		uint8x8_t	pa;
		uint8x8_t	pb;
		uint8x8_t	loA;
		uint8x8_t	hiA;
		uint8x8_t	loB;
		uint8x8_t	hiB;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		pa = vld1_u8(scratchA);
		pb = vld1_u8(scratchB);
		loA = vand_u8(pa, vdup_n_u8(0x0f));
		hiA = vshr_n_u8(pa, 4);
		loB = vand_u8(pb, vdup_n_u8(0x0f));
		hiB = vshr_n_u8(pb, 4);
		idxA = vcombine_u8(vzip1_u8(loA, hiA), vzip2_u8(loA, hiA));
		idxB = vcombine_u8(vzip1_u8(loB, hiB), vzip2_u8(loB, hiB));
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc1 = TQ_GRAPH_ARM_DOT(acc1, ca, cb);
		*sampleDims += tailDims;
	}

	return (int64) vaddvq_s32(acc0) + (int64) vaddvq_s32(acc1);
}

static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphCodeCode2RawNeonSdot(const uint8 *a, const uint8 *b, int dim,
							int *sampleDims)
{
	int32x4_t	acc0 = vdupq_n_s32(0);
	int32x4_t	acc1 = vdupq_n_s32(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) TqGraphCodebook2I8);
	uint8x16_t	mask = vdupq_n_u8(0x03);
	int8x16_t	shifts = {
		0, -2, -4, -6, 0, -2, -4, -6,
		0, -2, -4, -6, 0, -2, -4, -6
	};
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
#if TQ_GRAPH_COMPILE_ARM_I8MM
	bool		useI8mm = TqGraphArmI8mmAvailable();
#endif

	if (chunks > TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = TQ_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		uint8		repeatedA[16];
		uint8		repeatedB[16];
		const uint8 *bytesA = a + chunk * 4;
		const uint8 *bytesB = b + chunk * 4;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		for (int i = 0; i < 16; i++)
		{
			repeatedA[i] = bytesA[i / 4];
			repeatedB[i] = bytesB[i / 4];
		}

		idxA = vandq_u8(vshlq_u8(vld1q_u8(repeatedA), shifts), mask);
		idxB = vandq_u8(vshlq_u8(vld1q_u8(repeatedB), shifts), mask);
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc0 = TQ_GRAPH_ARM_DOT(acc0, ca, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		repeatedA[16] = {0};
		uint8		repeatedB[16] = {0};
		const uint8 *bytesA = a + chunks * 4;
		const uint8 *bytesB = b + chunks * 4;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		for (int i = 0; i < tailDims; i++)
		{
			repeatedA[i] = bytesA[i / 4];
			repeatedB[i] = bytesB[i / 4];
		}

		idxA = vandq_u8(vshlq_u8(vld1q_u8(repeatedA), shifts), mask);
		idxB = vandq_u8(vshlq_u8(vld1q_u8(repeatedB), shifts), mask);
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc1 = TQ_GRAPH_ARM_DOT(acc1, ca, cb);
		*sampleDims += tailDims;
	}

	return (int64) vaddvq_s32(acc0) + (int64) vaddvq_s32(acc1);
}

/*
 * NEON weighted symmetric (code-code) kernels for TQ+.
 *
 * Mirrors TqGraphCodeCodeWeightedRawAvx2 in math: per chunk of 16
 * coords (4-bit unpack via vqtbl1q_s8 codebook lookup), widen to
 * i16x16, multiply pairwise, multiply by per-coord i16 weight via
 * vmlal_s16 (i16×i16 → i32 multiply-accumulate), widen i32 → i64
 * via vpadalq_s32, accumulate.
 *
 * Returns Σ c_a · c_b · D'²_i16 as i64.  Caller divides by
 * `weight_scale · CODEBOOK_SCALE²` to recover the f32 weighted dot.
 *
 * Uses base NEON only (no SDOT or I8MM); compiles wherever the file
 * already targets aarch64.
 */
static inline int64x2_t TQ_GRAPH_ARM_DOT_TARGET
TqGraphWeightedDotI8x16NeonSdot(int8x16_t ca, int8x16_t cb,
								const int16 *weightsAt)
{
	int16x8_t	caLo = vmovl_s8(vget_low_s8(ca));
	int16x8_t	caHi = vmovl_s8(vget_high_s8(ca));
	int16x8_t	cbLo = vmovl_s8(vget_low_s8(cb));
	int16x8_t	cbHi = vmovl_s8(vget_high_s8(cb));
	int16x8_t	prodLo = vmulq_s16(caLo, cbLo);
	int16x8_t	prodHi = vmulq_s16(caHi, cbHi);
	int16x8_t	wLo = vld1q_s16(weightsAt);
	int16x8_t	wHi = vld1q_s16(weightsAt + 8);
	int32x4_t	chunkAcc = vdupq_n_s32(0);

	chunkAcc = vmlal_s16(chunkAcc, vget_low_s16(prodLo), vget_low_s16(wLo));
	chunkAcc = vmlal_s16(chunkAcc, vget_high_s16(prodLo), vget_high_s16(wLo));
	chunkAcc = vmlal_s16(chunkAcc, vget_low_s16(prodHi), vget_low_s16(wHi));
	chunkAcc = vmlal_s16(chunkAcc, vget_high_s16(prodHi), vget_high_s16(wHi));

	return vpadalq_s32(vdupq_n_s64(0), chunkAcc);
}

static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphCodeCodeWeightedRawNeonSdot(const uint8 *a, const uint8 *b,
								   const int16 *weights, int dim)
{
	int64x2_t	acc = vdupq_n_s64(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) TqGraphCodebookI8);
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		uint8x8_t	pa = vld1_u8(a + chunk * 8);
		uint8x8_t	pb = vld1_u8(b + chunk * 8);
		uint8x8_t	loA = vand_u8(pa, vdup_n_u8(0x0f));
		uint8x8_t	hiA = vshr_n_u8(pa, 4);
		uint8x8_t	loB = vand_u8(pb, vdup_n_u8(0x0f));
		uint8x8_t	hiB = vshr_n_u8(pb, 4);
		uint8x16_t	idxA = vcombine_u8(vzip1_u8(loA, hiA),
										vzip2_u8(loA, hiA));
		uint8x16_t	idxB = vcombine_u8(vzip1_u8(loB, hiB),
										vzip2_u8(loB, hiB));
		int8x16_t	ca = vqtbl1q_s8(codebook, idxA);
		int8x16_t	cb = vqtbl1q_s8(codebook, idxB);

		acc = vaddq_s64(acc,
						 TqGraphWeightedDotI8x16NeonSdot(ca, cb, weights + chunk * 16));
	}

	if (tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int16		scratchW[16] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		uint8x8_t	pa;
		uint8x8_t	pb;
		uint8x8_t	loA;
		uint8x8_t	hiA;
		uint8x8_t	loB;
		uint8x8_t	hiB;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		memcpy(scratchW, weights + chunks * 16, sizeof(int16) * tailDims);
		pa = vld1_u8(scratchA);
		pb = vld1_u8(scratchB);
		loA = vand_u8(pa, vdup_n_u8(0x0f));
		hiA = vshr_n_u8(pa, 4);
		loB = vand_u8(pb, vdup_n_u8(0x0f));
		hiB = vshr_n_u8(pb, 4);
		idxA = vcombine_u8(vzip1_u8(loA, hiA), vzip2_u8(loA, hiA));
		idxB = vcombine_u8(vzip1_u8(loB, hiB), vzip2_u8(loB, hiB));
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc = vaddq_s64(acc, TqGraphWeightedDotI8x16NeonSdot(ca, cb, scratchW));
	}

	return vaddvq_s64(acc);
}

static int64 TQ_GRAPH_ARM_DOT_TARGET
TqGraphCodeCode2WeightedRawNeonSdot(const uint8 *a, const uint8 *b,
									const int16 *weights, int dim)
{
	int64x2_t	acc = vdupq_n_s64(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) TqGraphCodebook2I8);
	uint8x16_t	mask = vdupq_n_u8(0x03);
	int8x16_t	shifts = {
		0, -2, -4, -6, 0, -2, -4, -6,
		0, -2, -4, -6, 0, -2, -4, -6
	};
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		uint8		repeatedA[16];
		uint8		repeatedB[16];
		const uint8 *bytesA = a + chunk * 4;
		const uint8 *bytesB = b + chunk * 4;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		for (int i = 0; i < 16; i++)
		{
			repeatedA[i] = bytesA[i / 4];
			repeatedB[i] = bytesB[i / 4];
		}

		idxA = vandq_u8(vshlq_u8(vld1q_u8(repeatedA), shifts), mask);
		idxB = vandq_u8(vshlq_u8(vld1q_u8(repeatedB), shifts), mask);
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);

		acc = vaddq_s64(acc,
						 TqGraphWeightedDotI8x16NeonSdot(ca, cb, weights + chunk * 16));
	}

	if (tailDims != 0)
	{
		uint8		repeatedA[16] = {0};
		uint8		repeatedB[16] = {0};
		int16		scratchW[16] = {0};
		const uint8 *bytesA = a + chunks * 4;
		const uint8 *bytesB = b + chunks * 4;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		for (int i = 0; i < tailDims; i++)
		{
			repeatedA[i] = bytesA[i / 4];
			repeatedB[i] = bytesB[i / 4];
		}
		memcpy(scratchW, weights + chunks * 16, sizeof(int16) * tailDims);

		idxA = vandq_u8(vshlq_u8(vld1q_u8(repeatedA), shifts), mask);
		idxB = vandq_u8(vshlq_u8(vld1q_u8(repeatedB), shifts), mask);
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc = vaddq_s64(acc, TqGraphWeightedDotI8x16NeonSdot(ca, cb, scratchW));
	}

	return vaddvq_s64(acc);
}

#endif

#if TQ_GRAPH_COMPILE_QUERY_SPLIT
static bool
TqGraphScoreNodeBatchQuerySplit4(HnswScanOpaque so,
								 TqGraphScanStorage *storage,
								 uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	double		dimSqrt;
	double		queryNormSqrt = 0;

	if (!so->tq.querySplitEnabled || so->tq.dimensions < 1024 ||
		so->tq.bits != TQ_DEFAULT_BITS || mode == TQ_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) so->tq.dimensions);
	if (mode == TQ_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeIds[j]];
		double		dot;
		int64		rawDot;

		if (node->code == NULL)
			return false;

#if TQ_GRAPH_COMPILE_ARM_DOT
		if (TqGraphArmDotprodAvailable())
			rawDot = TqGraphQuerySplitRawNeonSdot(&so->tq, node->code);
		else
#endif
#if TQ_GRAPH_COMPILE_AVX512VNNI
		if (TqGraphAvx512VnniAvailable())
			rawDot = TqGraphQuerySplitRawAvx512Vnni(&so->tq, node->code);
		else
#endif
#if TQ_GRAPH_COMPILE_AVXVNNI
		if (TqGraphAvxVnniAvailable())
			rawDot = TqGraphQuerySplitRawAvxVnni(&so->tq, node->code);
		else
#endif
#if TQ_GRAPH_COMPILE_AVX2
		if (TqGraphAvx2Available())
			rawDot = TqGraphQuerySplitRawAvx2(&so->tq, node->code);
		else
#endif
			return false;

		dot = (double) so->tq.querySplitPostprocessScale *
			(double) rawDot;
		dot += so->tq.ecCorrection;

		if (mode == TQ_SCORE_IP)
			distances[j] = -(node->scale * dot / dimSqrt);
		else if (mode == TQ_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dot / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = so->tq.scoringKernel;
	return true;
}

static bool
TqGraphScoreNodeBatchQuerySplit2(HnswScanOpaque so,
								 TqGraphScanStorage *storage,
								 uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	double		dimSqrt;
	double		queryNormSqrt = 0;

	if (!so->tq.querySplitEnabled || so->tq.dimensions < 1024 ||
		so->tq.bits != 2 || mode == TQ_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) so->tq.dimensions);
	if (mode == TQ_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeIds[j]];
		double		dot;
		int64		rawDot;

		if (node->code == NULL)
			return false;

#if TQ_GRAPH_COMPILE_ARM_DOT
		if (TqGraphArmDotprodAvailable())
			rawDot = TqGraphQuerySplit2RawNeonSdot(&so->tq, node->code);
		else
#endif
#if TQ_GRAPH_COMPILE_AVX512VNNI
		if (TqGraphAvx512VnniAvailable())
			rawDot = TqGraphQuerySplit2RawAvx512Vnni(&so->tq, node->code);
		else
#endif
#if TQ_GRAPH_COMPILE_AVXVNNI
		if (TqGraphAvxVnniAvailable())
			rawDot = TqGraphQuerySplit2RawAvxVnni(&so->tq, node->code);
		else
#endif
#if TQ_GRAPH_COMPILE_AVX2
		if (TqGraphAvx2Available())
			rawDot = TqGraphQuerySplit2RawAvx2(&so->tq, node->code);
		else
#endif
			return false;

		dot = (double) so->tq.querySplitPostprocessScale *
			(double) rawDot;
		dot += so->tq.ecCorrection;

		if (mode == TQ_SCORE_IP)
			distances[j] = -(node->scale * dot / dimSqrt);
		else if (mode == TQ_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dot / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = so->tq.scoringKernel;
	return true;
}

#endif

static bool
TqGraphScoreNodeBatchPacked4(HnswScanOpaque so, TqGraphScanStorage *storage,
							 uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	const uint8 *codes[4];
	float		scales[4];
	double		dots[4];
	int			dim = so->tq.dimensions;
	double		dimSqrt = sqrt((double) dim);

	if (!so->tq.enabled || dim <= 0 || mode == TQ_SCORE_L1 ||
		so->tq.bits != TQ_DEFAULT_BITS)
		return false;

	for (int j = 0; j < 4; j++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeIds[j]];

		if (node->code == NULL)
			return false;

		codes[j] = node->code;
		scales[j] = node->scale;
		dots[j] = so->tq.ecCorrection;
	}

	for (int i = 0; i + 1 < dim; i += 2)
	{
		float	   *loRow = so->tq.lut + (i * TQ_LUT_WIDTH);
		float	   *hiRow = loRow + TQ_LUT_WIDTH;
		int			byteIndex = i / 2;

		for (int j = 0; j < 4; j++)
		{
			uint8		packed = codes[j][byteIndex];

			dots[j] += loRow[packed & 0x0f] + hiRow[packed >> TQ_BITS];
		}
	}

	if ((dim & 1) != 0)
	{
		float	   *row = so->tq.lut + ((dim - 1) * TQ_LUT_WIDTH);
		int			byteIndex = dim / 2;

		for (int j = 0; j < 4; j++)
			dots[j] += row[codes[j][byteIndex] & 0x0f];
	}

	for (int j = 0; j < 4; j++)
	{
		if (mode == TQ_SCORE_IP)
			distances[j] = -(scales[j] * dots[j] / dimSqrt);
		else if (mode == TQ_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || scales[j] == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dots[j] / (sqrt(so->tq.queryNorm) * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) scales[j] * scales[j]) -
				(2 * scales[j] * dots[j] / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = TQ_SCORING_SCALAR;
	return true;
}

static bool
TqGraphScoreNodeBatchPackedLowBits(HnswScanOpaque so,
								   TqGraphScanStorage *storage,
								   uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	const uint8 *codes[4];
	float		scales[4];
	double		dots[4];
	int			dim = so->tq.dimensions;
	int			bits = so->tq.bits;
	double		dimSqrt = sqrt((double) dim);
	double		queryNormSqrt = 0;

	if (!so->tq.enabled || dim <= 0 || mode == TQ_SCORE_L1 ||
		(bits != 1 && bits != 2))
		return false;

	if (mode == TQ_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeIds[j]];

		if (node->code == NULL)
			return false;

		codes[j] = node->code;
		scales[j] = node->scale;
		dots[j] = so->tq.ecCorrection;
	}

	if (bits == 2)
	{
		int			i = 0;

		for (; i + 4 <= dim; i += 4)
		{
			float	   *row0 = so->tq.lut + (i * TQ_LUT_WIDTH);
			float	   *row1 = row0 + TQ_LUT_WIDTH;
			float	   *row2 = row1 + TQ_LUT_WIDTH;
			float	   *row3 = row2 + TQ_LUT_WIDTH;
			int			byteIndex = i / 4;

			for (int j = 0; j < 4; j++)
			{
				uint8		packed = codes[j][byteIndex];

				dots[j] += row0[packed & 0x03] +
					row1[(packed >> 2) & 0x03] +
					row2[(packed >> 4) & 0x03] +
					row3[(packed >> 6) & 0x03];
			}
		}

		for (; i < dim; i++)
		{
			float	   *row = so->tq.lut + (i * TQ_LUT_WIDTH);
			int			byteIndex = i / 4;
			int			shift = (i & 3) * 2;

			for (int j = 0; j < 4; j++)
				dots[j] += row[(codes[j][byteIndex] >> shift) & 0x03];
		}
	}
	else
	{
		int			i = 0;

		for (; i + 8 <= dim; i += 8)
		{
			float	   *row0 = so->tq.lut + (i * TQ_LUT_WIDTH);
			float	   *row1 = row0 + TQ_LUT_WIDTH;
			float	   *row2 = row1 + TQ_LUT_WIDTH;
			float	   *row3 = row2 + TQ_LUT_WIDTH;
			float	   *row4 = row3 + TQ_LUT_WIDTH;
			float	   *row5 = row4 + TQ_LUT_WIDTH;
			float	   *row6 = row5 + TQ_LUT_WIDTH;
			float	   *row7 = row6 + TQ_LUT_WIDTH;
			int			byteIndex = i / 8;

			for (int j = 0; j < 4; j++)
			{
				uint8		packed = codes[j][byteIndex];

				dots[j] += row0[packed & 0x01] +
					row1[(packed >> 1) & 0x01] +
					row2[(packed >> 2) & 0x01] +
					row3[(packed >> 3) & 0x01] +
					row4[(packed >> 4) & 0x01] +
					row5[(packed >> 5) & 0x01] +
					row6[(packed >> 6) & 0x01] +
					row7[(packed >> 7) & 0x01];
			}
		}

		for (; i < dim; i++)
		{
			float	   *row = so->tq.lut + (i * TQ_LUT_WIDTH);
			int			byteIndex = i / 8;
			int			shift = i & 7;

			for (int j = 0; j < 4; j++)
				dots[j] += row[(codes[j][byteIndex] >> shift) & 0x01];
		}
	}

	for (int j = 0; j < 4; j++)
	{
		if (mode == TQ_SCORE_IP)
			distances[j] = -(scales[j] * dots[j] / dimSqrt);
		else if (mode == TQ_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || scales[j] == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dots[j] / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) scales[j] * scales[j]) -
				(2 * scales[j] * dots[j] / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = TQ_SCORING_SCALAR;
	return true;
}

static int
TqGraphBit1PopcntRaw(HnswScanOpaque so, const uint8 *code)
{
	return TqGraphBit1PopcntRawCodes(so->tq.querySignBits, code,
									 so->tq.dimensions);
}

/*
 * Batch dispatch for the asymmetric 1-bit scoring path.
 *
 * Active when hnsw.tq_query_1bit_asymmetric is on AND the query
 * precompute populated tq->queryPlanes (1-bit indexes only).  When
 * either condition is missing, returns false so the existing 1-bit
 * popcount or LUT path takes over — preserving baseline behaviour
 * for users who don't opt in.
 *
 * Per-node scoring uses TqGraphAsymBit1Score (AVX2 / NEON / scalar
 * dispatched).  The IP / Cosine / L2 postprocess mirrors the
 * symmetric popcount path exactly so callers see the same shape of
 * output — only the dot value carries more magnitude information.
 */
static bool
TqGraphScoreNodeBatchAsymBit1(HnswScanOpaque so,
							  TqGraphScanStorage *storage,
							  uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	double		dimSqrt;
	double		queryNormSqrt = 0;

	if (!hnsw_tq_query_1bit_asymmetric || !so->tq.enabled ||
		so->tq.bits != 1 || so->tq.queryPlanes == NULL ||
		so->tq.dimensions <= 0 || mode == TQ_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) so->tq.dimensions);
	if (mode == TQ_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeIds[j]];
		double		dot;

		if (node->code == NULL)
			return false;

		dot = so->tq.ecCorrection +
			(double) TqGraphAsymBit1Score(&so->tq, node->code);

		if (mode == TQ_SCORE_IP)
			distances[j] = -(node->scale * dot / dimSqrt);
		else if (mode == TQ_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dot / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = so->tq.scoringKernel;
	return true;
}

static bool
TqGraphScoreNodeBatchPopcntLowBits(HnswScanOpaque so,
								   TqGraphScanStorage *storage,
								   uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	double		dimSqrt;
	double		queryNormSqrt = 0;
	double		center;

	if (!hnsw_tq_graph_lowbit_popcnt || !so->tq.enabled ||
		so->tq.bits != 1 || so->tq.querySignBits == NULL ||
		so->tq.dimensions <= 0 || mode == TQ_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) so->tq.dimensions);
	center = TqGraphCodeCenter(1, 1);
	if (mode == TQ_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		TqGraphScanNode *node = &storage->nodes[nodeIds[j]];
		double		dot;

		if (node->code == NULL)
			return false;

		dot = so->tq.ecCorrection +
			(center * (double) TqGraphBit1PopcntRaw(so, node->code));

		if (mode == TQ_SCORE_IP)
			distances[j] = -(node->scale * dot / dimSqrt);
		else if (mode == TQ_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dot / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = TQ_SCORING_SCALAR;
	return true;
}

void
TqGraphScoreNodeBatch(HnswScanOpaque so, TqGraphScanStorage *storage,
					  uint32 *nodeIds, int nodeCount, double *distances,
					  Datum query)
{
	if (hnsw_tq_graph_batch_scoring == TQ_GRAPH_BATCH_OFF ||
		hnsw_tq_graph_batch_size < 4)
	{
		for (int i = 0; i < nodeCount; i++)
			distances[i] = TqGraphScoreNode(so, &storage->nodes[nodeIds[i]]);
		return;
	}

	if (TqGraphUseExactLowBitRouting(so, query))
	{
		bool		exactBatch = true;

		for (int j = 0; j < nodeCount; j++)
		{
			TqGraphScanNode *node = &storage->nodes[nodeIds[j]];

			if (node->exactVector == NULL)
			{
				exactBatch = false;
				break;
			}
		}

		if (exactBatch)
		{
			for (int j = 0; j < nodeCount; j++)
				distances[j] = TqGraphExactVectorDistance(so, query,
														  storage->nodes[nodeIds[j]].exactVector);
			return;
		}
	}

	for (int i = 0; i < nodeCount;)
	{
		for (int j = i; hnsw_tq_graph_prefetch && j < Min(i + 8, nodeCount); j++)
		{
			TqGraphScanNode *node = &storage->nodes[nodeIds[j]];

			if (node->code != NULL)
				TQ_GRAPH_PREFETCH_READ(node->code);
		}

		if (i + 4 <= nodeCount)
		{
			bool		batchScored =
#if TQ_GRAPH_COMPILE_QUERY_SPLIT
				TqGraphScoreNodeBatchQuerySplit4(so, storage, nodeIds + i,
												 distances + i) ||
				TqGraphScoreNodeBatchQuerySplit2(so, storage, nodeIds + i,
												 distances + i) ||
#endif
				TqGraphScoreNodeBatchPacked4(so, storage, nodeIds + i,
											 distances + i) ||
				TqGraphScoreNodeBatchAsymBit1(so, storage, nodeIds + i,
											  distances + i) ||
				TqGraphScoreNodeBatchPopcntLowBits(so, storage, nodeIds + i,
												   distances + i) ||
				TqGraphScoreNodeBatchPackedLowBits(so, storage, nodeIds + i,
												   distances + i);

			if (batchScored)
			{
				i += 4;
				continue;
			}
		}

		distances[i] = TqGraphScoreNode(so, &storage->nodes[nodeIds[i]]);
		i++;
	}
}

bool
TqGraphCodeCodeWeightedRawSimdSelf(const uint8 *code, int dimensions, int bits,
								   const float *ecScale, double *raw)
{
	double		dPrimeSqMax = 0.0;
	double		weightScale;
	double		codebookScaleSq;
	int16	   *dPrimeSqI16;
	int64		rawI64 = 0;
	bool		simdRan = false;

	if (raw == NULL)
		return false;
	*raw = 0.0;

	if (code == NULL || ecScale == NULL)
		return false;

	for (int d = 0; d < dimensions; d++)
	{
		double		s = (double) ecScale[d];

		if (fabs(s) > FLT_EPSILON)
		{
			double		w = 1.0 / (s * s);

			if (w > dPrimeSqMax)
				dPrimeSqMax = w;
		}
	}

	if (dPrimeSqMax <= FLT_EPSILON)
		return false;

	weightScale = ((double) INT16_MAX - 1.0) / dPrimeSqMax;
	dPrimeSqI16 = palloc(sizeof(int16) * dimensions);

	for (int d = 0; d < dimensions; d++)
	{
		double		s = (double) ecScale[d];
		double		w = (fabs(s) > FLT_EPSILON) ? 1.0 / (s * s) : 0.0;
		double		q = round(w * weightScale);

		if (q < 0.0)
			q = 0.0;
		if (q > (double) (INT16_MAX - 1))
			q = (double) (INT16_MAX - 1);
		dPrimeSqI16[d] = (int16) q;
	}

	if (bits == TQ_DEFAULT_BITS)
		codebookScaleSq = TQ_GRAPH_CODEBOOK_SCALE * TQ_GRAPH_CODEBOOK_SCALE;
	else
		codebookScaleSq = TQ_GRAPH_CODEBOOK2_SCALE * TQ_GRAPH_CODEBOOK2_SCALE;

#if TQ_GRAPH_COMPILE_ARM_DOT
	if (bits == TQ_DEFAULT_BITS)
		rawI64 = TqGraphCodeCodeWeightedRawNeonSdot(code, code,
													dPrimeSqI16,
													dimensions);
	else
		rawI64 = TqGraphCodeCode2WeightedRawNeonSdot(code, code,
													 dPrimeSqI16,
													 dimensions);
	simdRan = true;
#endif
#if TQ_GRAPH_COMPILE_AVX2
	if (!simdRan && TqGraphAvx2Available())
	{
		if (bits == TQ_DEFAULT_BITS)
			rawI64 = TqGraphCodeCodeWeightedRawAvx2(code, code,
													dPrimeSqI16,
													dimensions);
		else
			rawI64 = TqGraphCodeCode2WeightedRawAvx2(code, code,
													 dPrimeSqI16,
													 dimensions);
		simdRan = true;
	}
#endif

	if (simdRan)
		*raw = (double) rawI64 / (weightScale * codebookScaleSq);

	pfree(dPrimeSqI16);
	return simdRan;
}
