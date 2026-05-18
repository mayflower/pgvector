#ifndef TQGRAPH_SCORE_H
#define TQGRAPH_SCORE_H

#include "tqgraph.h"

double		TqGraphBuildDistance(TqGraphBuildState *state, uint32 a, uint32 b);
bool		TqGraphCachedExactNodeDistance(HnswScanOpaque so, Datum query,
										   TqGraphScanNode *node,
										   double *distance);
bool		TqGraphCodeCodeWeightedRawSimdSelf(const uint8 *code,
											  int dimensions, int bits,
											  const float *ecScale,
											  double *raw);
bool		TqGraphCodeCodeDistance(HnswScanOpaque so, HnswMetaPageData *meta,
									TqGraphScanNode *aNode,
									TqGraphScanNode *bNode,
									double *distance);
float		TqGraphCodeNorm(const uint8 *code, int dimensions, int bits);
double		TqGraphCodeCodeWeightedRawScalar(const uint8 *a, const uint8 *b,
											  int dimensions, int bits,
											  const float *ecScale);
float		TqGraphEncodeVector(TqGraphBuildState *state, Vector *vector,
								uint8 *code);
float		TqGraphEncodeVectorWithXm(TqGraphBuildState *state, Vector *vector,
									  uint8 *code, float *xmOut);
float		TqGraphEncodeVectorWithXmRenorm(TqGraphBuildState *state,
											Vector *vector, uint8 *code,
											float *xmOut);
bool		TqGraphExactHighdimEntryDistance(HnswScanOpaque so, Datum query,
											 TqGraphScanNode *node,
											 double *distance);
double		TqGraphExactDistance(HnswSupport *support, Datum a, Datum b);
double		TqGraphExactVectorDistance(HnswScanOpaque so, Datum query,
										char *valuePtr);
TqScoreMode TqGraphGetScoreMode(HnswSupport *support);
double		TqGraphMmConstScalar(const float *ecShift, int dimensions);
void		TqGraphPrepareBuildQuery(TqGraphBuildState *state, uint32 nodeId);
double		TqGraphScoreNode(HnswScanOpaque so, TqGraphScanNode *node);
void		TqGraphScoreNodeBatch(HnswScanOpaque so, TqGraphScanStorage *storage,
								  uint32 *nodeIds, int nodeCount,
								  double *distances, Datum query);
float		TqGraphVectorNorm(Vector *vector);

#endif
