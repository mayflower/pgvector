#include "postgres.h"

#include <math.h>
#include <string.h>

#include "tqgraph_psquare.h"

void
TqPSquareInit(TqPSquareState *p, double quantile)
{
	p->count = 0;
	p->target_q = quantile;
	memset(p->heights, 0, sizeof(p->heights));
	memset(p->n_positions, 0, sizeof(p->n_positions));
}

/*
 * Adjust one interior marker (i ∈ {1, 2, 3} for N=5) using parabolic
 * prediction; falls back to linear interpolation when parabolic
 * lies outside the [prev, next] bracket.  The dsign argument is +1
 * or -1 indicating which neighbour to step toward.
 */
static void
TqPSquareAdjustStep(int i, double *heights, double *n_positions, double dsign)
{
	double		prev_h = heights[i - 1];
	double		next_h = heights[i + 1];
	double		prev_n = n_positions[i - 1];
	double		next_n = n_positions[i + 1];
	double		cur_h = heights[i];
	double		cur_n = n_positions[i];
	double		denom = next_n - prev_n;
	double		h_par = cur_h;

	if (denom != 0.0)
	{
		double		a = (cur_n - prev_n + dsign) / (next_n - cur_n) * (next_h - cur_h);
		double		b = (next_n - cur_n - dsign) / (cur_n - prev_n) * (cur_h - prev_h);

		h_par = cur_h + (a + b) * dsign / denom;
	}

	if (h_par > prev_h && h_par < next_h && isfinite(h_par))
		heights[i] = h_par;
	else if (dsign > 0.0)
		heights[i] = cur_h + (next_h - cur_h) / (next_n - cur_n);
	else
		heights[i] = cur_h + (prev_h - cur_h) / (prev_n - cur_n);

	n_positions[i] += dsign;
}

static void
TqPSquareAdjustMarker(int i, double *heights, double *n_positions,
					  const double *n_desired)
{
	while (true)
	{
		double		di = n_desired[i] - n_positions[i];

		if (di >= 1.0 && (n_positions[i + 1] - n_positions[i]) > 1.0)
			TqPSquareAdjustStep(i, heights, n_positions, 1.0);
		else if (di <= -1.0 && (n_positions[i - 1] - n_positions[i]) < -1.0)
			TqPSquareAdjustStep(i, heights, n_positions, -1.0);
		else
			break;
	}
}

void
TqPSquarePush(TqPSquareState *p, double x)
{
	double		target_probabilities[TQ_PSQUARE_N];
	double		n_desired[TQ_PSQUARE_N];
	double		count_minus_one;
	int			k;

	if (!isfinite(x))
		return;

	/* Bootstrap: collect first N observations into heights[]. */
	if (p->count < TQ_PSQUARE_N)
	{
		p->heights[p->count] = x;
		p->count++;
		if (p->count == TQ_PSQUARE_N)
		{
			/* Sort the initial heights and set marker positions. */
			for (int i = 0; i < TQ_PSQUARE_N - 1; i++)
			{
				for (int j = i + 1; j < TQ_PSQUARE_N; j++)
				{
					if (p->heights[j] < p->heights[i])
					{
						double		tmp = p->heights[i];

						p->heights[i] = p->heights[j];
						p->heights[j] = tmp;
					}
				}
				p->n_positions[i] = (double) (i + 1);
			}
			p->n_positions[TQ_PSQUARE_N - 1] = (double) TQ_PSQUARE_N;
		}
		return;
	}

	p->count++;

	/* Step 1: identify cell k and update extreme markers. */
	if (x < p->heights[0])
	{
		p->heights[0] = x;
		k = 0;
	}
	else if (x > p->heights[TQ_PSQUARE_N - 1])
	{
		p->heights[TQ_PSQUARE_N - 1] = x;
		k = TQ_PSQUARE_N - 1;
	}
	else
	{
		k = 0;
		for (int i = 1; i < TQ_PSQUARE_N; i++)
		{
			if (x < p->heights[i])
			{
				k = i - 1;
				break;
			}
			k = i;
		}
		if (k >= TQ_PSQUARE_N - 1)
			k = TQ_PSQUARE_N - 2;
	}

	/* Step 2: increment n_positions for markers above k. */
	for (int i = k + 1; i < TQ_PSQUARE_N; i++)
		p->n_positions[i] += 1.0;

	/* Step 3: update desired positions.  Target probabilities for N=5
	 * are [0, q/2, q, (1+q)/2, 1] per Jain & Chlamtac. */
	target_probabilities[0] = 0.0;
	target_probabilities[1] = p->target_q * 0.5;
	target_probabilities[2] = p->target_q;
	target_probabilities[3] = 0.5 * (1.0 + p->target_q);
	target_probabilities[4] = 1.0;

	count_minus_one = (double) (p->count - 1);
	for (int i = 0; i < TQ_PSQUARE_N; i++)
		n_desired[i] = 1.0 + target_probabilities[i] * count_minus_one;

	/* Step 4: parabolic adjust interior markers. */
	for (int i = 1; i < TQ_PSQUARE_N - 1; i++)
		TqPSquareAdjustMarker(i, p->heights, p->n_positions, n_desired);
}

/*
 * Return the current estimate of the target quantile.  Before N
 * observations have been collected, returns the median of available
 * observations (guarantees a finite value even on tiny inputs).
 */
double
TqPSquareEstimate(const TqPSquareState *p)
{
	if (p->count == 0)
		return 0.0;
	if (p->count < TQ_PSQUARE_N)
	{
		double		buf[TQ_PSQUARE_N];

		memcpy(buf, p->heights, sizeof(double) * (size_t) p->count);
		for (int i = 0; i < (int) p->count - 1; i++)
		{
			for (int j = i + 1; j < (int) p->count; j++)
			{
				if (buf[j] < buf[i])
				{
					double		tmp = buf[i];

					buf[i] = buf[j];
					buf[j] = tmp;
				}
			}
		}
		return buf[p->count / 2];
	}
	return p->heights[TQ_PSQUARE_N / 2];
}
