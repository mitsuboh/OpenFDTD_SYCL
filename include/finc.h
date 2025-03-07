/*
finc.h

incidence function (gauss derivative)
*/

#include <math.h>

static inline void finc(
	double x, double y, double z, double t,
	const double r0[], const double ri[], double fc, double ai, double dt,
	real_t *fi, real_t *dfi)
{
	const double c = 2.99792458e8;

	t -= ((x - r0[0]) * ri[0]
	    + (y - r0[1]) * ri[1]
	    + (z - r0[2]) * ri[2]) / c;

	const double at = ai * t;
	const double ex = (at * at < 16) ? exp(-at * at) : 0;
	//const double ex = exp(-at * at);
	*fi = (real_t)(at * ex * fc);
	*dfi = (real_t)(dt * ai * (1 - 2 * at * at) * ex * fc);
}

#ifdef _ONEAPI

static inline void finc_s(real_t x, real_t y, real_t z, real_t t,
	const real_t r0[], const real_t ri[],
	real_t fc, real_t ai, real_t dt,
	real_t* fi, real_t* dfi)
{
	const real_t c = 2.99792458e8f;

	t -= ((x - r0[0]) * ri[0]
		+ (y - r0[1]) * ri[1]
		+ (z - r0[2]) * ri[2]) / c;

	const real_t at = ai * t;
	const real_t ex = (at * at < 16) ? expf(-at * at) : 0;
	//const real_t ex = (real_t)exp(-at * at);
	*fi = at * ex * fc;
	*dfi = dt * ai * (1 - 2 * at * at) * ex * fc;
}

#endif
