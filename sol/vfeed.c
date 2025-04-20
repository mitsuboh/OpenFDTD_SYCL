/*
vfeed.c

feed voltage : gauss derivative
*/

#include <math.h>

double vfeed(double t, double tw, double td, int wf)
{
	const double arg = (t - tw - td) / (tw / 4);

	return (wf == 0) ? sqrt(2.0) * exp(0.5) * arg * exp(-arg * arg) :
	                   exp(-arg * arg);
}
