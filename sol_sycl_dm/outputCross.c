/*
outputCross.c

cross section
*/

#include "ofd.h"
#include "complex.h"
#include "ofd_prototype.h"


static void _outputCross(FILE *fp, const double bcs[], const double fcs[])
{
	fprintf(fp, "=== cross section ===\n");
	fprintf(fp, "  %s\n", "frequency[Hz] backward[m*m]  forward[m*m]");
	for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
		fprintf(fp, "  %13.5e  %12.4e  %12.4e\n", Freq2[ifreq], bcs[ifreq], fcs[ifreq]);
	}
}


void outputCross(FILE *fp)
{
	// alloc
	double *bcs = (double *)malloc(NFreq2 * sizeof(double));
	double *fcs = (double *)malloc(NFreq2 * sizeof(double));

	// calculation
	for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
		const double ffctr = farfactor(ifreq);
		d_complex_t etheta, ephi;
		// BCS
		farfield(ifreq,       Planewave.theta,       Planewave.phi, ffctr, &etheta, &ephi);
		bcs[ifreq] = d_norm(etheta) + d_norm(ephi);
		// FCS
		farfield(ifreq, 180 - Planewave.theta, 180 + Planewave.phi, ffctr, &etheta, &ephi);
		fcs[ifreq] = d_norm(etheta) + d_norm(ephi);
	}

	// output
	_outputCross(stdout, bcs, fcs);
	_outputCross(fp,     bcs, fcs);

	// free
	free(bcs);
	free(fcs);
}
