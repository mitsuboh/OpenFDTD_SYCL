/*
outputCoupling.c

coupling
*/

#include "ofd.h"
#include "complex.h"
#include "ofd_prototype.h"


d_complex_t coupling(int ifeed, int ipoint, int ifreq)
{
	const d_complex_t cvf = calcdft(Ntime, &VFeed[ifeed   * (Solver.maxiter + 1)], Freq1[ifreq], Dt, 0);
	const d_complex_t cvp = calcdft(Ntime, &VPoint[ipoint * (Solver.maxiter + 1)], Freq1[ifreq], Dt, 0);

	return d_div(cvp, cvf);
}


static void _outputCoupling(FILE *fp)
{
	char str[BUFSIZ];

	fprintf(fp, "=== coupling ===\n");

	fprintf(fp, "  frequency[Hz]");
	for (int ifeed = 0; ifeed < NFeed; ifeed++) {
		for (int ipoint = 0; ipoint < NPoint; ipoint++) {
			sprintf(str, "C%d%d", ipoint + 1, ifeed + 1);
			fprintf(fp, "  %s[dB] %s[deg]", str, str);
		}
	}
	fprintf(fp, "\n");

	for (int ifreq = 0; ifreq < NFreq1; ifreq++) {
		fprintf(fp, "  %13.5e", Freq1[ifreq]);
		for (int ifeed = 0; ifeed < NFeed; ifeed++) {
			for (int ipoint = 0; ipoint < NPoint; ipoint++) {
				const d_complex_t couple = coupling(ifeed, ipoint, ifreq);
				fprintf(fp, "%9.3f", 20 * log10(MAX(d_abs(couple), EPS2)));
				fprintf(fp, "%9.3f", d_deg(couple));
			}
		}
		fprintf(fp, "\n");
	}

	fflush(fp);
}


void outputCoupling(FILE *fp)
{
	_outputCoupling(stdout);
	_outputCoupling(fp);
}
