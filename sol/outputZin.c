/*
outputZin.c

input impedance
*/

#include "ofd.h"
#include "complex.h"
#include "ofd_prototype.h"


// calculate input impedance and reflection
void calcZin(void)
{
	if ((NFeed <= 0) || (NFreq1 <= 0)) return;

	Zin = (d_complex_t *)malloc(NFeed * NFreq1 * sizeof(d_complex_t));
	Ref =      (double *)malloc(NFeed * NFreq1 * sizeof(double));

	for (int ifeed = 0; ifeed < NFeed; ifeed++) {
		double *fv = &VFeed[ifeed * (Solver.maxiter + 1)];
		double *fi = &IFeed[ifeed * (Solver.maxiter + 1)];
		for (int ifreq = 0; ifreq < NFreq1; ifreq++) {
			const int id = (ifeed * NFreq1) + ifreq;

			// Zin
			const d_complex_t vin = calcdft(Ntime, fv, Freq1[ifreq], Dt, 0);
			const d_complex_t iin = calcdft(Ntime, fi, Freq1[ifreq], Dt, -0.5);
			Zin[id] = d_div(vin, iin);

			// Reflection = (Zin - Z0) / (Zin + Z0)
			const d_complex_t z0 = d_complex(Feed[ifeed].z0, 0);
			const d_complex_t ref = d_div(d_sub(Zin[id], z0), d_add(Zin[id], z0));
			Ref[id] = 10 * log10(d_norm(ref));
		}
	}
}


// calculate input Power (for far field gain : for post)
void calcPin(void)
{
	if ((NFeed <= 0) || (NFreq2 <= 0)) return;

	for (int i = 0; i < 2; i++) {
		Pin[i] = (double *)malloc(NFeed * NFreq2 * sizeof(double));
	}

	for (int ifeed = 0; ifeed < NFeed; ifeed++) {
		double *fv = &VFeed[ifeed * (Solver.maxiter + 1)];
		double *fi = &IFeed[ifeed * (Solver.maxiter + 1)];
		for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
			const d_complex_t vin = d_div(calcdft(Ntime, fv, Freq2[ifreq], Dt, 0),    cFdft[ifreq]);
			const d_complex_t iin = d_div(calcdft(Ntime, fi, Freq2[ifreq], Dt, -0.5), cFdft[ifreq]);
			const d_complex_t zin = d_div(vin, iin);
			const double rin = zin.r;
			const double xin = zin.i;
			const double z0 = Feed[ifeed].z0;
			const double denom = 1
				 - ((rin - z0) * (rin - z0) + (xin * xin))
				 / ((rin + z0) * (rin + z0) + (xin * xin));
			const int id = (ifeed * NFreq2) + ifreq;
			Pin[0][id] = (vin.r * iin.r) + (vin.i * iin.i);
			Pin[1][id] = Pin[0][id] / MAX(denom, EPS);
		}
	}
}


static void _outputZin(FILE *fp)
{
	fprintf(fp, "=== input impedance ===\n");

	for (int ifeed = 0; ifeed < NFeed; ifeed++) {
		fprintf(fp, "feed #%d (Z0[ohm] = %.2f)\n", ifeed + 1, Feed[ifeed].z0);
		fprintf(fp, "  %s\n", "frequency[Hz] Rin[ohm]   Xin[ohm]    Gin[mS]    Bin[mS]    Ref[dB]       VSWR");
		for (int ifreq = 0; ifreq < NFreq1; ifreq++) {
			const int id = (ifeed * NFreq1) + ifreq;
			const d_complex_t yin = d_inv(Zin[id]);
			const double gamma = pow(10, Ref[id] / 20);
			const double vswr = (fabs(1 - gamma) > EPS) ? (1 + gamma) / (1 - gamma) : 1000;
			fprintf(fp, "%13.5e%11.3f%11.3f%11.3f%11.3f%11.3f%11.3f\n",
				Freq1[ifreq], Zin[id].r, Zin[id].i, yin.r * 1e3, yin.i * 1e3, Ref[id], vswr);
		}
	}

	fflush(fp);
}


void outputZin(FILE *fp)
{
	_outputZin(stdout);
	_outputZin(fp);
}
