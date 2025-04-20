/*
outputPoint.c

output point data
*/

#include "ofd.h"
#include "complex.h"
#include "ofd_prototype.h"


void calcSpara(void)
{
	if ((NPoint <= 0) || (NFreq1 <= 0)) return;

	// alloc
	d_complex_t *cv = (d_complex_t *)malloc((NPoint + 2) * NFreq1 * sizeof(d_complex_t));

	// DFT
	for (int ipoint = 0; ipoint < NPoint + 2; ipoint++) {
		for (int ifreq = 0; ifreq < NFreq1; ifreq++) {
			const int id = (ipoint * NFreq1) + ifreq;
			cv[id] = calcdft(Ntime, &VPoint[ipoint * (Solver.maxiter + 1)], Freq1[ifreq], Dt, 0);
		}
	}

	// S-parameters
	for (int ifreq = 0; ifreq < NFreq1; ifreq++) {
		d_complex_t cv0 = cv[ 0           * NFreq1 + ifreq];  // V1
		d_complex_t cvp = cv[ NPoint      * NFreq1 + ifreq];  // V1+
		d_complex_t cvm = cv[(NPoint + 1) * NFreq1 + ifreq];  // V1-
		//printf("%d %f %f %f\n", ifreq, d_abs(cv0), d_abs(cvp), d_abs(cvm));
		//printf("%d %f %f %f %f %f %f\n", ifreq, cv0.r, cv0.i, cvp.r, cvp.i, cvm.r, cvm.i);
		d_complex_t c1 = d_div(d_add(cvp, cvm), cv0);
		d_complex_t c2 = d_sqrt(d_sub(d_mul(c1, c1), d_complex(4, 0)));
		d_complex_t c3 = d_add(c1, c2);
		if (c3.i < 0) c3 = d_sub(c1, c2);  // Im > 0
		c3 = d_rmul(0.5, c3);              // exp(+gd)
		d_complex_t c4 = d_div(d_complex(1, 0), c3);   // exp(-gd)
		d_complex_t c5 = d_sub(d_mul(c4, c4), d_mul(c3, c3));  // exp(-2gd) - exp(2gd)
		d_complex_t c6 = d_div(d_sub(d_mul(cvp, c4), d_mul(cvm, c3)), c5);  // V+
		d_complex_t c7 = d_div(d_sub(d_mul(cvm, c4), d_mul(cvp, c3)), c5);  // V-
		// S11 = E- / E+
		Spara[ifreq] = d_div(c7, c6);
		// Sn1 (n > 1)
		for (int ipoint = 1; ipoint < NPoint; ipoint++) {
			const int id = (ipoint * NFreq1) + ifreq;
			Spara[id] = d_div(cv[id], c6);  // Sn1 = Vn / V+
		}
	}

	// free
	free(cv);
}


void outputSpara(FILE *fp)
{
	fprintf(fp, "=== S-parameters ===\n");

	fprintf(fp, "  frequency[Hz]");
	for (int ipoint = 0; ipoint < NPoint; ipoint++) {
		fprintf(fp, "  S%d1[dB] S%d1[deg]", ipoint + 1, ipoint + 1);
	}
	fprintf(fp, "\n");

	for (int ifreq = 0; ifreq < NFreq1; ifreq++) {
		fprintf(fp, "  %13.5e", Freq1[ifreq]);
		for (int ipoint = 0; ipoint < NPoint; ipoint++) {
			const int id = (ipoint * NFreq1) + ifreq;
			fprintf(fp, "%9.3f%9.3f", 20 * log10(MAX(d_abs(Spara[id]), EPS2)), d_deg(Spara[id]));
		}
		fprintf(fp, "\n");
	}

	fflush(fp);
}


void outputPoint(const char fn_point[])
{
	FILE *fp;
	if ((fp = fopen(fn_point, "w")) == NULL) {
		fprintf(stderr, "*** %s open error.\n", fn_point);
		return;
	}

	for (int ipoint = 0; ipoint < NPoint; ipoint++) {
		fprintf(fp, "point #%d (waveform)\n", ipoint + 1);
		fprintf(fp, "%s\n", "    No.    time[sec]      V[V]");
		for (int itime = 0; itime < Ntime; itime++) {
			const int id = ipoint * (Solver.maxiter + 1) + itime;
			fprintf(fp, "%7d %13.5e %13.5e\n", itime, itime * Dt, VPoint[id]);
		}
	}

	for (int ipoint = 0; ipoint < NPoint; ipoint++) {
		fprintf(fp, "point #%d (spectrum)\n", ipoint + 1);
		fprintf(fp, "%s\n", " No. frequency[Hz]  amplitude   degree");
		for (int ifreq = 0; ifreq < NFreq1; ifreq++) {
			// DFT
			const d_complex_t vsum = calcdft(Ntime, &VPoint[ipoint * (Solver.maxiter + 1)], Freq1[ifreq], Dt, 0);
			fprintf(fp, "%4d %13.5e %9.5f %9.3f\n", ifreq, Freq1[ifreq], d_abs(vsum), d_deg(vsum));
		}
	}

	fflush(fp);
	fclose(fp);
}

/*
void outputPoint(int ilog, FILE *fp, const char fn_point[])
{
	// ofd.log and stdout
	if (ilog) {
		_outputSpara(stdout);
		_outputSpara(fp);
	}

	// point.log
	_outputPoint(fn_point);
}
*/
