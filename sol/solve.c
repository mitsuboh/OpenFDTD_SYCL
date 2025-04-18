/*
solve.c
*/

#include "ofd.h"
#include "ofd_prototype.h"

void solve(int ilog, double *tdft, FILE *fp)
{
	double fmax[] = {0, 0};
	char   str[BUFSIZ];
	int    converged = 0;

	// initial field
	initfield();

	// time step iteration
	int itime;
	double t = 0;
	for (itime = 0; itime <= Solver.maxiter; itime++) {

		// update H
		t += 0.5 * Dt;
		updateHx(t);
		updateHy(t);
		updateHz(t);

		// ABC H
		if      (iABC == 0) {
			murH(numMurHx, fMurHx, Hx);
			murH(numMurHy, fMurHy, Hy);
			murH(numMurHz, fMurHz, Hz);
		}
		else if (iABC == 1) {
			pmlHx();
			pmlHy();
			pmlHz();
		}

		// PBC H
		if (PBCx) {
			pbcx();
		}
		if (PBCy) {
			pbcy();
		}
		if (PBCz) {
			pbcz();
		}

		// update E
		t += 0.5 * Dt;
		updateEx(t);
		updateEy(t);
		updateEz(t);

		// dispersion E
		if (numDispersionEx) {
			dispersionEx(t);
		}
		if (numDispersionEy) {
			dispersionEy(t);
		}
		if (numDispersionEz) {
			dispersionEz(t);
		}

		// ABC E
		if      (iABC == 1) {
			pmlEx();
			pmlEy();
			pmlEz();
		}

		// feed
		if (NFeed) {
			efeed(itime);
		}

		// inductor
		if (NInductor) {
			eload();
		}

		// point
		if (NPoint) {
			vpoint(itime);
		}

		// DFT
		const double t0 = cputime();
		dftNear3d(itime);
		*tdft += cputime() - t0;

		// average and convergence
		if ((itime % Solver.nout == 0) || (itime == Solver.maxiter)) {
			// average
			double fsum[2];
			average(fsum);

			// average (post)
			Eiter[Niter] = fsum[0];
			Hiter[Niter] = fsum[1];
			Niter++;

			// monitor
			if (ilog) {
				sprintf(str, "%7d %.6f %.6f", itime, fsum[0], fsum[1]);
				fprintf(fp,     "%s\n", str);
				fprintf(stdout, "%s\n", str);
				fflush(fp);
				fflush(stdout);
			}

			// check convergence
			fmax[0] = MAX(fmax[0], fsum[0]);
			fmax[1] = MAX(fmax[1], fsum[1]);
			if ((fsum[0] < fmax[0] * Solver.converg) &&
			    (fsum[1] < fmax[1] * Solver.converg)) {
				converged = 1;
				break;
			}
		}
	}

	// result
	if (ilog) {
		sprintf(str, "    --- %s ---", (converged ? "converged" : "max steps"));
		fprintf(fp,     "%s\n", str);
		fprintf(stdout, "%s\n", str);
		fflush(fp);
		fflush(stdout);
	}

	// time steps
	Ntime = itime + converged;

	// free
	memfree2();
}
