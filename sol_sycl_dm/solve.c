/*
solve.c
*/

#include "ofd.h"
#include "ofd_prototype.h"

#ifdef _ONEAPI
#undef C        // C is used for (2.99792458e8) but <CL/sycl.
#include "ofd_dpcpp.h"
#endif
extern void		setup_xpl(void);

void solve(int io, double *tdft, FILE *fp)
{
	double fmax[] = {0, 0};
	char   str[BUFSIZ];
	int    converged = 0;

	// initial field
	initfield();

#ifdef _ONEAPI
        // setup xpu local memory
        setup_xpl();
#endif

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
#ifdef _ONEAPI
			murH(numMurHx, d_fMurHx, Hx);
			murH(numMurHy, d_fMurHy, Hy);
			murH(numMurHz, d_fMurHz, Hz);
#else
			murH(numMurHx, fMurHx, Hx);
			murH(numMurHy, fMurHy, Hy);
			murH(numMurHz, fMurHz, Hz);
#endif
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
			if (io) {
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
	if (io) {
		sprintf(str, "    --- %s ---", (converged ? "converged" : "max steps"));
		fprintf(fp,     "%s\n", str);
		fprintf(stdout, "%s\n", str);
		fflush(fp);
		fflush(stdout);
	}

	// time steps
	Ntime = itime + converged;

        // copy device to host
#ifdef _ONEAPI
        if (NFeed > 0) {
                myQ.memcpy(VFeed, d_VFeed, Feed_size).wait();
                myQ.memcpy(IFeed, d_IFeed, Feed_size).wait();
        }

	if (NN && NFreq2) {
		size_t size = NN * NFreq2 * sizeof(real_t);
		myQ.memcpy(cEx_r, d_cEx_r, size).wait();
		myQ.memcpy(cEy_r, d_cEy_r, size).wait();
		myQ.memcpy(cEz_r, d_cEz_r, size).wait();
		myQ.memcpy(cHx_r, d_cHx_r, size).wait();
		myQ.memcpy(cHy_r, d_cHy_r, size).wait();
		myQ.memcpy(cHz_r, d_cHz_r, size).wait();
		myQ.memcpy(cEx_i, d_cEx_i, size).wait();
		myQ.memcpy(cEy_i, d_cEy_i, size).wait();
		myQ.memcpy(cEz_i, d_cEz_i, size).wait();
		myQ.memcpy(cHx_i, d_cHx_i, size).wait();
		myQ.memcpy(cHy_i, d_cHy_i, size).wait();
		myQ.memcpy(cHz_i, d_cHz_i, size).wait();
	}
#endif

	// free
	memfree2();
}
