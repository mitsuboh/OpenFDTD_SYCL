/*
solve.cu (CUDA + MPI)
*/

#include "ofd.h"
#include "ofd_cuda.h"
#include "ofd_prototype.h"

static void setup_cuda_mpi();
static void copy_to_host();

void solve(int io, double *tdft, FILE *fp)
{
	double fmax[] = {0, 0};
	char   str[BUFSIZ];
	int    converged = 0;

	// setup boundary index (MPI)
	setup_mpi();

	// setup host memory
	setup_host();

	// setup (GPU)
	if (GPU) {
		setup_gpu();
		setup_cuda_mpi();
	}

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
			murH(numMurHx, (GPU ? d_fMurHx : fMurHx), Hx);
			murH(numMurHy, (GPU ? d_fMurHy : fMurHy), Hy);
			murH(numMurHz, (GPU ? d_fMurHz : fMurHz), Hz);
		}
		else if (iABC == 1) {
			pmlHx();
			pmlHy();
			pmlHz();
		}

		// PBC H
		if (PBCx) {
			if (Npx > 1) {
				comm_cuda_X(1);
			}
			else {
				pbcx();
			}
		}
		if (PBCy) {
			if (Npy > 1) {
				comm_cuda_Y(1);
			}
			else {
				pbcy();
			}
		}
		if (PBCz) {
			if (Npz > 1) {
				comm_cuda_Z(1);
			}
			else {
				pbcz();
			}
		}

		// share boundary H (MPI)
		if (Npx > 1) {
			comm_cuda_X(0);
		}
		if (Npy > 1) {
			comm_cuda_Y(0);
		}
		if (Npz > 1) {
			comm_cuda_Z(0);
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
		if (GPU) cudaDeviceSynchronize();
		const double t0 = comm_cputime();
		dftNear3d(itime);
		if (GPU) cudaDeviceSynchronize();
		*tdft += comm_cputime() - t0;

		// average and convergence
		if ((itime % Solver.nout == 0) || (itime == Solver.maxiter)) {
			// average
			double fsum[2];
			average(fsum);

			// allreduce average (MPI)
			if (commSize > 1) {
				comm_average(fsum);
			}

			// average (post)
			if (commRank == 0) {
				Eiter[Niter] = fsum[0];
				Hiter[Niter] = fsum[1];
				Niter++;
			}

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

	// copy point from device to host
	if (GPU) {
		copy_to_host();
	}

	// free
	memfree2_gpu();

	// copy near3d from device to host
	memcopy3_gpu();

	// free
	memfree3_gpu();

	// MPI : send to root
	if (commSize > 1) {
		// feed waveform
		if (NFeed) {
			comm_feed();
		}

		// point waveform
		if (NPoint) {
			comm_point();
		}

		// near3d
		if (NFreq2) {
			comm_near3d();
		}
	}
}


// setup
static void setup_cuda_mpi()
{
	size_t size;
	//printf("%d %d %d %d %d %d %d %d\n", commSize, commRank, bid.numhy_x, bid.numhz_x, bid.numhz_y, bid.numhx_y, bid.numhx_z, bid.numhy_z); fflush(stdout);

	// X boundary
	size = Bid.numhy_x * sizeof(real_t);
	cuda_malloc(GPU, UM, (void **)&d_Sendbuf_hy_x, size);
	cuda_malloc(GPU, UM, (void **)&d_Recvbuf_hy_x, size);

	size = Bid.numhz_x * sizeof(real_t);
	cuda_malloc(GPU, UM, (void **)&d_Sendbuf_hz_x, size);
	cuda_malloc(GPU, UM, (void **)&d_Recvbuf_hz_x, size);

	// Y boundary
	size = Bid.numhz_y * sizeof(real_t);
	cuda_malloc(GPU, UM, (void **)&d_Sendbuf_hz_y, size);
	cuda_malloc(GPU, UM, (void **)&d_Recvbuf_hz_y, size);

	size = Bid.numhx_y * sizeof(real_t);
	cuda_malloc(GPU, UM, (void **)&d_Sendbuf_hx_y, size);
	cuda_malloc(GPU, UM, (void **)&d_Recvbuf_hx_y, size);

	// Z boundary
	size = Bid.numhx_z * sizeof(real_t);
	cuda_malloc(GPU, UM, (void **)&d_Sendbuf_hx_z, size);
	cuda_malloc(GPU, UM, (void **)&d_Recvbuf_hx_z, size);

	size = Bid.numhy_z * sizeof(real_t);
	cuda_malloc(GPU, UM, (void **)&d_Sendbuf_hy_z, size);
	cuda_malloc(GPU, UM, (void **)&d_Recvbuf_hy_z, size);

	// block
	bufBlock = dim3(16, 16);

	// parameter
	//cudaMemcpyToSymbol(d_bid, &bid, sizeof(bid_t));
}


// copy from device to host
static void copy_to_host()
{
	if (NFeed) {
		cuda_memcpy(GPU, VFeed, d_VFeed, Feed_size, cudaMemcpyDeviceToHost);
		cuda_memcpy(GPU, IFeed, d_IFeed, Feed_size, cudaMemcpyDeviceToHost);
	}

	if (NPoint) {
		cuda_memcpy(GPU, VPoint, d_VPoint, Point_size, cudaMemcpyDeviceToHost);
	}
}
