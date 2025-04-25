/*
pmlEx.c (OpenMP)

PML for Ex
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void pmlEx(void)
{
	const int ly = cPML.l;
	const int lz = cPML.l;

#ifndef _ONEAPI
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
#ifdef __CLANG_FUJITSU
#pragma clang loop vectorize(assume_safety)
#endif
	for (n = 0; n < numPmlEx; n++) {
		const int  i = fPmlEx[n].i;
		const int  j = fPmlEx[n].j;
		const int  k = fPmlEx[n].k;
		const id_t m = fPmlEx[n].m;

		const real_t dhz = HZ(i, j, k) - HZ(i, j - 1, k);
		const real_t ry = RYn[MIN(MAX(j, 0), Ny    )] * rPmlE[m];
		Exy[n] = (Exy[n] + (ry * dhz)) / (1 + (gPmlYn[j + ly] * rPml[m]));

		const real_t dhy = HY(i, j, k) - HY(i, j, k - 1);
		const real_t rz = RZn[MIN(MAX(k, 0), Nz    )] * rPmlE[m];
		Exz[n] = (Exz[n] - (rz * dhy)) / (1 + (gPmlZn[k + lz] * rPml[m]));

		EX(i, j, k) = Exy[n] + Exz[n];
	}
#else //_ONEAPI

	const int pmlBlock = 256; 
	sycl::range<1> updateBlock = sycl::range<1>(pmlBlock);
	sycl::range<1> grid(CEIL(numPmlEx, pmlBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto numPmlEx = ::numPmlEx;
		auto fPmlEx = ::d_fPmlEx;
		auto Hz = ::Hz;
		auto Hy = ::Hy;
		auto RZn = :: d_RZn;
		auto RYn = :: d_RYn;
		auto rPmlE = ::d_rPmlE;
		auto rPml = ::d_rPml;
		auto gPmlYn = ::d_gPmlYn;
		auto gPmlZn = ::d_gPmlZn;

		auto Exy = ::Exy;
		auto Exz = ::Exz;
		auto Ex = ::Ex;
		auto Nz = ::Nz;
		auto Ny = ::Ny;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto N0 = ::N0;


		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < numPmlEx){
					const int  i = fPmlEx[n].i;
					const int  j = fPmlEx[n].j;
					const int  k = fPmlEx[n].k;
					const id_t m = fPmlEx[n].m;

					const real_t dhz = HZ(i, j, k) - HZ(i, j - 1, k);
					const real_t ry = RYn[MIN(MAX(j, 0), Ny    )] * rPmlE[m];
					Exy[n] = (Exy[n] + (ry * dhz)) / (1 + (gPmlYn[j + ly] * rPml[m]));

					const real_t dhy = HY(i, j, k) - HY(i, j, k - 1);
					const real_t rz = RZn[MIN(MAX(k, 0), Nz    )] * rPmlE[m];
					Exz[n] = (Exz[n] - (rz * dhy)) / (1 + (gPmlZn[k + lz] * rPml[m]));

					EX(i, j, k) = Exy[n] + Exz[n];
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}

