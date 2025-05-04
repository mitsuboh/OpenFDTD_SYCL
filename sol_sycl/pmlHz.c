/*
pmlHz.c (OpenMP)

PML for Hz
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void pmlHz(void)
{
	const int lx = cPML.l;
	const int ly = cPML.l;

#ifndef _ONEAPI
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
#ifdef __CLANG_FUJITSU
#pragma clang loop vectorize(assume_safety)
#endif
	for (n = 0; n < numPmlHz; n++) {
		const int  i = fPmlHz[n].i;
		const int  j = fPmlHz[n].j;
		const int  k = fPmlHz[n].k;
		const id_t m = fPmlHz[n].m;

		const real_t dey = EY(i + 1, j, k) - EY(i, j, k);
		const real_t rx = RXc[MIN(MAX(i, 0), Nx - 1)] * rPmlH[m];
		Hzx[n] = (Hzx[n] - (rx * dey)) / (1 + (gPmlXc[i + lx] * rPml[m]));

		const real_t dex = EX(i, j + 1, k) - EX(i, j, k);
		const real_t ry = RYc[MIN(MAX(j, 0), Ny - 1)] * rPmlH[m];
		Hzy[n] = (Hzy[n] + (ry * dex)) / (1 + (gPmlYc[j + ly] * rPml[m]));

		HZ(i, j, k) = Hzx[n] + Hzy[n];
	}
#else //_ONEAPI
	const int pmlBlock = 256; 
	sycl::range<1> updateBlock = sycl::range<1>(pmlBlock);
	sycl::range<1> grid(CEIL(numPmlHz, pmlBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto numPmlHz = ::numPmlHz;
		auto fPmlHz = ::fPmlHz;
		auto Ey = ::Ey;
		auto Ex = ::Ex;
		auto RYc = :: RYc;
		auto RXc = :: RXc;
		auto rPmlH = ::rPmlH;
		auto rPml = ::rPml;
		auto gPmlXc = ::gPmlXc;
		auto gPmlYc = ::gPmlYc;

		auto Hzx = ::Hzx;
		auto Hzy = ::Hzy;
		auto Hz = ::Hz;
		auto Ny = ::Ny;
		auto Nx = ::Nx;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto N0 = ::N0;

		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < numPmlHz){
		const int  i = fPmlHz[n].i;
		const int  j = fPmlHz[n].j;
		const int  k = fPmlHz[n].k;
		const id_t m = fPmlHz[n].m;

		const real_t dey = EY(i + 1, j, k) - EY(i, j, k);
		const real_t rx = RXc[MIN(MAX(i, 0), Nx - 1)] * rPmlH[m];
		Hzx[n] = (Hzx[n] - (rx * dey)) / (1 + (gPmlXc[i + lx] * rPml[m]));

		const real_t dex = EX(i, j + 1, k) - EX(i, j, k);
		const real_t ry = RYc[MIN(MAX(j, 0), Ny - 1)] * rPmlH[m];
		Hzy[n] = (Hzy[n] + (ry * dex)) / (1 + (gPmlYc[j + ly] * rPml[m]));

		HZ(i, j, k) = Hzx[n] + Hzy[n];
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}
