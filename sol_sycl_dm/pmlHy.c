/*
pmlHy.c (OpenMP)

PML for Hy
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void pmlHy(void)
{
	const int lz = cPML.l;
	const int lx = cPML.l;

#ifndef _ONEAPI
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
#ifdef __CLANG_FUJITSU
#pragma clang loop vectorize(assume_safety)
#endif
	for (n = 0; n < numPmlHy; n++) {
		const int  i = fPmlHy[n].i;
		const int  j = fPmlHy[n].j;
		const int  k = fPmlHy[n].k;
		const id_t m = fPmlHy[n].m;

		const real_t dex = EX(i, j, k + 1) - EX(i, j, k);
		const real_t rz = RZc[MIN(MAX(k, 0), Nz - 1)] * rPmlH[m];
		Hyz[n] = (Hyz[n] - (rz * dex)) / (1 + (gPmlZc[k + lz] * rPml[m]));

		const real_t dez = EZ(i + 1, j, k) - EZ(i, j, k);
		const real_t rx = RXc[MIN(MAX(i, 0), Nx - 1)] * rPmlH[m];
		Hyx[n] = (Hyx[n] + (rx * dez)) / (1 + (gPmlXc[i + lx] * rPml[m]));

		HY(i, j, k) = Hyz[n] + Hyx[n];
	}
#else //_ONEAPI
	const int pmlBlock = 256; 
	sycl::range<1> updateBlock = sycl::range<1>(pmlBlock);
	sycl::range<1> grid(CEIL(numPmlHy, pmlBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto numPmlHy = ::numPmlHy;
		auto fPmlHy = ::d_fPmlHy;
		auto Ez = ::Ez;
		auto Ex = ::Ex;
		auto RZc = :: d_RZc;
		auto RXc = :: d_RXc;
		auto rPmlH = ::d_rPmlH;
		auto rPml = ::d_rPml;
		auto gPmlXc = ::d_gPmlXc;
		auto gPmlZc = ::d_gPmlZc;

		auto Hyz = ::Hyz;
		auto Hyx = ::Hyx;
		auto Hy = ::Hy;
		auto Nz = ::Nz;
		auto Nx = ::Nx;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto N0 = ::N0;

		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < numPmlHy){
		const int  i = fPmlHy[n].i;
		const int  j = fPmlHy[n].j;
		const int  k = fPmlHy[n].k;
		const id_t m = fPmlHy[n].m;

		const real_t dex = EX(i, j, k + 1) - EX(i, j, k);
		const real_t rz = RZc[MIN(MAX(k, 0), Nz - 1)] * rPmlH[m];
		Hyz[n] = (Hyz[n] - (rz * dex)) / (1 + (gPmlZc[k + lz] * rPml[m]));

		const real_t dez = EZ(i + 1, j, k) - EZ(i, j, k);
		const real_t rx = RXc[MIN(MAX(i, 0), Nx - 1)] * rPmlH[m];
		Hyx[n] = (Hyx[n] + (rx * dez)) / (1 + (gPmlXc[i + lx] * rPml[m]));

		HY(i, j, k) = Hyz[n] + Hyx[n];
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}
