/*
pmlEy.c (OpenMP)

PML for Ey
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void pmlEy(void)
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
	for (n = 0; n < numPmlEy; n++) {
		const int  i = fPmlEy[n].i;
		const int  j = fPmlEy[n].j;
		const int  k = fPmlEy[n].k;
		const id_t m = fPmlEy[n].m;

		const real_t dhx = HX(i, j, k) - HX(i, j, k - 1);
		const real_t rz = RZn[MIN(MAX(k, 0), Nz    )] * rPmlE[m];
		Eyz[n] = (Eyz[n] + (rz * dhx)) / (1 + (gPmlZn[k + lz] * rPml[m]));

		const real_t dhz = HZ(i, j, k) - HZ(i - 1, j, k);
		const real_t rx = RXn[MIN(MAX(i, 0), Nx    )] * rPmlE[m];
		Eyx[n] = (Eyx[n] - (rx * dhz)) / (1 + (gPmlXn[i + lx] * rPml[m]));

		EY(i, j, k) = Eyz[n] + Eyx[n];
	}
#else //_ONEAPI
	const int pmlBlock = 256; 
	sycl::range<1> updateBlock = sycl::range<1>(pmlBlock);
	sycl::range<1> grid(CEIL(numPmlEy, pmlBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto numPmlEy = ::numPmlEy;
		auto fPmlEy = ::fPmlEy;
		auto Hx = ::Hx;
		auto Hz = ::Hz;
		auto RXn = :: RXn;
		auto RZn = :: RZn;
		auto rPmlE = ::rPmlE;
		auto rPml = ::rPml;
		auto gPmlXn = ::gPmlXn;
		auto gPmlZn = ::gPmlZn;

		auto Eyz = ::Eyz;
		auto Eyx = ::Eyx;
		auto Ey = ::Ey;
		auto Nx = ::Nx;
		auto Nz = ::Nz;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto N0 = ::N0;


		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < numPmlEy){
		const int  i = fPmlEy[n].i;
		const int  j = fPmlEy[n].j;
		const int  k = fPmlEy[n].k;
		const id_t m = fPmlEy[n].m;

		const real_t dhx = HX(i, j, k) - HX(i, j, k - 1);
		const real_t rz = RZn[MIN(MAX(k, 0), Nz    )] * rPmlE[m];
		Eyz[n] = (Eyz[n] + (rz * dhx)) / (1 + (gPmlZn[k + lz] * rPml[m]));

		const real_t dhz = HZ(i, j, k) - HZ(i - 1, j, k);
		const real_t rx = RXn[MIN(MAX(i, 0), Nx    )] * rPmlE[m];
		Eyx[n] = (Eyx[n] - (rx * dhz)) / (1 + (gPmlXn[i + lx] * rPml[m]));

		EY(i, j, k) = Eyz[n] + Eyx[n];
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}
