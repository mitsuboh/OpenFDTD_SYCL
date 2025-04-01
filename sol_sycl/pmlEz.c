/*
pmlEz.c (OpenMP)

PML for Ez
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void pmlEz(void)
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
	for (n = 0; n < numPmlEz; n++) {
		const int  i = fPmlEz[n].i;
		const int  j = fPmlEz[n].j;
		const int  k = fPmlEz[n].k;
		const id_t m = fPmlEz[n].m;

		const real_t dhy = HY(i, j, k) - HY(i - 1, j, k);
		const real_t rx = RXn[MIN(MAX(i, 0), Nx    )] * rPmlE[m];
		Ezx[n] = (Ezx[n] + (rx * dhy)) / (1 + (gPmlXn[i + lx] * rPml[m]));

		const real_t dhx = HX(i, j, k) - HX(i, j - 1, k);
		const real_t ry = RYn[MIN(MAX(j, 0), Ny    )] * rPmlE[m];
		Ezy[n] = (Ezy[n] - (ry * dhx)) / (1 + (gPmlYn[j + ly] * rPml[m]));

		EZ(i, j, k) = Ezx[n] + Ezy[n];
	}
#else //_ONEAPI
	const int pmlBlock = 256; 
	sycl::range<1> updateBlock = sycl::range<1>(pmlBlock);
	sycl::range<1> grid(CEIL(numPmlEz, pmlBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto numPmlEz = ::numPmlEz;
		auto fPmlEz = ::fPmlEz;
		auto Hx = ::Hx;
		auto Hy = ::Hy;
		auto RXn = :: RXn;
		auto RYn = :: RYn;
		auto rPmlE = ::rPmlE;
		auto rPml = ::rPml;
		auto gPmlXn = ::gPmlXn;
		auto gPmlYn = ::gPmlYn;

		auto Ezx = ::Ezx;
		auto Ezy = ::Ezy;
		auto Ez = ::Ez;
		auto Nx = ::Nx;
		auto Ny = ::Ny;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto N0 = ::N0;


		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < numPmlEz){
		const int  i = fPmlEz[n].i;
		const int  j = fPmlEz[n].j;
		const int  k = fPmlEz[n].k;
		const id_t m = fPmlEz[n].m;

		const real_t dhy = HY(i, j, k) - HY(i - 1, j, k);
		const real_t rx = RXn[MIN(MAX(i, 0), Nx    )] * rPmlE[m];
		Ezx[n] = (Ezx[n] + (rx * dhy)) / (1 + (gPmlXn[i + lx] * rPml[m]));

		const real_t dhx = HX(i, j, k) - HX(i, j - 1, k);
		const real_t ry = RYn[MIN(MAX(j, 0), Ny    )] * rPmlE[m];
		Ezy[n] = (Ezy[n] - (ry * dhx)) / (1 + (gPmlYn[j + ly] * rPml[m]));

		EZ(i, j, k) = Ezx[n] + Ezy[n];
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}
