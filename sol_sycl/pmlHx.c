/*
pmlHx.c (OpenMP)

PML for Hx
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void pmlHx(void)
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
	for (n = 0; n < numPmlHx; n++) {
		const int  i = fPmlHx[n].i;
		const int  j = fPmlHx[n].j;
		const int  k = fPmlHx[n].k;
		const id_t m = fPmlHx[n].m;

		const real_t dez = EZ(i, j + 1, k) - EZ(i, j, k);
		const real_t ry = RYc[MIN(MAX(j, 0), Ny - 1)] * rPmlH[m];
		Hxy[n] = (Hxy[n] - (ry * dez)) / (1 + (gPmlYc[j + ly] * rPml[m]));

		const real_t dey = EY(i, j, k + 1) - EY(i, j, k);
		const real_t rz = RZc[MIN(MAX(k, 0), Nz - 1)] * rPmlH[m];
		Hxz[n] = (Hxz[n] + (rz * dey)) / (1 + (gPmlZc[k + lz] * rPml[m]));

		HX(i, j, k) = Hxy[n] + Hxz[n];
	}

#else //_ONEAPI
	const int pmlBlock = 256; 
	sycl::range<1> updateBlock = sycl::range<1>(pmlBlock);
	sycl::range<1> grid(CEIL(numPmlHx, pmlBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto numPmlHx = ::numPmlHx;
		auto fPmlHx = ::fPmlHx;
		auto Ez = ::Ez;
		auto Ey = ::Ey;
		auto RZc = :: RZc;
		auto RYc = :: RYc;
		auto rPmlH = ::rPmlH;
		auto rPml = ::rPml;
		auto gPmlYc = ::gPmlYc;
		auto gPmlZc = ::gPmlZc;

		auto Hxy = ::Hxy;
		auto Hxz = ::Hxz;
		auto Hx = ::Hx;
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
				if (n < numPmlHx){
		const int  i = fPmlHx[n].i;
		const int  j = fPmlHx[n].j;
		const int  k = fPmlHx[n].k;
		const id_t m = fPmlHx[n].m;

		const real_t dez = EZ(i, j + 1, k) - EZ(i, j, k);
		const real_t ry = RYc[MIN(MAX(j, 0), Ny - 1)] * rPmlH[m];
		Hxy[n] = (Hxy[n] - (ry * dez)) / (1 + (gPmlYc[j + ly] * rPml[m]));

		const real_t dey = EY(i, j, k + 1) - EY(i, j, k);
		const real_t rz = RZc[MIN(MAX(k, 0), Nz - 1)] * rPmlH[m];
		Hxz[n] = (Hxz[n] + (rz * dey)) / (1 + (gPmlZc[k + lz] * rPml[m]));

		HX(i, j, k) = Hxy[n] + Hxz[n];
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}
