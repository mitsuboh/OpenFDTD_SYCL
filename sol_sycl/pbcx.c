/*
pbcx.c

PBC for X boundary
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void pbcx(void)
{
	const int id1 = -1;
	const int id2 = 0;
	const int id3 = Nx - 1;
	const int id4 = Nx;
#ifndef _ONEAPI
	int j;

	// Hy
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    j = jMin - 0; j <= jMax; j++) {
	for (int k = kMin - 1; k <= kMax; k++) {
		HY(id1, j, k) = HY(id3, j, k);
		HY(id4, j, k) = HY(id2, j, k);
	}
	}
#else //_ONEAPI
	const int pbcBlock = 16;
	sycl::range<2> updateBlock = sycl::range<2>(pbcBlock,pbcBlock);
	sycl::range<2> grid_hy(CEIL(jMax - jMin + 1, updateBlock[0]),
		             CEIL(kMax - kMin + 2, updateBlock[1]));
	sycl::range<2> all_grid_y = grid_hy * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto jMin = ::jMin;
		auto kMin = ::kMin;
		auto jMax = ::jMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Hy = ::Hy;
		hndl.parallel_for(
			sycl::nd_range<2>(all_grid_y, updateBlock),
			[=](sycl::nd_item<2> idx) {
				auto j = jMin + idx.get_global_id(0);
				auto k = kMin - 1 + idx.get_global_id(1);
				if((j <= jMax) && (k <= kMax)){
					HY(id1, j, k) = HY(id3, j, k);
					HY(id4, j, k) = HY(id2, j, k);
				 }
			});
		});
	myQ.wait();
#endif
#ifndef _ONEAPI
	// Hz
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    j = jMin - 1; j <= jMax; j++) {
	for (int k = kMin - 0; k <= kMax; k++) {
		HZ(id1, j, k) = HZ(id3, j, k);
		HZ(id4, j, k) = HZ(id2, j, k);
	}
	}
#else //_ONEAPI
	sycl::range<2> grid_hz(CEIL(jMax - jMin + 2, updateBlock[0]),
		             CEIL(kMax - kMin + 1, updateBlock[1]));
	sycl::range<2> all_grid_z = grid_hz * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto jMin = ::jMin;
		auto kMin = ::kMin;
		auto jMax = ::jMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Hz = ::Hz;
		hndl.parallel_for(
			sycl::nd_range<2>(all_grid_z, updateBlock),
			[=](sycl::nd_item<2> idx) {
				auto j = jMin - 1 + idx.get_global_id(0);
				auto k = kMin + idx.get_global_id(1);
				if((j <= jMax) && (k <= kMax)){
					HZ(id1, j, k) = HZ(id3, j, k); 
					HZ(id4, j, k) = HZ(id2, j, k);
				 }
			});
		});
	myQ.wait();
#endif
}
