/*
pbcy.c

PBC for Y boundary
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void pbcy(void)
{
	const int id1 = -1;
	const int id2 = 0;
	const int id3 = Ny - 1;
	const int id4 = Ny;
#ifndef _ONEAPI
	int k;

	// Hz
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    k = kMin - 0; k <= kMax; k++) {
	for (int i = iMin - 1; i <= iMax; i++) {
		HZ(i, id1, k) = HZ(i, id3, k);
		HZ(i, id4, k) = HZ(i, id2, k);
	}
	}
#else //_ONEAPI
	const int pbcBlock = 16;
	sycl::range<2> updateBlock = sycl::range<2>(pbcBlock,pbcBlock);
	sycl::range<2> grid_hz(CEIL(kMax - kMin + 1, updateBlock[0]),
		             CEIL(iMax - iMin + 2, updateBlock[1]));
	sycl::range<2> all_grid_z = grid_hz * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto kMin = ::kMin;
		auto iMax = ::iMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Hz = ::Hz;
		hndl.parallel_for(
			sycl::nd_range<2>(all_grid_z, updateBlock),
			[=](sycl::nd_item<2> idx) {
				auto k = kMin + idx.get_global_id(0);
				auto i = iMin - 1 + idx.get_global_id(1);
				if((k <= kMax) && (i <= iMax)){
					HZ(i, id1, k) = HZ(i, id3, k);
					HZ(i, id4, k) = HZ(i, id2, k);
				 }
			});
		});
	myQ.wait();
#endif
#ifndef _ONEAPI
	// Hx
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    k = kMin - 1; k <= kMax; k++) {
	for (int i = iMin - 0; i <= iMax; i++) {
		HX(i, id1, k) = HX(i, id3, k);
		HX(i, id4, k) = HX(i, id2, k);
	}
	}
}
#else //_ONEAPI
	sycl::range<2> grid_hx(CEIL(kMax - kMin + 2, updateBlock[0]),
		             CEIL(iMax - iMin + 1, updateBlock[1]));
	sycl::range<2> all_grid_x = grid_hx * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto kMin = ::kMin;
		auto iMax = ::iMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Hx = ::Hx;
		hndl.parallel_for(
			sycl::nd_range<2>(all_grid_x, updateBlock),
			[=](sycl::nd_item<2> idx) {
				auto k = kMin - 1 + idx.get_global_id(0);
				auto i = iMin + idx.get_global_id(1);
				if((k <= kMax) && (i <= iMax)){
					HX(i, id1, k) = HX(i, id3, k);
					HX(i, id4, k) = HX(i, id2, k);
				 }
			});
		});
	myQ.wait();
#endif
}
