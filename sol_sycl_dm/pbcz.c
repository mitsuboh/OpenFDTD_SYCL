/*
pbcz.c

PBC for Z boundary
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void pbcz(void)
{
	const int id1 = -1;
	const int id2 = 0;
	const int id3 = Nz - 1;
	const int id4 = Nz;
#ifndef _ONEAPI
	int i;

	// Hx
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin - 0; i <= iMax; i++) {
	for (int j = jMin - 1; j <= jMax; j++) {
		HX(i, j, id1) = HX(i, j, id3);
		HX(i, j, id4) = HX(i, j, id2);
	}
	}
#else //_ONEAPI
	const int pbcBlock = 16;
	sycl::range<2> updateBlock = sycl::range<2>(pbcBlock,pbcBlock);
	sycl::range<2> grid_hx(CEIL(iMax - iMin + 1, updateBlock[0]),
		             CEIL(jMax - jMin + 2, updateBlock[1]));
	sycl::range<2> all_grid_x = grid_hx * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto jMin = ::jMin;
		auto iMax = ::iMax;
		auto jMax = ::jMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Hx = ::Hx;
		hndl.parallel_for(
			sycl::nd_range<2>(all_grid_x, updateBlock),
			[=](sycl::nd_item<2> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin - 1 + idx.get_global_id(1);
				if((i <= iMax) && (j <= jMax)){
					HX(i, j, id1) = HX(i, j, id3);
					HX(i, j, id4) = HX(i, j, id2);
				 }
			});
		});
	myQ.wait();
#endif
#ifndef _ONEAPI

	// Hy
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin - 1; i <= iMax; i++) {
	for (int j = jMin - 0; j <= jMax; j++) {
		HY(i, j, id1) = HY(i, j, id3);
		HY(i, j, id4) = HY(i, j, id2);
	}
	}
#else //_ONEAPI
	sycl::range<2> grid_hy(CEIL(iMax - iMin + 2, updateBlock[0]),
		             CEIL(jMax - jMin + 1, updateBlock[1]));
	sycl::range<2> all_grid_y = grid_hy * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto jMin = ::jMin;
		auto iMax = ::iMax;
		auto jMax = ::jMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Hy = ::Hy;
		hndl.parallel_for(
			sycl::nd_range<2>(all_grid_y, updateBlock),
			[=](sycl::nd_item<2> idx) {
				auto i = iMin - 1 + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				if((i <= iMax) && (j <= jMax)){
					HY(i, j, id1) = HY(i, j, id3);
					HY(i, j, id4) = HY(i, j, id2);
				 }
			});
		});
	myQ.wait();
#endif
}
