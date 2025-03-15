/*
dftNear3d.c (OpenMP)

DFT of near field
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void dftNear3d(int itime)
{
	for (int ifreq = 0; ifreq < NFreq2; ifreq++) {
		const int64_t n0 = ifreq * NN;
		const int id = (itime * NFreq2) + ifreq;

		const float ef_r = (float)cEdft[id].r;
		const float ef_i = (float)cEdft[id].i;
		const float hf_r = (float)cHdft[id].r;
		const float hf_i = (float)cHdft[id].i;

#ifdef _ONEAPI
        sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
        sycl::range<3> grid;
        sycl::range<3> all_grid;
#endif
#ifndef _ONEAPI
		int i;
#endif
		// Ex
#ifndef _ONEAPI
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin; i < iMax + 0; i++) {
		for (int j = jMin; j < jMax + 1; j++) {
		for (int k = kMin; k < kMax + 1; k++) {
			const int64_t n = NA(i, j, k);
			cEx_r[n0 + n] += (float)Ex[n] * ef_r;
			cEx_i[n0 + n] += (float)Ex[n] * ef_i;
		}
		}
		}
#else	// _ONEAPI
	grid = sycl::range<3> (CEIL(iMax - iMin + 0, updateBlock[0]),
		CEIL(jMax - jMin + 1, updateBlock[1]),
		CEIL(kMax - kMin + 1, updateBlock[2]));
	all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto jMin = ::jMin;
		auto kMin = ::kMin;
		auto iMax = ::iMax;
		auto jMax = ::jMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Ex = ::Ex;
		auto cEx_r = ::cEx_r;
		auto cEx_i = ::cEx_i;

		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				const int i = iMin + idx.get_global_id(0);
				const int j = jMin + idx.get_global_id(1);
				const int k = kMin + idx.get_global_id(2);
                                if ((i < iMax) &&
                                        (j <= jMax) &&
                                        (k <= kMax)) {
					const int64_t n = NA(i, j, k);
					cEx_r[n0 + n] += (float)Ex[n] * ef_r;
					cEx_i[n0 + n] += (float)Ex[n] * ef_i;
			}
		});
	});
	myQ.wait();
#endif	// _ONEAPI

		// Ey
#ifndef _ONEAPI
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin; i < iMax + 1; i++) {
		for (int j = jMin; j < jMax + 0; j++) {
		for (int k = kMin; k < kMax + 1; k++) {
			const int64_t n = NA(i, j, k);
			cEy_r[n0 + n] += (float)Ey[n] * ef_r;
			cEy_i[n0 + n] += (float)Ey[n] * ef_i;
		}
		}
		}
#else	// _ONEAPI
//	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	grid = sycl::range<3> (CEIL(iMax - iMin + 1, updateBlock[0]),
		CEIL(jMax - jMin + 0, updateBlock[1]),
		CEIL(kMax - kMin + 1, updateBlock[2]));
	all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto jMin = ::jMin;
		auto kMin = ::kMin;
		auto iMax = ::iMax;
		auto jMax = ::jMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Ey = ::Ey;
		auto cEy_r = ::cEy_r;
		auto cEy_i = ::cEy_i;

		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				const int i = iMin + idx.get_global_id(0);
				const int j = jMin + idx.get_global_id(1);
				const int k = kMin + idx.get_global_id(2);
                                if ((i <= iMax) &&
                                        (j < jMax) &&
                                        (k <= kMax)) {
					const int64_t n = NA(i, j, k);
					cEy_r[n0 + n] += (float)Ey[n] * ef_r;
					cEy_i[n0 + n] += (float)Ey[n] * ef_i;
			}
		});
	});
	myQ.wait();
#endif	// _ONEAPI

		// Ez
#ifndef _ONEAPI
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin; i < iMax + 1; i++) {
		for (int j = jMin; j < jMax + 1; j++) {
		for (int k = kMin; k < kMax + 0; k++) {
			const int64_t n = NA(i, j, k);
			cEz_r[n0 + n] += (float)Ez[n] * ef_r;
			cEz_i[n0 + n] += (float)Ez[n] * ef_i;
		}
		}
		}
#else	// _ONEAPI
//	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	grid = sycl::range<3> (CEIL(iMax - iMin + 1, updateBlock[0]),
		CEIL(jMax - jMin + 1, updateBlock[1]),
		CEIL(kMax - kMin + 0, updateBlock[2]));
	all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto jMin = ::jMin;
		auto kMin = ::kMin;
		auto iMax = ::iMax;
		auto jMax = ::jMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Ez = ::Ez;
		auto cEz_r = ::cEz_r;
		auto cEz_i = ::cEz_i;

		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				const int i = iMin + idx.get_global_id(0);
				const int j = jMin + idx.get_global_id(1);
				const int k = kMin + idx.get_global_id(2);
                                if ((i <= iMax) &&
                                        (j <= jMax) &&
                                        (k < kMax)) {
					const int64_t n = NA(i, j, k);
					cEz_r[n0 + n] += (float)Ez[n] * ef_r;
					cEz_i[n0 + n] += (float)Ez[n] * ef_i;
			}
		});
	});
	myQ.wait();
#endif	// _ONEAPI
		// Hx
#ifndef _ONEAPI
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin - 1; i < iMax + 2; i++) {
		for (int j = jMin - 1; j < jMax + 1; j++) {
		for (int k = kMin - 1; k < kMax + 1; k++) {
			const int64_t n = NA(i, j, k);
			cHx_r[n0 + n] += (float)Hx[n] * hf_r;
			cHx_i[n0 + n] += (float)Hx[n] * hf_i;
		}
		}
		}
#else	// _ONEAPI
	grid = sycl::range<3> (CEIL(iMax - iMin + 3, updateBlock[0]),
		CEIL(jMax - jMin + 2, updateBlock[1]),
		CEIL(kMax - kMin + 2, updateBlock[2]));
	all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto jMin = ::jMin;
		auto kMin = ::kMin;
		auto iMax = ::iMax;
		auto jMax = ::jMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Hx = ::Hx;
		auto cHx_r = ::cHx_r;
		auto cHx_i = ::cHx_i;

		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				const int i = iMin - 1 + idx.get_global_id(0);
				const int j = jMin - 1 + idx.get_global_id(1);
				const int k = kMin - 1 + idx.get_global_id(2);
                                if ((i <= iMax + 1) &&
                                        (j < jMax + 1) &&
                                        (k < kMax + 1)) {
					const int64_t n = NA(i, j, k);
					cHx_r[n0 + n] += (float)Hx[n] * hf_r;
					cHx_i[n0 + n] += (float)Hx[n] * hf_i;
			}
		});
	});
	myQ.wait();
#endif	// _ONEAPI

		// Hy
#ifndef _ONEAPI
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin - 1; i < iMax + 1; i++) {
		for (int j = jMin - 1; j < jMax + 2; j++) {
		for (int k = kMin - 1; k < kMax + 1; k++) {
			const int64_t n = NA(i, j, k);
			cHy_r[n0 + n] += (float)Hy[n] * hf_r;
			cHy_i[n0 + n] += (float)Hy[n] * hf_i;
		}
		}
		}
#else	// _ONEAPI
	grid = sycl::range<3> (CEIL(iMax - iMin + 2, updateBlock[0]),
		CEIL(jMax - jMin + 3, updateBlock[1]),
		CEIL(kMax - kMin + 2, updateBlock[2]));
	all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto jMin = ::jMin;
		auto kMin = ::kMin;
		auto iMax = ::iMax;
		auto jMax = ::jMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Hy = ::Hy;
		auto cHy_r = ::cHy_r;
		auto cHy_i = ::cHy_i;

		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				const int i = iMin - 1 + idx.get_global_id(0);
				const int j = jMin - 1 + idx.get_global_id(1);
				const int k = kMin - 1 + idx.get_global_id(2);
                                if ((i < iMax + 1) &&
                                        (j <= jMax + 1) &&
                                        (k < kMax + 1)) {
					const int64_t n = NA(i, j, k);
					cHy_r[n0 + n] += (float)Hy[n] * hf_r;
					cHy_i[n0 + n] += (float)Hy[n] * hf_i;
			}
		});
	});
	myQ.wait();
#endif	// _ONEAPI

		// Hz
#ifndef _ONEAPI
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (    i = iMin - 1; i < iMax + 1; i++) {
		for (int j = jMin - 1; j < jMax + 1; j++) {
		for (int k = kMin - 1; k < kMax + 2; k++) {
			const int64_t n = NA(i, j, k);
			cHz_r[n0 + n] += (float)Hz[n] * hf_r;
			cHz_i[n0 + n] += (float)Hz[n] * hf_i;
		}
		}
		}
#else	// _ONEAPI
	grid = sycl::range<3> (CEIL(iMax - iMin + 2, updateBlock[0]),
		CEIL(jMax - jMin + 2, updateBlock[1]),
		CEIL(kMax - kMin + 3, updateBlock[2]));
	all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto iMin = ::iMin;
		auto jMin = ::jMin;
		auto kMin = ::kMin;
		auto iMax = ::iMax;
		auto jMax = ::jMax;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Hz = ::Hz;
		auto cHz_r = ::cHz_r;
		auto cHz_i = ::cHz_i;

		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				const int i = iMin - 1 + idx.get_global_id(0);
				const int j = jMin - 1 + idx.get_global_id(1);
				const int k = kMin - 1 + idx.get_global_id(2);
                                if ((i < iMax + 1) &&
                                        (j < jMax + 1) &&
                                        (k <= kMax + 1)) {
					const int64_t n = NA(i, j, k);
					cHz_r[n0 + n] += (float)Hz[n] * hf_r;
					cHz_i[n0 + n] += (float)Hz[n] * hf_i;
			}
		});
	});
	myQ.wait();
#endif	// _ONEAPI
	}
}
