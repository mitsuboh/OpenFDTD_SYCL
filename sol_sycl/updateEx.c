/*
updateEx.c

update Ex
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

static void updateEx_f_vector(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			Ex[n] = K1Ex[n] * Ex[n]
			      + K2Ex[n] * (RYn[j] * (Hz[n] - Hz[n - Nj])
			                 - RZn[k] * (Hy[n] - Hy[n - Nk]));
			n++;
		}
	}
	}
}


static void updateEx_f_no_vector(void)
{
	assert(Nk == 1);

#ifndef _ONEAPI
	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			const int64_t m = iEx[n];
			Ex[n] = C1[m] * Ex[n]
			      + C2[m] * (RYn[j] * (Hz[n] - Hz[n - Nj])
			               - RZn[k] * (Hy[n] - Hy[n - Nk]));
			n++;
		}
	}
	}

#else //_ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 0, updateBlock[0]),
		CEIL(jMax - jMin + 1, updateBlock[1]),
		CEIL(kMax - kMin + 1, updateBlock[2]));
	sycl::range<3> all_grid = grid * updateBlock;

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
		auto Hy = ::Hy;
		auto Hz = ::Hz;
		auto iEx = ::iEx;
		auto C1 = ::C1;
		auto C2 = ::C2;
		auto RYn = ::RYn;
		auto RZn = ::RZn;
		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i < iMax) &&
					(j <= jMax) &&
					(k <= kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t m = iEx[n];
					const int64_t n1 = n - Nj;
					const int64_t n2 = n - Nk;
					Ex[n] = C1[m] * Ex[n]
						+ C2[m] * (RYn[j] * (Hz[n] - Hz[n1])
							- RZn[k] * (Hy[n] - Hy[n2]));
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}


static void updateEx_p_vector(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			real_t fi, dfi;
			finc(Xc[i], Yn[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[0], Planewave.ai, Dt, &fi, &dfi);
			Ex[n] = K1Ex[n] * Ex[n]
			      + K2Ex[n] * (RYn[j] * (Hz[n] - Hz[n - Nj])
			                 - RZn[k] * (Hy[n] - Hy[n - Nk]))
			      - (K1Ex[n] - K2Ex[n]) * dfi
			      - (1 - K1Ex[n]) * fi;
			n++;
		}
	}
	}
}


static void updateEx_p_no_vector(double t)
{
	assert(Nk == 1);

#ifndef _ONEAPI
	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			const int64_t m = iEx[n];
			if (m == 0) {
				Ex[n] += RYn[j] * (Hz[n] - Hz[n - Nj])
				       - RZn[k] * (Hy[n] - Hy[n - Nk]);
			}
			else {
				real_t fi, dfi;
				finc(Xc[i], Yn[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[0], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Ex[n] = -fi;
				}
				else {
					Ex[n] = C1[m] * Ex[n]
					      + C2[m] * (RYn[j] * (Hz[n] - Hz[n - Nj])
					               - RZn[k] * (Hy[n] - Hy[n - Nk]))
					      - (C1[m] - C2[m]) * dfi
					      - (1 - C1[m]) * fi;
				}
			}
			n++;
		}
	}
	}

#else //_ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 0, updateBlock[0]),
		CEIL(jMax - jMin + 1, updateBlock[1]),
		CEIL(kMax - kMin + 1, updateBlock[2]));
	sycl::range<3> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto s_t = (real_t)t;
		auto SPlanewave = ::SPlanewave;
		real_t s_Dt = (real_t)Dt;
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
		auto Hy = ::Hy;
		auto Hz = ::Hz;
		auto iEx = ::iEx;
		auto s_Xc = ::s_Xc;
		auto s_Yn = ::s_Yn;
		auto s_Zn = ::s_Zn;
		auto C1 = ::C1;
		auto C2 = ::C2;
		auto RYn = ::RYn;
		auto RZn = ::RZn;
		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i < iMax) &&
					(j <= jMax) &&
					(k <= kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t m = iEx[n];
					const int64_t n1 = n - Nj;
					const int64_t n2 = n - Nk;
					if (m == 0) {
						Ex[n] += RYn[j] * (Hz[n] - Hz[n1])
							- RZn[k] * (Hy[n] - Hy[n2]);
					}
					else {
						real_t fi, dfi;
						finc_s(s_Xc[i], s_Yn[j], s_Zn[k], s_t, SPlanewave.r0, SPlanewave.ri, SPlanewave.ei[0], SPlanewave.ai, s_Dt, &fi, &dfi);
						if (m == PEC) {
							Ex[n] = -fi;
						}
						else {
							Ex[n] = C1[m] * Ex[n]
								+ C2[m] * (RYn[j] * (Hz[n] - Hz[n1])
								- RZn[k] * (Hy[n] - Hy[n2]))
								- (C1[m] - C2[m]) * dfi
								- (1 - C1[m]) * fi;
						}
					}
				}
			});
		});

	myQ.wait();
#endif // _ONEAPI
}


void updateEx(double t)
{
	if (NFeed) {
		if (VECTOR) {
			updateEx_f_vector();
		}
		else {
			updateEx_f_no_vector();
		}
	}
	else if (IPlanewave) {
		if (VECTOR) {
			updateEx_p_vector(t);
		}
		else {
			updateEx_p_no_vector(t);
		}
	}
}
