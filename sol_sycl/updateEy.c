/*
updateEy.c

update Ey
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

static void updateEy_f_vector(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			Ey[n] = K1Ey[n] * Ey[n]
			      + K2Ey[n] * (RZn[k] * (Hx[n] - Hx[n - Nk])
			                 - RXn[i] * (Hz[n] - Hz[n - Ni]));
			n++;
		}
	}
	}
}


static void updateEy_f_no_vector(void)
{
	assert(Nk == 1);

#ifndef _ONEAPI
	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			const int64_t m = iEy[n];
			Ey[n] = C1[m] * Ey[n]
			      + C2[m] * (RZn[k] * (Hx[n] - Hx[n - Nk])
			               - RXn[i] * (Hz[n] - Hz[n - Ni]));
			n++;
		}
	}
	}
#else  // _ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 1, updateBlock[0]),
		CEIL(jMax - jMin + 0, updateBlock[1]),
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
		auto Hx = ::Hx;
		auto Ey = ::Ey;
		auto Hz = ::Hz;
		auto iEy = ::iEy;
		auto C1 = ::C1;
		auto C2 = ::C2;
		auto RZn = ::RZn;
		auto RXn = ::RXn;
		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i <= iMax) &&
					(j < jMax) &&
					(k <= kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t m = iEy[n];
					const int64_t n1 = n - Nk;
					const int64_t n2 = n - Ni;
					Ey[n] = C1[m] * Ey[n]
						+ C2[m] * (RZn[k] * (Hx[n] - Hx[n1])
							- RXn[i] * (Hz[n] - Hz[n2]));
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}


static void updateEy_p_vector(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			real_t fi, dfi;
			finc(Xn[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[1], Planewave.ai, Dt, &fi, &dfi);
			Ey[n] = K1Ey[n] * Ey[n]
			      + K2Ey[n] * (RZn[k] * (Hx[n] - Hx[n - Nk])
			                 - RXn[i] * (Hz[n] - Hz[n - Ni]))
			      - (K1Ey[n] - K2Ey[n]) * dfi
			      - (1 - K1Ey[n]) * fi;
			n++;
		}
	}
	}
}


static void updateEy_p_no_vector(double t)
{
	assert(Nk == 1);

#ifndef _ONEAPI
	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			const int64_t m = iEy[n];
			if (m == 0) {
				Ey[n] += RZn[k] * (Hx[n] - Hx[n - Nk])
				       - RXn[i] * (Hz[n] - Hz[n - Ni]);
			}
			else {
				real_t fi, dfi;
				finc(Xn[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[1], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Ey[n] = -fi;
				}
				else {
					Ey[n] = C1[m] * Ey[n]
					      + C2[m] * (RZn[k] * (Hx[n] - Hx[n - Nk])
					               - RXn[i] * (Hz[n] - Hz[n - Ni]))
					      - (C1[m] - C2[m]) * dfi
					      - (1 - C1[m]) * fi;
				}
			}
			n++;
		}
	}
	}
#else  // _ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 1, updateBlock[0]),
		CEIL(jMax - jMin + 0, updateBlock[1]),
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
		auto Hx = ::Hx;
		auto Ey = ::Ey;
		auto Hz = ::Hz;
		auto iEy = ::iEy;
		auto s_Xn = ::s_Xn;
		auto s_Yc = ::s_Yc;
		auto s_Zn = ::s_Zn;
		auto C1 = ::C1;
		auto C2 = ::C2;
		auto RZn = ::RZn;
		auto RXn = ::RXn;
		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i <= iMax) &&
					(j < jMax) &&
					(k <= kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t m = iEy[n];
					const int64_t n1 = n - Nk;
					const int64_t n2 = n - Ni;
					if (m == 0) {
						Ey[n] += RZn[k] * (Hx[n] - Hx[n1])
							- RXn[i] * (Hz[n] - Hz[n2]);
					}
					else {
						real_t fi, dfi;
						finc_s(s_Xn[i], s_Yc[j], s_Zn[k], s_t, SPlanewave->r0, SPlanewave->ri, SPlanewave->ei[1], SPlanewave->ai, s_Dt, &fi, &dfi);
						if (m == PEC) {
							Ey[n] = -fi;
						}
						else {
							Ey[n] = C1[m] * Ey[n]
								+ C2[m] * (RZn[k] * (Hx[n] - Hx[n1])
									- RXn[i] * (Hz[n] - Hz[n2]))
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


void updateEy(double t)
{
	if (NFeed) {
		if (VECTOR) {
			updateEy_f_vector();
		}
		else {
			updateEy_f_no_vector();
		}
	}
	else if (IPlanewave) {
		if (VECTOR) {
			updateEy_p_vector(t);
		}
		else {
			updateEy_p_no_vector(t);
		}
	}
}
