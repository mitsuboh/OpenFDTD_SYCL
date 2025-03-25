/*
updateHz.c

update Hz
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

static void updateHz_f_vector(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			Hz[n] = K1Hz[n] * Hz[n]
			      - K2Hz[n] * (RXc[i] * (Ey[n + Ni] - Ey[n])
			                 - RYc[j] * (Ex[n + Nj] - Ex[n]));
			n++;
		}
	}
	}
}


static void updateHz_f_no_vector(void)
{
	assert(Nk == 1);

#ifndef _ONEAPI
	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			const int64_t m = iHz[n];
			Hz[n] = D1[m] * Hz[n]
			      - D2[m] * (RXc[i] * (Ey[n + Ni] - Ey[n])
			               - RYc[j] * (Ex[n + Nj] - Ex[n]));
			n++;
		}
	}
	}

#else // _ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 0, updateBlock[0]),
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
		auto Hz = ::Hz;
		auto Ex = ::Ex;
		auto Ey = ::Ey;
		auto iHz = ::d_iHz;
		auto D1 = ::d_D1;
		auto D2 = ::d_D2;
		auto RXc = ::d_RXc;
		auto RYc = ::d_RYc;
		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i < iMax) &&
					(j < jMax) &&
					(k <= kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t m = iHz[n];
					const int64_t n1 = n + Ni;
					const int64_t n2 = n + Nj;
					Hz[n] = D1[m] * Hz[n]
			     			 - D2[m] * (RXc[i] * (Ey[n1] - Ey[n])
			               		     - RYc[j] * (Ex[n2] - Ex[n]));
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}


static void updateHz_p_vector(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			real_t fi, dfi;
			finc(Xc[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.hi[2], Planewave.ai, Dt, &fi, &dfi);
			Hz[n] = K1Hz[n] * Hz[n]
			      - K2Hz[n] * (RXc[i] * (Ey[n + Ni] - Ey[n])
			                 - RYc[j] * (Ex[n + Nj] - Ex[n]))
			      - (K1Hz[n] - K2Hz[n]) * dfi
			      - (1 - K1Hz[n]) * fi;
			n++;
		}
	}
	}
}


static void updateHz_p_no_vector(double t)
{
	assert(Nk == 1);

#ifndef _ONEAPI
	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <= kMax; k++) {
			const int64_t m = iHz[n];
			if (m == 0) {
				Hz[n] -= RXc[i] * (Ey[n + Ni] - Ey[n])
				       - RYc[j] * (Ex[n + Nj] - Ex[n]);
			}
			else {
				real_t fi, dfi;
				finc(Xc[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.hi[2], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Hz[n] = -fi;
				}
				else {
					Hz[n] = D1[m] * Hz[n]
					      - D2[m] * (RXc[i] * (Ey[n + Ni] - Ey[n])
					               - RYc[j] * (Ex[n + Nj] - Ex[n]))
					      - (D1[m] - D2[m]) * dfi
					      - (1 - D1[m]) * fi;
				}
			}
			n++;
		}
	}
	}

#else  // _ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 0, updateBlock[0]),
		CEIL(jMax - jMin + 0, updateBlock[1]),
		CEIL(kMax - kMin + 1, updateBlock[2]));
	sycl::range<3> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto s_t = (real_t)t;
		auto SPlanewave = ::d_SPlanewave;
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
		auto Hz = ::Hz;
		auto Ex = ::Ex;
		auto Ey = ::Ey;
		auto iHz = ::d_iHz;
		auto s_Xc = ::s_Xc;
		auto s_Yc = ::s_Yc;
		auto s_Zn = ::s_Zn;
		auto D1 = ::d_D1;
		auto D2 = ::d_D2;
		auto RXc = ::d_RXc;
		auto RYc = ::d_RYc;
		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i < iMax) &&
					(j < jMax) &&
					(k <= kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t m = iHz[n];
					const int64_t n1 = n + Ni;
					const int64_t n2 = n + Nj;
					if (m == 0) {
						Hz[n] -= RXc[i] * (Ey[n1] - Ey[n])
							- RYc[i] * (Ex[n2] - Ex[n]);
					}
					else {
						real_t fi, dfi;
						finc_s(s_Xc[i], s_Yc[j], s_Zn[k], s_t, SPlanewave->r0, SPlanewave->ri, SPlanewave->hi[2], SPlanewave->ai, s_Dt, &fi, &dfi);

						if (m == PEC) {
							Hz[n] = -fi;
						}
						else {
							Hz[n] = D1[m] * Hz[n]
								- D2[m] * (RXc[i] * (Ey[n1] - Ey[n])
									- RYc[j] * (Ex[n2] - Ex[n]))
								- (D1[m] - D2[m]) * dfi
								- (1 - D1[m]) * fi;
						}
					}
				}
			});
		});

	myQ.wait();
#endif // _ONEAPI
}


void updateHz(double t)
{
	if (NFeed) {
		if (VECTOR) {
			updateHz_f_vector();
		}
		else {
			updateHz_f_no_vector();
		}
	}
	else if (IPlanewave) {
		if (VECTOR) {
			updateHz_p_vector(t);
		}
		else {
			updateHz_p_no_vector(t);
		}
	}
}
