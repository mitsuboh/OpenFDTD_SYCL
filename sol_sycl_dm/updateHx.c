/*
updateHx.c

update Hx
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

static void updateHx_f_vector(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <  kMax; k++) {
			Hx[n] = K1Hx[n] * Hx[n]
			      - K2Hx[n] * (RYc[j] * (Ez[n + Nj] - Ez[n])
			                 - RZc[k] * (Ey[n + Nk] - Ey[n]));
			n++;
		}
	}
	}
}


static void updateHx_f_no_vector(void)
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
		for (int k = kMin; k <  kMax; k++) {
			const int64_t m = iHx[n];
			Hx[n] = D1[m] * Hx[n]
			      - D2[m] * (RYc[j] * (Ez[n + Nj] - Ez[n])
			               - RZc[k] * (Ey[n + Nk] - Ey[n]));
			n++;
		}
	}
	}

#else	// _ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 1, updateBlock[0]),
		CEIL(jMax - jMin + 0, updateBlock[1]),
		CEIL(kMax - kMin + 0, updateBlock[2]));
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
		auto Ez = ::Ez;
		auto iHx = ::d_iHx;
		auto D1 = ::d_D1;
		auto D2 = ::d_D2;
		auto RYc = ::d_RYc;
		auto RZc = ::d_RZc;
		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i <= iMax) &&
					(j < jMax) &&
					(k < kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t m = iHx[n];
					const int64_t n1 = n + Nj;
					const int64_t n2 = n + Nk;
					Hx[n] = D1[m] * Hx[n]
					      - D2[m] * (RYc[j] * (Ez[n1] - Ez[n])
					                    - RZc[k] * (Ey[n2] - Ey[n]));
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}


static void updateHx_p_vector(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <  jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <  kMax; k++) {
			real_t fi, dfi;
			finc(Xn[i], Yc[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.hi[0], Planewave.ai, Dt, &fi, &dfi);
			Hx[n] = K1Hx[n] * Hx[n]
			      - K2Hx[n] * (RYc[j] * (Ez[n + Nj] - Ez[n])
			                 - RZc[k] * (Ey[n + Nk] - Ey[n]))
			      - (K1Hx[n] - K2Hx[n]) * dfi
			      - (1 - K1Hx[n]) * fi;
			n++;
		}
	}
	}
}


static void updateHx_p_no_vector(double t)
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
		for (int k = kMin; k <  kMax; k++) {
			const int64_t m = iHx[n];
			if (m == 0) {
				Hx[n] -= RYc[j] * (Ez[n + Nj] - Ez[n])
				       - RZc[k] * (Ey[n + Nk] - Ey[n]);
			}
			else {
				real_t fi, dfi;
				finc(Xn[i], Yc[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.hi[0], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Hx[n] = -fi;
				}
				else {
					Hx[n] = D1[m] * Hx[n]
					      - D2[m] * (RYc[j] * (Ez[n + Nj] - Ez[n])
					               - RZc[k] * (Ey[n + Nk] - Ey[n]))
					      - (D1[m] - D2[m]) * dfi
					      - (1 - D1[m]) * fi;
				}
			}
			n++;
		}
	}
	}
#else	// _ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 1, updateBlock[0]),
		CEIL(jMax - jMin + 0, updateBlock[1]),
		CEIL(kMax - kMin + 0, updateBlock[2]));
	sycl::range<3> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto s_t = (real_t) t;
		auto SPlanewave = ::d_SPlanewave;
		real_t s_Dt = (real_t) Dt;
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
		auto Ez = ::Ez;
		auto iHx = ::d_iHx;
		auto s_Xn = ::s_Xn;
		auto s_Yc = ::s_Yc;
		auto s_Zc = ::s_Zc;
		auto D1 = ::d_D1;
		auto D2 = ::d_D2;
		auto RYc = ::d_RYc;
		auto RZc = ::d_RZc;
		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i <= iMax) &&
					(j < jMax) &&
					(k < kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t m = iHx[n];
					const int64_t n1 = n + Nj;
					const int64_t n2 = n + Nk;
					if (m == 0) {
						Hx[n] -= RYc[j] * (Ez[n1] - Ez[n])
							- RZc[k] * (Ey[n2] - Ey[n]);
					}
					else {
						real_t fi, dfi;
						finc_s(s_Xn[i], s_Yc[j],s_Zc[k], s_t, SPlanewave->r0, SPlanewave->ri, SPlanewave->hi[0], SPlanewave->ai, s_Dt, &fi, &dfi);
						if (m == PEC) {
							Hx[n] = -fi;
						}
						else {
							Hx[n] = D1[m] * Hx[n]
								- D2[m] * (RYc[j] * (Ez[n1] - Ez[n])
									- RZc[k] * (Ey[n2] - Ey[n]))
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


void updateHx(double t)
{
	if (NFeed) {
		if (VECTOR) {
			updateHx_f_vector();
		}
		else {
			updateHx_f_no_vector();
		}
	}
	else if (IPlanewave) {
		if (VECTOR) {
			updateHx_p_vector(t);
		}
		else {
			updateHx_p_no_vector(t);
		}
	}
}
