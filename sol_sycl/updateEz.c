/*
updateEz.c

update Ez
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

static void updateEz_f_vector(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <  kMax; k++) {
			Ez[n] = K1Ez[n] * Ez[n]
			      + K2Ez[n] * (RXn[i] * (Hy[n] - Hy[n - Ni])
			                 - RYn[j] * (Hx[n] - Hx[n - Nj]));
			n++;
		}
	}
	}
}


static void updateEz_f_no_vector(void)
{
	assert(Nk == 1);

#ifndef _ONEAPI
	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <  kMax; k++) {
			const int64_t m = iEz[n];
			Ez[n] = C1[m] * Ez[n]
			      + C2[m] * (RXn[i] * (Hy[n] - Hy[n - Ni])
			               - RYn[j] * (Hx[n] - Hx[n - Nj]));
			n++;
		}
	}
	}
#else // _ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 1, updateBlock[0]),
		CEIL(jMax - jMin + 1, updateBlock[1]),
		CEIL(kMax - kMin + 0, updateBlock[2]));
	sycl::range<3> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& h) {
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
		auto Hy = ::Hy;
		auto Ez = ::Ez;
		auto iEz = ::iEz;
		auto C1 = ::C1;
		auto C2 = ::C2;
		auto RXn = ::RXn;
		auto RYn = ::RYn;
		h.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i <= iMax) &&
					(j <= jMax) &&
					(k < kMax)) {
					const int64_t n = NA(i, j, k);
					int64_t n1 = n - Ni;
					int64_t n2 = n - Nj;
					Ez[n] = C1[iEz[n]] * Ez[n]
						+ C2[iEz[n]] * (RXn[i] * (Hy[n] - Hy[n1])
							- RYn[j] * (Hx[n] - Hx[n2]));
				}
			});
		});
	myQ.wait();
#endif //_ONEAPI
}


static void updateEz_p_vector(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <  kMax; k++) {
			real_t fi, dfi;
			finc(Xn[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.ei[2], Planewave.ai, Dt, &fi, &dfi);
			Ez[n] = K1Ez[n] * Ez[n]
			      + K2Ez[n] * (RXn[i] * (Hy[n] - Hy[n - Ni])
			                 - RYn[j] * (Hx[n] - Hx[n - Nj]))
			      - (K1Ez[n] - K2Ez[n]) * dfi
			      - (1 - K1Ez[n]) * fi;
			n++;
		}
	}
	}
}


static void updateEz_p_no_vector(double t)
{
	assert(Nk == 1);

#ifndef _ONEAPI
	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <= iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <  kMax; k++) {
			const int m = iEz[n];
			if (m == 0) {
				Ez[n] += RXn[i] * (Hy[n] - Hy[n - Ni])
				       - RYn[j] * (Hx[n] - Hx[n - Nj]);
			}
			else {
				real_t fi, dfi;
				finc(Xn[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.ei[2], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Ez[n] = -fi;
				}
				else {
					Ez[n] = C1[m] * Ez[n]
					      + C2[m] * (RXn[i] * (Hy[n] - Hy[n - Ni])
					               - RYn[j] * (Hx[n] - Hx[n - Nj]))
					      - (C1[m] - C2[m]) * dfi
					      - (1 - C1[m]) * fi;
				}
			}
			n++;
		}
	}
	}
#else // _ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 1, updateBlock[0]),
		CEIL(jMax - jMin + 1, updateBlock[1]),
		CEIL(kMax - kMin + 0, updateBlock[2]));
	sycl::range<3> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& h) {
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
		auto Hy = ::Hy;
		auto Ez = ::Ez;
		auto iEz = ::iEz;
		auto s_Xn = ::s_Xn;
		auto s_Yn = ::s_Yn;
		auto s_Zc = ::s_Zc;
		auto C1 = ::C1;
		auto C2 = ::C2;
		auto C3 = ::C3;
		auto C4 = ::C4;
		auto RXn = ::RXn;
		auto RYn = ::RYn;
		h.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i <= iMax) &&
					(j <= jMax) &&
					(k < kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t n1 = n - Ni;
					const int64_t n2 = n - Nj;
					const id_t m = iEz[n];
					if (m == 0) {
						Ez[n] += RXn[i] * (Hy[n] - Hy[n1])
							- RYn[j] * (Hx[n] - Hx[n2]);
					}
					else {
						real_t fi, dfi;
						finc_s(s_Xn[i], s_Yn[j], s_Zc[k], s_t, SPlanewave->r0, SPlanewave->ri, SPlanewave->ei[2], SPlanewave->ai, s_Dt, &fi, &dfi);
						if (m == PEC) {
							Ez[n] = -fi;
						}
						else {
							Ez[n] = C1[m] * Ez[n]
								+ C2[m] * (RXn[i] * (Hy[n] - Hy[n1])
									- RYn[j] * (Hx[n] - Hx[n2]))
								- C3[m] * dfi
								- C4[m] * fi;
						}
					}
				}
			});
		});
	myQ.wait();
#endif //_ONEAPI
}


void updateEz(double t)
{
	if (NFeed) {
		if (VECTOR) {
			updateEz_f_vector();
		}
		else {
			updateEz_f_no_vector();
		}
	}
	else if (IPlanewave) {
		if (VECTOR) {
			updateEz_p_vector(t);
		}
		else {
			updateEz_p_no_vector(t);
		}
	}
}
