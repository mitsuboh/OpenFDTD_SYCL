/*
updateHy.c

update Hy
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

static void updateHy_f_vector(void)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <  kMax; k++) {
			Hy[n] = K1Hy[n] * Hy[n]
			      - K2Hy[n] * (RZc[k] * (Ex[n + Nk] - Ex[n])
			                 - RXc[i] * (Ez[n + Ni] - Ez[n]));
			n++;
		}
	}
	}
}


static void updateHy_f_no_vector(void)
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
		for (int k = kMin; k <  kMax; k++) {
			const int64_t m = iHy[n];
			Hy[n] = D1[m] * Hy[n]
			      - D2[m] * (RZc[k] * (Ex[n + Nk] - Ex[n])
			               - RXc[i] * (Ez[n + Ni] - Ez[n]));
			n++;
		}
	}
	}

#else	// _ONEAPI

	sycl::range<3> updateBlock = sycl::range<3>(1, 4, 32);
	sycl::range<3> grid(CEIL(iMax - iMin + 0, updateBlock[0]),
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
		auto Hy = ::Hy;
		auto Ex = ::Ex;
		auto Ez = ::Ez;
		auto iHy = ::iHy;
		auto D1 = ::D1;
		auto D2 = ::D2;
		auto RXc = ::RXc;
		auto RZc = ::RZc;
		h.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i < iMax) &&
					(j <= jMax) &&
					(k < kMax)) {
					const int64_t n = NA(i, j, k);
					const int64_t n1 = n + Nk;
					const int64_t n2 = n + Ni;
					Hy[n] = D1[iHy[n]] * Hy[n]
			      			- D2[iHy[n]] * (RZc[k] * (Ex[n1] - Ex[n])
			                    	- RXc[i] * (Ez[n2] - Ez[n]));
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}


static void updateHy_p_vector(double t)
{
	assert(Nk == 1);

	int i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (    i = iMin; i <  iMax; i++) {
	for (int j = jMin; j <= jMax; j++) {
		int64_t n = NA(i, j, kMin);
		for (int k = kMin; k <  kMax; k++) {
			real_t fi, dfi;
			finc(Xc[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.hi[1], Planewave.ai, Dt, &fi, &dfi);
			Hy[n] = K1Hy[n] * Hy[n]
			      - K2Hy[n] * (RZc[k] * (Ex[n + Nk] - Ex[n])
			                 - RXc[i] * (Ez[n + Ni] - Ez[n]))
			      - (K1Hy[n] - K2Hy[n]) * dfi
			      - (1 - K1Hy[n]) * fi;
			n++;
		}
	}
	}
}


static void updateHy_p_no_vector(double t)
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
		for (int k = kMin; k <  kMax; k++) {
			const int64_t m = iHy[n];
			if (m == 0) {
				Hy[n] -= RZc[k] * (Ex[n + Nk] - Ex[n])
				       - RXc[i] * (Ez[n + Ni] - Ez[n]);
			}
			else {
				real_t fi, dfi;
				finc(Xc[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.hi[1], Planewave.ai, Dt, &fi, &dfi);
				if (m == PEC) {
					Hy[n] = -fi;
				}
				else {
					Hy[n] = D1[m] * Hy[n]
					      - D2[m] * (RZc[k] * (Ex[n + Nk] - Ex[n])
					               - RXc[i] * (Ez[n + Ni] - Ez[n]))
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
	sycl::range<3> grid(CEIL(iMax - iMin + 0, updateBlock[0]),
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
		auto Hy = ::Hy;
		auto Ex = ::Ex;
		auto Ez = ::Ez;
		auto iHy = ::iHy;
		auto s_Xc = ::s_Xc;
		auto s_Yn = ::s_Yn;
		auto s_Zc = ::s_Zc;
		auto D1 = ::D1;
		auto D2 = ::D2;
		auto D3 = ::D3;
		auto D4 = ::D4;
		auto RXc = ::RXc;
		auto RZc = ::RZc;
		h.parallel_for(
			sycl::nd_range<3>(all_grid, updateBlock),
			[=](sycl::nd_item<3> idx) {
				auto i = iMin + idx.get_global_id(0);
				auto j = jMin + idx.get_global_id(1);
				auto k = kMin + idx.get_global_id(2);
				if ((i < iMax) &&
					(j <= jMax) &&
					(k < kMax)) {
					int64_t n = NA(i, j, k);
					int64_t n1 = n + Nk;
					int64_t n2 = n + Ni;
					const id_t m = iHy[n];
					if (m == 0) {
						Hy[n] -= RZc[k] * (Ex[n1] - Ex[n])
							- RXc[i] * (Ez[n2] - Ez[n]);
					}
					else {
						real_t fi, dfi;
						finc_s(s_Xc[i], s_Yn[j], s_Zc[k], s_t, SPlanewave->r0, SPlanewave->ri, SPlanewave->hi[1], SPlanewave->ai, s_Dt, &fi, &dfi);

						if (m == PEC) {
							Hy[n] = -fi;
						}
						else {
							Hy[n] = D1[m] * Hy[n]
								- D2[m] * (RZc[k] * (Ex[n1] - Ex[n])
									- RXc[i] * (Ez[n2] - Ez[n]))
								- D3[m] * dfi
								- D4[m] * fi;
						}
					}
				}
			});
		});

	myQ.wait();
#endif // _ONEAPI
}


void updateHy(double t)
{
	if (NFeed) {
		if (VECTOR) {
			updateHy_f_vector();
		}
		else {
			updateHy_f_no_vector();
		}
	}
	else if (IPlanewave) {
		if (VECTOR) {
			updateHy_p_vector(t);
		}
		else {
			updateHy_p_no_vector(t);
		}
	}
}
