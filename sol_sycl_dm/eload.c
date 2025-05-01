/*
eload.c

E on loads (inductors)
*/

#include "ofd.h"
#include "ofd_prototype.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void eload(void)
{
	if (NInductor <= 0) return;

	const double cdt = (2.99792458e8) * Dt;

#ifndef _ONEAPI
	for (int n = 0; n < NInductor; n++) {
		inductor_t *ptr = &Inductor[n];

		int     i = ptr->i;
		int     j = ptr->j;
		int     k = ptr->k;
		double dx = ptr->dx;
		double dy = ptr->dy;
		double dz = ptr->dz;

		if      ((ptr->dir == 'X') &&
		         (iMin <= i) && (i <  iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			const double roth = (HZ(i, j, k) - HZ(i,     j - 1, k    )) / dy
			                  - (HY(i, j, k) - HY(i,     j,     k - 1)) / dz;
			EX(i, j, k) = (real_t)(ptr->e + (cdt * roth) - (ptr->fctr * cdt * cdt * ptr->esum));
			ptr->e = EX(i, j, k);
			ptr->esum += ptr->e;
		}
		else if ((ptr->dir == 'Y') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <  jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			const double roth = (HX(i, j, k) - HX(i,     j,     k - 1)) / dz
			                  - (HZ(i, j, k) - HZ(i - 1, j,     k    )) / dx;
			EY(i, j, k) = (real_t)(ptr->e + (cdt * roth) - (ptr->fctr * cdt * cdt * ptr->esum));
			ptr->e = EY(i, j, k);
			ptr->esum += ptr->e;
		}
		else if ((ptr->dir == 'Z') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <  kMax)) {  // MPI
			const double roth = (HY(i, j, k) - HY(i - 1, j,     k    )) / dx
			                  - (HX(i, j, k) - HX(i,     j - 1, k    )) / dy;
			EZ(i, j, k) = (real_t)(ptr->e + (cdt * roth) - (ptr->fctr * cdt * cdt * ptr->esum));
			ptr->e = EZ(i, j, k);
			ptr->esum += ptr->e;
		}
	}
#else	// _ONEAPI

	const int block = 256;
	sycl::range<1> updateBlock = sycl::range<1>(block);
	sycl::range<1> grid(CEIL(NInductor, block));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto NInductor = ::NInductor;
		auto Inductor = ::d_Inductor;
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
		auto Ey = ::Ey;
		auto Ez = ::Ez;
		auto Hx = ::Hx;
		auto Hy = ::Hy;
		auto Hz = ::Hz;
		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
			const int64_t n = idx.get_global_id(0);
			if( n < NInductor) {
				inductor_t *ptr = &Inductor[n];
				int     i = ptr->i;
				int     j = ptr->j;
				int     k = ptr->k;
				double dx = ptr->dx;
				double dy = ptr->dy;
				double dz = ptr->dz;

		if      ((ptr->dir == 'X') &&
		         (iMin <= i) && (i <  iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			const double roth = (HZ(i, j, k) - HZ(i,     j - 1, k    )) / dy
			                  - (HY(i, j, k) - HY(i,     j,     k - 1)) / dz;
			EX(i, j, k) = (real_t)(ptr->e + (cdt * roth) - (ptr->fctr * cdt * cdt * ptr->esum));
			ptr->e = EX(i, j, k);
			ptr->esum += ptr->e;
		}
		else if ((ptr->dir == 'Y') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <  jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			const double roth = (HX(i, j, k) - HX(i,     j,     k - 1)) / dz
			                  - (HZ(i, j, k) - HZ(i - 1, j,     k    )) / dx;
			EY(i, j, k) = (real_t)(ptr->e + (cdt * roth) - (ptr->fctr * cdt * cdt * ptr->esum));
			ptr->e = EY(i, j, k);
			ptr->esum += ptr->e;
		}
		else if ((ptr->dir == 'Z') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <  kMax)) {  // MPI
			const double roth = (HY(i, j, k) - HY(i - 1, j,     k    )) / dx
			                  - (HX(i, j, k) - HX(i,     j - 1, k    )) / dy;
			EZ(i, j, k) = (real_t)(ptr->e + (cdt * roth) - (ptr->fctr * cdt * cdt * ptr->esum));
			ptr->e = EZ(i, j, k);
			ptr->esum += ptr->e;
		}
			}
			});
		});
	myQ.wait();
#endif	// _ONEAPI
}
