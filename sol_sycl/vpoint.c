/*
vpoint.c

V waveform on points
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void vpoint(int itime)
{
	if (NPoint <= 0) return;
#ifndef _ONEAPI
	real_t fi, dfi;

	for (int n = 0; n < NPoint + 2; n++) {
		const int i = Point[n].i;
		const int j = Point[n].j;
		const int k = Point[n].k;

		double e = 0;
		double d = 0;
		if      ((Point[n].dir == 'X') &&
		         (iMin <= i) && (i <  iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			e = EX(i, j, k);
			d = Point[n].dx;
			if (IPlanewave) {
				const double t = (itime + 1) * Dt;
				finc(Xc[i], Yn[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[0], Planewave.ai, Dt, &fi, &dfi);
				e += fi;
			}
		}
		else if ((Point[n].dir == 'Y') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <  jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			e = EY(i, j, k);
			d = Point[n].dy;
			if (IPlanewave) {
				const double t = (itime + 1) * Dt;
				finc(Xn[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[1], Planewave.ai, Dt, &fi, &dfi);
				e += fi;
			}
		}
		else if ((Point[n].dir == 'Z') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <  kMax)) {  // MPI
			e = EZ(i, j, k);
			d = Point[n].dz;
			if (IPlanewave) {
				const double t = (itime + 1) * Dt;
				finc(Xn[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.ei[2], Planewave.ai, Dt, &fi, &dfi);
				e += fi;
			}
		}
		const int id = n * (Solver.maxiter + 1) + itime;
		VPoint[id] = e * (-d);
	}
#else //_ONEAPI
	const int PointBlock = 256;
	sycl::range<1> block = sycl::range<1>(MIN(PointBlock, NPoint + 2));
	sycl::range<1> grid(CEIL(NPoint + 2, block));
	sycl::range<1> all_grid = grid * block;

	myQ.submit([&](sycl::handler& hndl) {
		auto NPoint = ::NPoint;
		auto Point = ::Point;
		auto VPoint = ::VPoint;
		auto IPlanewave = ::IPlanewave;
		auto SPlanewave = ::d_SPlanewave;
		auto iMin = ::iMin;
		auto iMax = ::iMax;
		auto jMin = ::jMin;
		auto jMax = ::jMax;
		auto kMin = ::kMin;
		auto kMax = ::kMax;
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Ex = ::Ex;
		auto Ey = ::Ey;
		auto Ez = ::Ez;
		auto s_Xn = ::s_Xn;
		auto s_Yn = ::s_Yn;
		auto s_Zn = ::s_Zn;
		auto s_Xc = ::s_Xc;
		auto s_Yc = ::s_Yc;
		auto s_Zc = ::s_Zc;
		auto Dt = ::Dt;
		real_t s_t = (real_t) ::Dt;
		auto Solver = ::Solver;
		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, block),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < NPoint + 2){
					const int i = Point[n].i;
					const int j = Point[n].j;
					const int k = Point[n].k;

		double e = 0;
		double d = 0;
		real_t fi = 0;
		real_t dfi = 0;
		if      ((Point[n].dir == 'X') &&
		         (iMin <= i) && (i <  iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			e = EX(i, j, k);
			d = Point[n].dx;
			if (IPlanewave) {
				const real_t t = (itime + 1) * Dt;
				finc_s(s_Xc[i], s_Yn[j], s_Zn[k], t, SPlanewave->r0, SPlanewave->ri, SPlanewave->ei[0], SPlanewave->ai, s_t, &fi, &dfi);
				e += fi;
			}
		}
		else if ((Point[n].dir == 'Y') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <  jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			e = EY(i, j, k);
			d = Point[n].dy;
			if (IPlanewave) {
				const double t = (itime + 1) * Dt;
				finc_s(s_Xn[i], s_Yc[j], s_Zn[k], t, SPlanewave->r0, SPlanewave->ri, SPlanewave->ei[1], SPlanewave->ai, s_t, &fi, &dfi);
				e += fi;
			}
		}
		else if ((Point[n].dir == 'Z') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <  kMax)) {  // MPI
			e = EZ(i, j, k);
			d = Point[n].dz;
			if (IPlanewave) {
				const double t = (itime + 1) * Dt;
				finc_s(s_Xn[i], s_Yn[j], s_Zc[k], t, SPlanewave->r0, SPlanewave->ri, SPlanewave->ei[2], SPlanewave->ai, s_t, &fi, &dfi);
				e += fi;
			}
		}

		const int id = n * (Solver.maxiter + 1) + itime;
		VPoint[id] = e * (-d);
				}
		});
	});
#endif // _ONEAPI
}
