/*
dispersionEy.c

update Ey (dispersion)
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void dispersionEy(double t)
{
#ifndef _ONEAPI
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (n = 0; n < numDispersionEy; n++) {
		const int     i = mDispersionEy[n].i;
		const int     j = mDispersionEy[n].j;
		const int     k = mDispersionEy[n].k;
		const real_t f1 = mDispersionEy[n].f1;
		const real_t f2 = mDispersionEy[n].f2;
		const real_t f3 = mDispersionEy[n].f3;

		real_t fi = 0;
		if (IPlanewave) {
			real_t dfi;
			finc(Xn[i], Yc[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[1], Planewave.ai, Dt, &fi, &dfi);
		}

		EY(i, j, k) += f1 * DispersionEy[n];

		DispersionEy[n] = f2 * (EY(i, j, k) + fi)
		                + f3 * DispersionEy[n];
	}
#else //_ONEAPI
	const int dispersionBlock = 256;
	sycl::range<1> updateBlock = sycl::range<1>(dispersionBlock);
	sycl::range<1> grid(CEIL(numDispersionEy,dispersionBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto numDispersionEy = ::numDispersionEy;
		auto mDispersionEy = ::mDispersionEy;
		auto DispersionEy = ::DispersionEy;
		auto IPlanewave = ::IPlanewave;
		auto s_Xn = ::s_Xn;
		auto s_Yc = ::s_Yc;
		auto s_Zn = ::s_Zn;
		auto s_t = (real_t)t;
		auto SPlanewave = ::SPlanewave;
		auto s_Dt = (real_t)Dt; 
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Ey = ::Ey;

		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < numDispersionEy){
					const int     i = mDispersionEy[n].i;
					const int     j = mDispersionEy[n].j;
					const int     k = mDispersionEy[n].k;
					const real_t f1 = mDispersionEy[n].f1;
					const real_t f2 = mDispersionEy[n].f2;
					const real_t f3 = mDispersionEy[n].f3;

					real_t fi = 0;
					if (IPlanewave) {
						real_t dfi;
						finc_s(s_Xn[i], s_Yc[j], s_Zn[k], s_t, SPlanewave->r0, SPlanewave->ri, SPlanewave->ei[1], SPlanewave->ai, s_Dt, &fi, &dfi);
					}

					EY(i, j, k) += f1 * DispersionEy[n];

					DispersionEy[n] = f2 * (EY(i, j, k) + fi)
		                		+ f3 * DispersionEy[n];
				}
		});
	});
	myQ.wait();
#endif // _ONEAPI
}
