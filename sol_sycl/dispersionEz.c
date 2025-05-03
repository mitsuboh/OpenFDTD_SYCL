/*
dispersionEz.c

update Ez (dispersion)
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void dispersionEz(double t)
{
#ifndef _ONEAPI
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (n = 0; n < numDispersionEz; n++) {
		const int     i = mDispersionEz[n].i;
		const int     j = mDispersionEz[n].j;
		const int     k = mDispersionEz[n].k;
		const real_t f1 = mDispersionEz[n].f1;
		const real_t f2 = mDispersionEz[n].f2;
		const real_t f3 = mDispersionEz[n].f3;

		real_t fi = 0;
		if (IPlanewave) {
			real_t dfi;
			finc(Xn[i], Yn[j], Zc[k], t, Planewave.r0, Planewave.ri, Planewave.ei[2], Planewave.ai, Dt, &fi, &dfi);
		}

		EZ(i, j, k) += f1 * DispersionEz[n];

		DispersionEz[n] = f2 * (EZ(i, j, k) + fi)
		                + f3 * DispersionEz[n];
	}
#else //_ONEAPI
	const int dispersionBlock = 256;
	sycl::range<1> updateBlock = sycl::range<1>(dispersionBlock);
	sycl::range<1> grid(CEIL(numDispersionEz,dispersionBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto numDispersionEz = ::numDispersionEz;
		auto mDispersionEz = ::mDispersionEz;
		auto DispersionEz = ::DispersionEz;
		auto IPlanewave = ::IPlanewave;
		auto s_Xn = ::s_Xn;
		auto s_Yn = ::s_Yn;
		auto s_Zc = ::s_Zc;
		auto s_t = (real_t)t;
		auto SPlanewave = ::SPlanewave;
		auto s_Dt = (real_t)Dt; 
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Ez = ::Ez;

		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < numDispersionEz){
					const int     i = mDispersionEz[n].i;
					const int     j = mDispersionEz[n].j;
					const int     k = mDispersionEz[n].k;
					const real_t f1 = mDispersionEz[n].f1;
					const real_t f2 = mDispersionEz[n].f2;
					const real_t f3 = mDispersionEz[n].f3;

					real_t fi = 0;
					if (IPlanewave) {
						real_t dfi;
						finc_s(s_Xn[i], s_Yn[j], s_Zc[k], s_t, SPlanewave->r0, SPlanewave->ri, SPlanewave->ei[2], SPlanewave->ai, s_Dt, &fi, &dfi);
					}

					EZ(i, j, k) += f1 * DispersionEz[n];

					DispersionEz[n] = f2 * (EZ(i, j, k) + fi)
		       			         + f3 * DispersionEz[n];
				}
		});
	});
	myQ.wait();
#endif // _ONEAPI
}
