/*
dispersionEx.c

update Ex (dispersion)
*/

#include "ofd.h"
#include "finc.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void dispersionEx(double t)
{
#ifndef _ONEAPI
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (n = 0; n < numDispersionEx; n++) {
		const int     i = mDispersionEx[n].i;
		const int     j = mDispersionEx[n].j;
		const int     k = mDispersionEx[n].k;
		const real_t f1 = mDispersionEx[n].f1;
		const real_t f2 = mDispersionEx[n].f2;
		const real_t f3 = mDispersionEx[n].f3;

		real_t fi = 0;
		if (IPlanewave) {
			real_t dfi;
			finc(Xc[i], Yn[j], Zn[k], t, Planewave.r0, Planewave.ri, Planewave.ei[0], Planewave.ai, Dt, &fi, &dfi);
		}

		EX(i, j, k) += f1 * DispersionEx[n];

		DispersionEx[n] = f2 * (EX(i, j, k) + fi)
		                + f3 * DispersionEx[n];
	}
#else //_ONEAPI
	const int dispersionBlock = 256;
	sycl::range<1> updateBlock = sycl::range<1>(dispersionBlock);
	sycl::range<1> grid(CEIL(numDispersionEx,dispersionBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& hndl) {
		auto numDispersionEx = ::numDispersionEx;
		auto mDispersionEx = ::mDispersionEx;
		auto DispersionEx = ::DispersionEx;
		auto IPlanewave = ::IPlanewave;
		auto s_Xc = ::s_Xc;
		auto s_Yn = ::s_Yn;
		auto s_Zn = ::s_Zn;
		auto s_t = (real_t)t;
		auto SPlanewave = ::SPlanewave;
		auto s_Dt = (real_t)Dt; 
		auto N0 = ::N0;
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto Ex = ::Ex;

		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < numDispersionEx){
					const int     i = mDispersionEx[n].i;
					const int     j = mDispersionEx[n].j;
					const int     k = mDispersionEx[n].k;
					const real_t f1 = mDispersionEx[n].f1;
					const real_t f2 = mDispersionEx[n].f2;
					const real_t f3 = mDispersionEx[n].f3;

					real_t fi = 0;
					if (IPlanewave) {
						real_t dfi;
						finc_s(s_Xc[i], s_Yn[j], s_Zn[k], s_t, SPlanewave.r0, SPlanewave.ri, SPlanewave.ei[0], SPlanewave.ai, s_Dt, &fi, &dfi);
					}

					EX(i, j, k) += f1 * DispersionEx[n];

					DispersionEx[n] = f2 * (EX(i, j, k) + fi)
		       			         + f3 * DispersionEx[n];
				}
		});
	});
	myQ.wait();
#endif // _ONEAPI
}
