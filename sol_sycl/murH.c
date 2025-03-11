/*
murH.c (OpenMP)

Mur for Hx/Hy/Hz
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void murH(int64_t num, mur_t *fmur, real_t *hh)
{
#ifndef _ONEAPI
	int64_t n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
#ifdef __NEC__
#pragma _NEC ivdep
#endif
#ifdef __CLANG_FUJITSU
#pragma clang loop vectorize(assume_safety)
#endif
	for (n = 0; n < num; n++) {
		const int i  = fmur[n].i;
		const int j  = fmur[n].j;
		const int k  = fmur[n].k;
		const int i1 = fmur[n].i1;
		const int j1 = fmur[n].j1;
		const int k1 = fmur[n].k1;
		hh[NA(i, j, k)] = fmur[n].f
		               + fmur[n].g * (hh[NA(i1, j1, k1)] - hh[NA(i, j, k)]);
		fmur[n].f = hh[NA(i1, j1, k1)];
	}
#else // _ONEAPI
	const int murBlock = 256;
	sycl::range<1> updateBlock = sycl::range<1>(murBlock);
	sycl::range<1> grid(CEIL(num, murBlock));
	sycl::range<1> all_grid = grid * updateBlock;

	myQ.submit([&](sycl::handler& h) {
		auto Ni = ::Ni;
		auto Nj = ::Nj;
		auto Nk = ::Nk;
		auto N0 = ::N0;

		h.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t n = idx.get_global_id(0);
				if (n < num) {
					const int i = fmur[n].i;
					const int j = fmur[n].j;
					const int k = fmur[n].k;
					const int i1 = fmur[n].i1;
					const int j1 = fmur[n].j1;
					const int k1 = fmur[n].k1;
					hh[NA(i, j, k)] = fmur[n].f
						+ fmur[n].g * (hh[NA(i1, j1, k1)] - hh[NA(i, j, k)]);
					fmur[n].f = hh[NA(i1, j1, k1)];
				}
			});
		});
	myQ.wait();
#endif // _ONEAPI
}
