/*
average.c (OpenMP)

E/H average
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void average(double fsum[])
{
	double se = 0;
	double sh = 0;
#ifndef _ONEAPI
	int    i;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : se, sh)
#endif
	for (    i = iMin; i < iMax; i++) {
	for (int j = jMin; j < jMax; j++) {
	for (int k = kMin; k < kMax; k++) {
		se +=
			+ fabs(
				+ EX(i    , j    , k    )
				+ EX(i    , j + 1, k    )
				+ EX(i    , j    , k + 1)
				+ EX(i    , j + 1, k + 1))
			+ fabs(
				+ EY(i    , j    , k    )
				+ EY(i    , j    , k + 1)
				+ EY(i + 1, j    , k    )
				+ EY(i + 1, j    , k + 1))
			+ fabs(
				+ EZ(i    , j    , k    )
				+ EZ(i + 1, j    , k    )
				+ EZ(i    , j + 1, k    )
				+ EZ(i + 1, j + 1, k    ));
		sh +=
			+ fabs(
				+ HX(i    , j    , k    )
				+ HX(i + 1, j    , k    ))
			+ fabs(
				+ HY(i    , j    , k    )
				+ HY(i    , j + 1, k    ))
			+ fabs(
				+ HZ(i    , j    , k    )
				+ HZ(i    , j    , k + 1));
	}
	}
	}

#else	// _ONEAPI

	sycl::range<3> sumBlock = sycl::range<3>(1, 8, 32);
	sycl::range<3> sumGrid = sycl::range<3>(CEIL(Nx, sumBlock[0]),
		CEIL(Ny, sumBlock[1]), CEIL(Nz, sumBlock[2]));
	sycl::range<3> all_grid = sumGrid * sumBlock;
	const int wgroup_size = sumBlock[2] * sumBlock[1] * sumBlock[0];
	const int wgroup_num = sumGrid[2] * sumGrid[1] * sumGrid[0];
	float* sumH = static_cast<float*> malloc_dev(wgroup_num * sizeof(float));
	float* sumE = static_cast<float*> malloc_dev(wgroup_num * sizeof(float));

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
		auto Hy = ::Hy;
		auto Hz = ::Hz;
		auto Ex = ::Ex;
		auto Ey = ::Ey;
		auto Ez = ::Ez;
		sycl::local_accessor<real_t> se(wgroup_size, hndl);
		sycl::local_accessor<real_t> sh(wgroup_size, hndl);
		hndl.parallel_for(
			sycl::nd_range<3>(all_grid, sumBlock),
			[=](sycl::nd_item<3> idx) {
				const int i = iMin + idx.get_global_id(0);
				const int j = jMin + idx.get_global_id(1);
				const int k = kMin + idx.get_global_id(2);

				const int tid = idx.get_local_linear_id();
				const int bid = idx.get_group_linear_id();

				if ((i < iMax) && (j < jMax) && (k < kMax)) {
					se[tid] = fabs(EX(i, j, k) + EX(i, j + 1, k) + EX(i, j, k + 1) + EX(i, j + 1, k + 1)) +
						fabs(EY(i, j, k) + EY(i + 1, j, k) + EY(i, j, k + 1) + EY(i + 1, j, k + 1)) +
						fabs(EZ(i, j, k) + EZ(i, j + 1, k) + EZ(i + 1, j, k) + EZ(i + 1, j + 1, k));
					sh[tid] = fabs(HX(i, j, k) + HX(i + 1, j, k)) +
						fabs(HY(i, j, k) + HY(i, j + 1, k)) +
						fabs(HZ(i, j, k) + HZ(i, j, k + 1));
				}
				else {
					se[tid] = 0;
					sh[tid] = 0;
				}

				sumE[bid] = joint_reduce(idx.get_group(), &se[0], &se[wgroup_size], sycl::plus<>());
				sumH[bid] = joint_reduce(idx.get_group(), &sh[0], &sh[wgroup_size], sycl::plus<>());
			});
		});
	myQ.wait();
	// partial sum

	float sef = 0;
	sycl::buffer<float> sefBuf{ &sef, 1 };
	float shf = 0;
	sycl::buffer<float> shfBuf{ &shf, 1 };

	myQ.submit([&](sycl::handler& hndl) {
		hndl.parallel_for(sycl::range<1>(wgroup_num),sycl::reduction(sefBuf, hndl, sycl::plus<>()),
		[=](sycl::id<1> idx, auto& sum) {
				sum += sumE[idx];
		});
	});
	myQ.submit([&](sycl::handler& hndl) {
		hndl.parallel_for(sycl::range<1>(wgroup_num), sycl::reduction(shfBuf, hndl, sycl::plus<>()),
		[=](sycl::id<1> idx,auto& sum) {
				sum += sumH[idx];
		});
	});
	se = sefBuf.get_host_access()[0];
	sh = shfBuf.get_host_access()[0];

#endif	// _ONEAPI

	fsum[0] = se / (4.0 * Nx * Ny * Nz);
	fsum[1] = sh / (2.0 * Nx * Ny * Nz);
}
