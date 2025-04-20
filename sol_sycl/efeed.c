/*
efeed.c

E on feeds
*/

#include "ofd.h"
#include "ofd_prototype.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
//extern Pdouble rcp(Pdouble);
#undef ETA0
#define ETA0 ((2.99792458e8) * MU0)		//Maybe better to change ofd.h
#endif

void efeed(int itime)
{
	if (NFeed <= 0) return;

	const double eps = 1e-6;

	const double t = (itime + 1) * Dt;

#ifndef _ONEAPI

	for (int ifeed = 0; ifeed < NFeed; ifeed++) {
		const int     i = Feed[ifeed].i;
		const int     j = Feed[ifeed].j;
		const int     k = Feed[ifeed].k;
		const double dx = Feed[ifeed].dx;
		const double dy = Feed[ifeed].dy;
		const double dz = Feed[ifeed].dz;

		// V
		const double v0 = vfeed(t, Tw, Feed[ifeed].delay, WFeed);
		double v = v0 * Feed[ifeed].volt;

		// E, V, I
		double c = 0;
		if      ((Feed[ifeed].dir == 'X') &&
		         (iMin <= i) && (i <  iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			c = dz * (HZ(i, j, k) - HZ(i,     j - 1, k    ))
			  - dy * (HY(i, j, k) - HY(i,     j,     k - 1));
			c /= ETA0;
			v -= rFeed * c;
			if ((IEX(i, j, k) == PEC) || (fabs(v0) > eps)) {
				EX(i, j, k) = -(real_t)(v / dx);
			}
		}
		else if ((Feed[ifeed].dir == 'Y') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <  jMax) &&
		         (kMin <= k) && (k <= kMax)) {  // MPI
			c = dx * (HX(i, j, k) - HX(i,     j,     k - 1))
			  - dz * (HZ(i, j, k) - HZ(i - 1, j,     k    ));
			c /= ETA0;
			v -= rFeed * c;
			if ((IEY(i, j, k) == PEC) || (fabs(v0) > eps)) {
				EY(i, j, k) = -(real_t)(v / dy);
			}
		}
		else if ((Feed[ifeed].dir == 'Z') &&
		         (iMin <= i) && (i <= iMax) &&
		         (jMin <= j) && (j <= jMax) &&
		         (kMin <= k) && (k <  kMax)) {  // MPI
			c = dy * (HY(i, j, k) - HY(i - 1, j,     k    ))
			  - dx * (HX(i, j, k) - HX(i,     j - 1, k    ));
			c /= ETA0;
			v -= rFeed * c;
			if ((IEZ(i, j, k) == PEC) || (fabs(v0) > eps)) {
				EZ(i, j, k) = -(real_t)(v / dz);
			}
		}

		// V/I waveform
		const int id = ifeed * (Solver.maxiter + 1) + itime;
		VFeed[id] = v;
		IFeed[id] = c;
	}

#else	// _ONEAPI
SYCL_EXTERNAL double vfeed(double,double,double,int);

	const int Block = 128;
	sycl::range<1> updateBlock = sycl::range<1>(Block);
	sycl::range<1> grid(CEIL(NFeed, Block));
	sycl::range<1> all_grid = grid * updateBlock;

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
		auto Ex = ::Ex;
		auto Ey = ::Ey;
		auto Ez = ::Ez;
		auto Hx = ::Hx;
		auto Hy = ::Hy;
		auto Hz = ::Hz;
		auto iEx = ::iEx;
		auto iEy = ::iEy;
		auto iEz = ::iEz;
		auto NFeed = ::NFeed;
		auto Feed = ::Feed;
		auto VFeed = ::VFeed;
		auto IFeed = ::IFeed;
		auto WFeed = ::WFeed;
		int maxiter_l = Solver.maxiter;
		auto rFeed = ::rFeed;
		auto Tw = ::Tw;
		hndl.parallel_for(
			sycl::nd_range<1>(all_grid, updateBlock),
			[=](sycl::nd_item<1> idx) {
				const int64_t ifeed = idx.get_global_id(0);
				if (ifeed < NFeed) {
					const int     i = Feed[ifeed].i;
					const int     j = Feed[ifeed].j;
					const int     k = Feed[ifeed].k;
					double dx = Feed[ifeed].dx;
					double dy = Feed[ifeed].dy;
					double dz = Feed[ifeed].dz;

					// V
					double v0 = vfeed(t, Tw, Feed[ifeed].delay, WFeed);
					double v = v0 * Feed[ifeed].volt;

					// E, V, I
					double c = 0;
					if ((Feed[ifeed].dir == 'X') &&
						(iMin <= i) && (i < iMax) &&
						(jMin <= j) && (j <= jMax) &&
						(kMin <= k) && (k <= kMax)) {  // MPI
						c = dz * (HZ(i, j, k) - HZ(i, j - 1, k))
							- dy * (HY(i, j, k) - HY(i, j, k - 1));
						c /= ETA0;
						v -= rFeed * c;
						if ((IEX(i, j, k) == PEC) || (fabs(v0) > eps)) {
							EX(i, j, k) = -(real_t)(v / dx);
						}
					}
					else if ((Feed[ifeed].dir == 'Y') &&
						(iMin <= i) && (i <= iMax) &&
						(jMin <= j) && (j < jMax) &&
						(kMin <= k) && (k <= kMax)) {  // MPI
						c = dx * (HX(i, j, k) - HX(i, j, k - 1))
							- dz * (HZ(i, j, k) - HZ(i - 1, j, k));
						c /= ETA0;
						v -= rFeed * c;
						if ((IEY(i, j, k) == PEC) || (fabs(v0) > eps)) {
							EY(i, j, k) = -(real_t)(v / dy);
						}
					}
					else if ((Feed[ifeed].dir == 'Z') &&
						(iMin <= i) && (i <= iMax) &&
						(jMin <= j) && (j <= jMax) &&
						(kMin <= k) && (k < kMax)) {  // MPI
						c = dy * (HY(i, j, k) - HY(i - 1, j, k))
							- dx * (HX(i, j, k) - HX(i, j - 1, k));
						c /= ETA0;
						v -= rFeed * c;
						if ((IEZ(i, j, k) == PEC) || (fabs(v0) > eps)) {
							EZ(i, j, k) = -(real_t)(v / dz);
						}
					}

					// V/I waveform
					const int id = ifeed * (maxiter_l + 1) + itime;
					VFeed[id] = v;
					IFeed[id] = c;
				}
			});
		});
	myQ.wait();

#endif	// _ONEAPI
}
