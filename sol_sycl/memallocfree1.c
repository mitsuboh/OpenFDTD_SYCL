/*
memallocfree1.c

alloc and free
(1) input parameters
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void memalloc1(void)
{
	size_t size, xsize, ysize, zsize;

#ifdef _ONEAPI
	SPlanewave = (splanewave_t*)malloc_shm(sizeof(splanewave_t));
#endif

	// mesh factor
	xsize = (Nx + 1) * sizeof(real_t);
	ysize = (Ny + 1) * sizeof(real_t);
	zsize = (Nz + 1) * sizeof(real_t);
#ifdef _ONEAPI
	RXn = (real_t *)malloc_shm(xsize);
	RYn = (real_t *)malloc_shm(ysize);
	RZn = (real_t *)malloc_shm(zsize);
#else
	RXn = (real_t *)malloc(xsize);
	RYn = (real_t *)malloc(ysize);
	RZn = (real_t *)malloc(zsize);
#endif

	xsize = (Nx + 0) * sizeof(real_t);
	ysize = (Ny + 0) * sizeof(real_t);
	zsize = (Nz + 0) * sizeof(real_t);
#ifdef _ONEAPI
	RXc = (real_t *)malloc_shm(xsize);
	RYc = (real_t *)malloc_shm(ysize);
	RZc = (real_t *)malloc_shm(zsize);
#else
	RXc = (real_t *)malloc(xsize);
	RYc = (real_t *)malloc(ysize);
	RZc = (real_t *)malloc(zsize);
#endif

	// material ID
	size = NN * sizeof(id_t);
#ifdef _ONEAPI
	iEx = (id_t *)malloc_shm(size);
	iEy = (id_t *)malloc_shm(size);
	iEz = (id_t *)malloc_shm(size);
	iHx = (id_t *)malloc_shm(size);
	iHy = (id_t *)malloc_shm(size);
	iHz = (id_t *)malloc_shm(size);
#else
	iEx = (id_t *)malloc(size);
	iEy = (id_t *)malloc(size);
	iEz = (id_t *)malloc(size);
	iHx = (id_t *)malloc(size);
	iHy = (id_t *)malloc(size);
	iHz = (id_t *)malloc(size);
#endif
	memset(iEx, 0, size);
	memset(iEy, 0, size);
	memset(iEz, 0, size);
	memset(iHx, 0, size);
	memset(iHy, 0, size);
	memset(iHz, 0, size);

	// material factor
	size = NMaterial * sizeof(real_t);
#ifdef _ONEAPI
	C1 = (real_t *)malloc_shm(size);
	C2 = (real_t *)malloc_shm(size);
	D1 = (real_t *)malloc_shm(size);
	D2 = (real_t *)malloc_shm(size);
#else
	C1 = (real_t *)malloc(size);
	C2 = (real_t *)malloc(size);
	D1 = (real_t *)malloc(size);
	D2 = (real_t *)malloc(size);
#endif
	memset(C1, 0, size);
	memset(C2, 0, size);
	memset(D1, 0, size);
	memset(D2, 0, size);

	if (VECTOR) {
		size = NN * sizeof(real_t);
		K1Ex = (real_t *)malloc(size);
		K2Ex = (real_t *)malloc(size);
		K1Ey = (real_t *)malloc(size);
		K2Ey = (real_t *)malloc(size);
		K1Ez = (real_t *)malloc(size);
		K2Ez = (real_t *)malloc(size);
		K1Hx = (real_t *)malloc(size);
		K2Hx = (real_t *)malloc(size);
		K1Hy = (real_t *)malloc(size);
		K2Hy = (real_t *)malloc(size);
		K1Hz = (real_t *)malloc(size);
		K2Hz = (real_t *)malloc(size);
		memset(K1Ex, 0, size);
		memset(K2Ex, 0, size);
		memset(K1Ey, 0, size);
		memset(K2Ey, 0, size);
		memset(K1Ez, 0, size);
		memset(K2Ez, 0, size);
		memset(K1Hx, 0, size);
		memset(K2Hx, 0, size);
		memset(K1Hy, 0, size);
		memset(K2Hy, 0, size);
		memset(K1Hz, 0, size);
		memset(K2Hz, 0, size);
	}

	// ABC
	if      (iABC == 0) {
#ifdef _ONEAPI
		fMurHx = (mur_t *)malloc_shm(numMurHx * sizeof(mur_t));
		fMurHy = (mur_t *)malloc_shm(numMurHy * sizeof(mur_t));
		fMurHz = (mur_t *)malloc_shm(numMurHz * sizeof(mur_t));
#else
		fMurHx = (mur_t *)malloc(numMurHx * sizeof(mur_t));
		fMurHy = (mur_t *)malloc(numMurHy * sizeof(mur_t));
		fMurHz = (mur_t *)malloc(numMurHz * sizeof(mur_t));
#endif
		memset(fMurHx, 0, numMurHx * sizeof(mur_t));
		memset(fMurHy, 0, numMurHy * sizeof(mur_t));
		memset(fMurHz, 0, numMurHz * sizeof(mur_t));
	}
	else if (iABC == 1) {
		xsize = numPmlEx * sizeof(pml_t);
		ysize = numPmlEy * sizeof(pml_t);
		zsize = numPmlEz * sizeof(pml_t);
		fPmlEx = (pml_t *)malloc(xsize);
		fPmlEy = (pml_t *)malloc(ysize);
		fPmlEz = (pml_t *)malloc(zsize);
		memset(fPmlEx, 0, xsize);
		memset(fPmlEy, 0, ysize);
		memset(fPmlEz, 0, zsize);

		xsize = numPmlHx * sizeof(pml_t);
		ysize = numPmlHy * sizeof(pml_t);
		zsize = numPmlHz * sizeof(pml_t);
		fPmlHx = (pml_t *)malloc(xsize);
		fPmlHy = (pml_t *)malloc(ysize);
		fPmlHz = (pml_t *)malloc(zsize);
		memset(fPmlHx, 0, xsize);
		memset(fPmlHy, 0, ysize);
		memset(fPmlHz, 0, zsize);

		xsize = (Nx + (2 * cPML.l)) * sizeof(real_t);
		ysize = (Ny + (2 * cPML.l)) * sizeof(real_t);
		zsize = (Nz + (2 * cPML.l)) * sizeof(real_t);
		gPmlXn = (real_t *)malloc(xsize);
		gPmlXc = (real_t *)malloc(xsize);
		gPmlYn = (real_t *)malloc(ysize);
		gPmlYc = (real_t *)malloc(ysize);
		gPmlZn = (real_t *)malloc(zsize);
		gPmlZc = (real_t *)malloc(zsize);
		memset(gPmlXn, 0, xsize);
		memset(gPmlXc, 0, xsize);
		memset(gPmlYn, 0, ysize);
		memset(gPmlYc, 0, ysize);
		memset(gPmlZn, 0, zsize);
		memset(gPmlZc, 0, zsize);

		size = NMaterial * sizeof(real_t);
		rPmlE = (real_t *)malloc(size);
		rPmlH = (real_t *)malloc(size);
		rPml  = (real_t *)malloc(size);
		memset(rPmlE, 0, size);
		memset(rPmlH, 0, size);
		memset(rPml,  0, size);
	}

	// iteration
	Iter_size = (Solver.maxiter + 1) * sizeof(double);
	Eiter = (double *)malloc(Iter_size);
	Hiter = (double *)malloc(Iter_size);

	// feed
	if (NFeed > 0) {
		Feed_size = NFeed * (Solver.maxiter + 1) * sizeof(double);
#ifdef _ONEAPI
		VFeed = (double *)malloc_shm(Feed_size);
		IFeed = (double *)malloc_shm(Feed_size);
#else
		VFeed = (double *)malloc(Feed_size);
		IFeed = (double *)malloc(Feed_size);
#endif
	}

	// point
	if (NPoint > 0) {
		Point_size = (NPoint + 2) * (Solver.maxiter + 1) * sizeof(double);
#ifdef _ONEAPI
		VPoint = (double *)malloc_shm(Point_size);
#else
		VPoint = (double *)malloc(Point_size);
#endif
	}
	if ((NPoint > 0) && (NFreq1 > 0)) {
		Spara_size = NPoint * NFreq1 * sizeof(d_complex_t);
		Spara = (d_complex_t *)malloc(Spara_size);
	}

	// DFT factor
	if (NFreq2 > 0) {
		size = NFreq2 * (Solver.maxiter + 1) * sizeof(d_complex_t);
#ifdef _ONEAPI
		cEdft = (d_complex_t *)malloc_shm(size);
		cHdft = (d_complex_t *)malloc_shm(size);
#else
		cEdft = (d_complex_t *)malloc(size);
		cHdft = (d_complex_t *)malloc(size);
#endif
		memset(cEdft, 0, size);
		memset(cHdft, 0, size);

		size = NFreq2 * sizeof(d_complex_t);
		cFdft = (d_complex_t *)malloc(size);
		memset(cFdft, 0, size);
	}
}


void memfree1(void)
{
#ifdef _ONEAPI
	free_shm(SPlanewave);
	free_shm(C1);
	free_shm(C2);
	free_shm(D1);
	free_shm(D2);
#else
	free(C1);
	free(C2);
	free(D1);
	free(D2);
#endif

	if (VECTOR) {
		free(K1Ex);
		free(K2Ex);
		free(K1Ey);
		free(K2Ey);
		free(K1Ez);
		free(K2Ez);
		free(K1Hx);
		free(K2Hx);
		free(K1Hy);
		free(K2Hy);
		free(K1Hz);
		free(K2Hz);
	}

	if      (iABC == 0) {
#ifdef _ONEAPI
		free_shm(fMurHx);
		free_shm(fMurHy);
		free_shm(fMurHz);
#else
		free(fMurHx);
		free(fMurHy);
		free(fMurHz);
#endif
	}
	else if (iABC == 1) {
		free(fPmlEx);
		free(fPmlEy);
		free(fPmlEz);

		free(fPmlHx);
		free(fPmlHy);
		free(fPmlHz);

		free(gPmlXn);
		free(gPmlYn);
		free(gPmlZn);

		free(gPmlXc);
		free(gPmlYc);
		free(gPmlZc);

		free(rPmlE);
		free(rPmlH);
		free(rPml);
	}

	free(Xn);
	free(Yn);
	free(Zn);

	free(Xc);
	free(Yc);
	free(Zc);

#ifdef _ONEAPI
	free_shm(RXn);
	free_shm(RYn);
	free_shm(RZn);

	free_shm(RXc);
	free_shm(RYc);
	free_shm(RZc);
#else
	free(RXn);
	free(RYn);
	free(RZn);

	free(RXc);
	free(RYc);
	free(RZc);
#endif

	//free(Material);

	if (NGeometry > 0) {
		free(Geometry);
	}

	free(Eiter);
	free(Hiter);

	if (NFeed > 0) {
#ifdef _ONEAPI
		free_shm(Feed);
#else
		free(Feed);
#endif
	}

	if (NFreq1 > 0) {
		free(Freq1);
	}

	if (NFreq2 > 0) {
		free(Freq2);
	}

	if (NFeed > 0) {
#ifdef _ONEAPI
		free_shm(VFeed);
		free_shm(IFeed);
#else
		free(VFeed);
		free(IFeed);
#endif
	}

	if (NPoint > 0) {
		free(Point);
#ifdef _ONEAPI
		free_shm(VPoint);
#else
		free(VPoint);
#endif
	}
	if ((NPoint > 0) && (NFreq1 > 0)) {
		free(Spara);
	}

	if (NFreq2 > 0) {
#ifdef _ONEAPI
		free_shm(cEdft);
		free_shm(cHdft);
#else
		free(cEdft);
		free(cHdft);
#endif
		free(cFdft);
	}
}
