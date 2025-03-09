/*
memallocfree2.c

alloc and free
(2) iteration variables
*/

#include "ofd.h"

#ifdef _ONEAPI
#undef C        // C is used for (2.99792458e8) but &lt;CL/sycl.
#include "ofd_dpcpp.h"
#endif

void memalloc2(void)
{
	size_t size, xsize, ysize, zsize;

	size = NN * sizeof(real_t);
#ifdef _ONEAPI
	Ex = (real_t *)malloc_shm(size);
	Ey = (real_t *)malloc_shm(size);
	Ez = (real_t *)malloc_shm(size);
	Hx = (real_t *)malloc_shm(size);
	Hy = (real_t *)malloc_shm(size);
	Hz = (real_t *)malloc_shm(size);
        SPlanewave = (splanewave*)malloc_shm(sizeof(splanewave));
#else
	Ex = (real_t *)malloc(size);
	Ey = (real_t *)malloc(size);
	Ez = (real_t *)malloc(size);
	Hx = (real_t *)malloc(size);
	Hy = (real_t *)malloc(size);
	Hz = (real_t *)malloc(size);
#endif

	// ABC
	if      (iABC == 1) {
		xsize = numPmlEx * sizeof(real_t);
		ysize = numPmlEy * sizeof(real_t);
		zsize = numPmlEz * sizeof(real_t);
		Exy = (real_t *)malloc(xsize);
		Exz = (real_t *)malloc(xsize);
		Eyz = (real_t *)malloc(ysize);
		Eyx = (real_t *)malloc(ysize);
		Ezx = (real_t *)malloc(zsize);
		Ezy = (real_t *)malloc(zsize);

		xsize = numPmlHx * sizeof(real_t);
		ysize = numPmlHy * sizeof(real_t);
		zsize = numPmlHz * sizeof(real_t);
		Hxy = (real_t *)malloc(xsize);
		Hxz = (real_t *)malloc(xsize);
		Hyz = (real_t *)malloc(ysize);
		Hyx = (real_t *)malloc(ysize);
		Hzx = (real_t *)malloc(zsize);
		Hzy = (real_t *)malloc(zsize);
	}
}


void memfree2(void)
{
#ifdef _ONEAPI
	free_shm(Ex);
	free_shm(Ey);
	free_shm(Ez);
	free_shm(Hx);
	free_shm(Hy);
	free_shm(Hz);
	free_shm(SPlanewave);

	free_shm(iEx);
	free_shm(iEy);
	free_shm(iEz);
	free_shm(iHx);
	free_shm(iHy);
	free_shm(iHz);
#else
	free(Ex);
	free(Ey);
	free(Ez);
	free(Hx);
	free(Hy);
	free(Hz);

	free(iEx);
	free(iEy);
	free(iEz);
	free(iHx);
	free(iHy);
	free(iHz);
#endif

	if      (iABC == 1) {
		free(Exy);
		free(Exz);
		free(Eyz);
		free(Eyx);
		free(Ezx);
		free(Ezy);

		free(Hxy);
		free(Hxz);
		free(Hyz);
		free(Hyx);
		free(Hzx);
		free(Hzy);
	}
}
