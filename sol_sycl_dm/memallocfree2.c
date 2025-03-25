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
	Ex = (real_t *)malloc_dev(size);
	Ey = (real_t *)malloc_dev(size);
	Ez = (real_t *)malloc_dev(size);
	Hx = (real_t *)malloc_dev(size);
	Hy = (real_t *)malloc_dev(size);
	Hz = (real_t *)malloc_dev(size);
        SPlanewave = (splanewave*)malloc(sizeof(splanewave));
        d_SPlanewave = (splanewave*)malloc_dev(sizeof(splanewave));
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

        // feed
#ifdef _ONEAPI
	if (NFeed > 0) {
	size = NFeed * sizeof(feed_t);
	d_Feed = (feed_t*)malloc_dev(size);
	myQ.memcpy(d_Feed, Feed, size).wait();

	d_VFeed = (double*)malloc_dev(Feed_size);
	d_IFeed = (double*)malloc_dev(Feed_size);
        }
#endif
}


void memfree2(void)
{
#ifdef _ONEAPI
	free_dev(Ex);
	free_dev(Ey);
	free_dev(Ez);
	free_dev(Hx);
	free_dev(Hy);
	free_dev(Hz);
	free_dev(d_cEx_r);
	free_dev(d_cEy_r);
	free_dev(d_cEz_r);
	free_dev(d_cHx_r);
	free_dev(d_cHy_r);
	free_dev(d_cHz_r);
	free_dev(d_cEx_i);
	free_dev(d_cEy_i);
	free_dev(d_cEz_i);
	free_dev(d_cHx_i);
	free_dev(d_cHy_i);
	free_dev(d_cHz_i);

	free(SPlanewave);
	free_dev(d_SPlanewave);
#else
	free(Ex);
	free(Ey);
	free(Ez);
	free(Hx);
	free(Hy);
	free(Hz);
#endif

	free(iEx);
	free(iEy);
	free(iEz);
	free(iHx);
	free(iHy);
	free(iHz);

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
#ifdef _ONEAPI
        if (NFeed > 0) {
                free_dev(d_Feed);
                free_dev(d_VFeed);
                free_dev(d_IFeed);
        }
#endif
}
