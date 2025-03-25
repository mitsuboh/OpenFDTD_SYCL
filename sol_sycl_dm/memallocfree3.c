/*
memallocfree3.c

alloc and free
(3) near3d
*/

#include "ofd.h"
#include "ofd_prototype.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

void memalloc3(void)
{
	if ((NN > 0) && (NFreq2 > 0)) {
		const size_t size = NFreq2 * NN * sizeof(float);

#ifdef _ONEAPI
		d_cEx_r = (float *)malloc_dev(size);
		d_cEx_i = (float *)malloc_dev(size);
		d_cEy_r = (float *)malloc_dev(size);
		d_cEy_i = (float *)malloc_dev(size);
		d_cEz_r = (float *)malloc_dev(size);
		d_cEz_i = (float *)malloc_dev(size);
		d_cHx_r = (float *)malloc_dev(size);
		d_cHx_i = (float *)malloc_dev(size);
		d_cHy_r = (float *)malloc_dev(size);
		d_cHy_i = (float *)malloc_dev(size);
		d_cHz_r = (float *)malloc_dev(size);
		d_cHz_i = (float *)malloc_dev(size);
#endif
		cEx_r = (float *)malloc(size);
		cEx_i = (float *)malloc(size);
		cEy_r = (float *)malloc(size);
		cEy_i = (float *)malloc(size);
		cEz_r = (float *)malloc(size);
		cEz_i = (float *)malloc(size);
		cHx_r = (float *)malloc(size);
		cHx_i = (float *)malloc(size);
		cHy_r = (float *)malloc(size);
		cHy_i = (float *)malloc(size);
		cHz_r = (float *)malloc(size);
		cHz_i = (float *)malloc(size);
	}
}


void memfree3(void)
{
	if ((NN > 0) && (NFreq2 > 0)) {
		free(cEx_r);
		free(cEx_i);
		free(cEy_r);
		free(cEy_i);
		free(cEz_r);
		free(cEz_i);
		free(cHx_r);
		free(cHx_i);
		free(cHy_r);
		free(cHy_i);
		free(cHz_r);
		free(cHz_i);
	}
}
