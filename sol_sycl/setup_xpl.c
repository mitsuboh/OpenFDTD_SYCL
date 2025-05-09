#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

#ifdef _ONEAPI
void setup_xpl(void)
{

	for (int m = 0; m < 3; m++) {
		SPlanewave.ei[m] = (real_t)Planewave.ei[m];
		SPlanewave.hi[m] = (real_t)Planewave.hi[m];
		SPlanewave.r0[m] = (real_t)Planewave.r0[m];
		SPlanewave.ri[m] = (real_t)Planewave.ri[m];
	}
	SPlanewave.ai = (real_t)Planewave.ai;

	// mesh (real_t)

	s_Xn = (real_t*)malloc_shm((Nx + 1) * sizeof(real_t));
	s_Yn = (real_t*)malloc_shm((Ny + 1) * sizeof(real_t));
	s_Zn = (real_t*)malloc_shm((Nz + 1) * sizeof(real_t));
	for (int i = 0; i <= Nx; i++) {
		s_Xn[i] = (real_t)Xn[i];
	}
	for (int j = 0; j <= Ny; j++) {
		s_Yn[j] = (real_t)Yn[j];
	}
	for (int k = 0; k <= Nz; k++) {
		s_Zn[k] = (real_t)Zn[k];
	}

	s_Xc = (real_t*)malloc_shm((Nx + 0) * sizeof(real_t));
	s_Yc = (real_t*)malloc_shm((Ny + 0) * sizeof(real_t));
	s_Zc = (real_t*)malloc_shm((Nz + 0) * sizeof(real_t));
	for (int i = 0; i < Nx; i++) {
		s_Xc[i] = (real_t)Xc[i];
	}
	for (int j = 0; j < Ny; j++) {
		s_Yc[j] = (real_t)Yc[j];
	}
	for (int k = 0; k < Nz; k++) {
		s_Zc[k] = (real_t)Zc[k];
	}
}
#endif
