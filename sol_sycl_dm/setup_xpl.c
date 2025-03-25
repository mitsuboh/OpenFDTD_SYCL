#include "ofd.h"

#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

#ifdef _ONEAPI
void setup_xpl(void)
{
	size_t size, xsize, ysize, zsize; 	
	SPlanewave->theta = (real_t)Planewave.theta;
	SPlanewave->phi = (real_t)Planewave.phi;
	for (int m = 0; m < 3; m++) {
		SPlanewave->ei[m] = (real_t)Planewave.ei[m];
		SPlanewave->hi[m] = (real_t)Planewave.hi[m];
		SPlanewave->r0[m] = (real_t)Planewave.r0[m];
		SPlanewave->ri[m] = (real_t)Planewave.ri[m];
	}
	SPlanewave->ai = (real_t)Planewave.ai;
	myQ.memcpy(d_SPlanewave, SPlanewave, sizeof(splanewave)).wait();

       // mesh (real_t)

        h_Xn = (real_t *)malloc((Nx + 1) * sizeof(real_t));
        h_Yn = (real_t *)malloc((Ny + 1) * sizeof(real_t));
        h_Zn = (real_t *)malloc((Nz + 1) * sizeof(real_t));
        for (int i = 0; i <= Nx; i++) {
                h_Xn[i] = (real_t)Xn[i];
        }
        for (int j = 0; j <= Ny; j++) {
                h_Yn[j] = (real_t)Yn[j];
        }
        for (int k = 0; k <= Nz; k++) {
                h_Zn[k] = (real_t)Zn[k];
        }

        h_Xc = (real_t *)malloc((Nx + 0) * sizeof(real_t));
        h_Yc = (real_t *)malloc((Ny + 0) * sizeof(real_t));
        h_Zc = (real_t *)malloc((Nz + 0) * sizeof(real_t));
        for (int i = 0; i < Nx; i++) {
                h_Xc[i] = (real_t)Xc[i];
        }
        for (int j = 0; j < Ny; j++) {
                h_Yc[j] = (real_t)Yc[j];
        }
        for (int k = 0; k < Nz; k++) {
                h_Zc[k] = (real_t)Zc[k];
        }

	// mesh (real_t)

	s_Xn = (real_t*)malloc_dev((Nx + 1) * sizeof(real_t));
	s_Yn = (real_t*)malloc_dev((Ny + 1) * sizeof(real_t));
	s_Zn = (real_t*)malloc_dev((Nz + 1) * sizeof(real_t));
	d_RXn = (real_t*)malloc_dev((Nx + 1) * sizeof(real_t));
	d_RYn = (real_t*)malloc_dev((Ny + 1) * sizeof(real_t));
	d_RZn = (real_t*)malloc_dev((Nz + 1) * sizeof(real_t));
	myQ.memcpy(s_Xn, h_Xn, (Nx + 1) * sizeof(real_t)).wait();
	myQ.memcpy(s_Yn, h_Yn, (Ny + 1) * sizeof(real_t)).wait();
	myQ.memcpy(s_Zn, h_Zn, (Nz + 1) * sizeof(real_t)).wait();
	myQ.memcpy(d_RXn, RXn, (Nx + 1) * sizeof(real_t)).wait();
	myQ.memcpy(d_RYn, RYn, (Ny + 1) * sizeof(real_t)).wait();
	myQ.memcpy(d_RZn, RZn, (Nz + 1) * sizeof(real_t)).wait();
//	for (int i = 0; i <= Nx; i++) {
//		s_Xn[i] = h_Xn[i];
//	}

	s_Xc = (real_t*)malloc_dev((Nx + 0) * sizeof(real_t));
	s_Yc = (real_t*)malloc_dev((Ny + 0) * sizeof(real_t));
	s_Zc = (real_t*)malloc_dev((Nz + 0) * sizeof(real_t));
	d_RXc = (real_t*)malloc_dev((Nx + 0) * sizeof(real_t));
	d_RYc = (real_t*)malloc_dev((Ny + 0) * sizeof(real_t));
	d_RZc = (real_t*)malloc_dev((Nz + 0) * sizeof(real_t));
	myQ.memcpy(s_Xc, h_Xc, (Nx + 0) * sizeof(real_t)).wait();
	myQ.memcpy(s_Yc, h_Yc, (Ny + 0) * sizeof(real_t)).wait();
	myQ.memcpy(s_Zc, h_Zc, (Nz + 0) * sizeof(real_t)).wait();
	myQ.memcpy(d_RXc, RXc, (Nx + 0) * sizeof(real_t)).wait();
	myQ.memcpy(d_RYc, RYc, (Ny + 0) * sizeof(real_t)).wait();
	myQ.memcpy(d_RZc, RZc, (Nz + 0) * sizeof(real_t)).wait();

	      // material ID

        size = NN * sizeof(id_t);
	d_iEx = (id_t*)malloc_dev(size);
	d_iEy = (id_t*)malloc_dev(size);
	d_iEz = (id_t*)malloc_dev(size);
	d_iHx = (id_t*)malloc_dev(size);
	d_iHy = (id_t*)malloc_dev(size);
	d_iHz = (id_t*)malloc_dev(size);
	myQ.memcpy(d_iEx, iEx, size).wait();
	myQ.memcpy(d_iEy, iEy, size).wait();
	myQ.memcpy(d_iEz, iEz, size).wait();
	myQ.memcpy(d_iHx, iHx, size).wait();
	myQ.memcpy(d_iHy, iHy, size).wait();
	myQ.memcpy(d_iHz, iHz, size).wait();

	        // material factor

        size = NMaterial * sizeof(real_t);
	d_C1 = (real_t*)malloc_dev(size);
	d_C2 = (real_t*)malloc_dev(size);
	d_D1 = (real_t*)malloc_dev(size);
	d_D2 = (real_t*)malloc_dev(size);
	myQ.memcpy(d_C1, C1, size).wait();
	myQ.memcpy(d_C2, C2, size).wait();
	myQ.memcpy(d_D1, D1, size).wait();
	myQ.memcpy(d_D2, D2, size).wait();

		// ABC

	if (iABC == 0) {
		xsize = numMurHx * sizeof(mur_t);
		ysize = numMurHy * sizeof(mur_t);
		zsize = numMurHz * sizeof(mur_t);
		d_fMurHx = (mur_t*)malloc_dev(xsize);
		d_fMurHy = (mur_t*)malloc_dev(ysize);
		d_fMurHz = (mur_t*)malloc_dev(zsize);
		myQ.memcpy(d_fMurHx, fMurHx, xsize).wait();
		myQ.memcpy(d_fMurHy, fMurHy, ysize).wait();
		myQ.memcpy(d_fMurHz, fMurHz, zsize).wait();
	}

}
#endif
