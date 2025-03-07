/*
OpenFDTD Version DPCPP Header
  ofd_dpcpp.h
*/
#ifndef _OFD_DPCPP_H_
#define _OFD_DPCPP_H_

#include <sycl/sycl.hpp>

#define CEIL(n,d) (((n) + (d) - 1) / (d))

#define malloc_shm(size) ((void*)sycl::malloc_shared(size, myQ))
#define malloc_dev(size) ((void*)sycl::malloc_device(size, myQ))
#define free_shm(ptr) (sycl::free(ptr, myQ))
#define free_dev(ptr) (sycl::free(ptr, myQ))

typedef struct {
	real_t theta, phi;        // direction
	real_t ei[3], hi[3];      // E and H unit vector
	real_t ri[3], r0[3], ai;  // incidence vector and factor
	int    pol;               // polarization : 1=V, 2=H
} splanewave;

#ifdef MAIN
sycl::device myDevice = sycl::device(sycl::cpu_selector_v);
sycl::queue myQ(myDevice);
splanewave* SPlanewave, * d_SPlanewave;
#else
extern sycl::device myDevice;
extern sycl::queue myQ;
extern splanewave* SPlanewave, * d_SPlanewave;
#endif
EXTERN int CPU;

// Host FP32 data
EXTERN real_t* h_Xn, * h_Yn, * h_Zn;
EXTERN real_t* h_Xc, * h_Yc, * h_Zc;

// Devxe FP32 data
EXTERN real_t* s_Xn, * s_Yn, * s_Zn;
EXTERN real_t* s_Xc, * s_Yc, * s_Zc;

// Device memory
EXTERN real_t* d_RXn, * d_RYn, * d_RZn;
EXTERN real_t* d_RXc, * d_RYc, * d_RZc;
EXTERN id_t *d_iEx, *d_iEy, *d_iEz;        // material ID of E
EXTERN id_t *d_iHx, *d_iHy, *d_iHz;        // material ID of H
EXTERN real_t* d_C1, * d_C2, * d_C3, * d_C4;
EXTERN real_t* d_D1, * d_D2, * d_D3, * d_D4;
EXTERN mur_t* d_fMurHx, * d_fMurHy, * d_fMurHz;
EXTERN feed_t * d_Feed;
EXTERN double * d_VFeed, * d_IFeed;
EXTERN real_t       *d_Ex_r, *d_Ex_i, *d_Ey_r, *d_Ey_i, *d_Ez_r, *d_Ez_i;
EXTERN real_t       *d_Hx_r, *d_Hx_i, *d_Hy_r, *d_Hy_i, *d_Hz_r, *d_Hz_i;

extern void             check_xpu(sycl::queue * ,int);

#endif   //  _OFD_DPCPP_H_
