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
} splanewave_t;

#ifdef MAIN
sycl::device myDevice = sycl::device(sycl::cpu_selector_v);
sycl::queue myQ(myDevice);
#else
extern sycl::device myDevice;
extern sycl::queue myQ;
#endif
EXTERN splanewave_t  SPlanewave;
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
EXTERN real_t* d_C1, * d_C2;
EXTERN real_t* d_D1, * d_D2;
EXTERN mur_t* d_fMurHx, * d_fMurHy, * d_fMurHz;
EXTERN feed_t * d_Feed;
EXTERN double * d_VFeed, * d_IFeed;
EXTERN real_t       *d_cEx_r, *d_cEx_i, *d_cEy_r, *d_cEy_i, *d_cEz_r, *d_cEz_i;
EXTERN real_t       *d_cHx_r, *d_cHx_i, *d_cHy_r, *d_cHy_i, *d_cHz_r, *d_cHz_i;

extern void             check_xpu(sycl::queue * ,int);

#endif   //  _OFD_DPCPP_H_
