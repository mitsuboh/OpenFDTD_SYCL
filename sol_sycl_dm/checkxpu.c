/*
check_xpu.cpp

return = 0/1 : OK/NG
device : I : device number (=0,1,...)
*/

#include <stdio.h>
#include <ofd.h>
#ifdef _ONEAPI
#undef C	// C is used for (2.99792458e8) but <CL/sycl.hpp> refuses it
#include "ofd_dpcpp.h"
#endif

#ifdef _ONEAPI

void check_xpu(sycl::queue* myQ, int idevice) try {
	int deviceCount;
	auto devices = sycl::device::get_devices();
    std::cout << "Devices found:" << std::endl;
    for (auto device : devices) {
        std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
            << std::endl;
    }
    deviceCount = devices.size();
    if (deviceCount <= 0) {
        std::cerr << "*** There is no device supporting Sycl" << std::endl;
		std::exit(1);
	}
	
	if ((idevice < 0) || (idevice >= deviceCount)) {
		std::cerr << "*** Invalid device number = " << std::dec << idevice << std::endl;
		std::exit(1);
	}
	sycl::device myDevice = devices[idevice];
	sycl::queue myQ2(myDevice);
	*myQ = myQ2;

    // GPU info
    printf("%s, Global MEM %luMB, MaxWG %zu, MaxSG %d, Max EUCount %d \n", myDevice.get_info<sycl::info::device::name>().c_str(),
       (myDevice.get_info<sycl::info::device::global_mem_size>() / 1024 / 1024),
        myDevice.get_info<sycl::info::device::max_work_group_size>(),
        myDevice.get_info<sycl::info::device::max_num_sub_groups>(),
        myDevice.get_info<sycl::info::device::max_compute_units>());
    //        int max_dim = myDevice.get_info<sycl::info::device::max_work_item_sizes<3>>();
     //       sprintf(msg, " WK Item DIM %d, WK item sizes (%d)", myDevice.get_info<sycl::info::device::max_work_item_dimensions>(),
    //            myDevice.get_info<sycl::info::device::max_work_item_sizes<1>>());
                //            max_dim[0],max_dim[1],max_dim[2]);

	
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif

