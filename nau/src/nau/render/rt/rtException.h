
#pragma once

// optix 7
#include <cuda_runtime.h>
#include <optix.h>
//#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)								\
    {													\
      cudaError_t rc = call;                            \
      if (rc != cudaSuccess) {                          \
        std::stringstream txt;                          \
        cudaError_t err =  rc;						    \
        txt << "CUDA Error " << cudaGetErrorName(err)   \
            << " (" << cudaGetErrorString(err) << ")";  \
		throw std::runtime_error(txt.str());            \
      }                                                 \
    }

#define OPTIX_CHECK( call )                             \
  {                                                     \
    OptixResult res = call;                             \
    if( res != OPTIX_SUCCESS ) {                        \
        std::stringstream txt;                          \
		txt << "Optix call (" << #call << 				\
			") failed with code " << res << " (line " <<\
			__LINE__  << ")\n";							\
		throw std::runtime_error(txt.str());            \
	}                                                   \
  }

#define CUDA_SYNC_CHECK()                               \
  {                                                     \
    cudaError_t rc = cudaDeviceSynchronize();           \
    if (rc != cudaSuccess) {                            \
        std::stringstream txt;							\
		txt << "error (" << __FILE__ << ": line " <<    \
			__LINE__ << "): " <<						\
			cudaGetErrorString(rc) << "\n";			    \
		throw std::runtime_error(txt.str());            \
	}                                                   \
  }

