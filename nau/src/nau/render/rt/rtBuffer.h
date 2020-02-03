#ifndef _NAU_OPTIX_CUDA_BUFFER_
#define _NAU_OPTIX_CUDA_BUFFER_

#include <assert.h>

#include <cuda.h>

#include "rtException.h"

namespace nau {

	namespace render {

		namespace rt{

			class RTBuffer {

			public:

				RTBuffer() : byteSize(0), ptr(nullptr) {

				}

				~RTBuffer() {
				}

				void clear() {
					CUDA_CHECK(cudaFree(ptr));
					ptr = nullptr;
					byteSize = 0;
				}

				void setSize(size_t aSize) {

					if (ptr != nullptr)
						clear();
					byteSize = aSize;
					CUDA_CHECK(cudaMalloc((void**)&ptr, byteSize));
				}

				void copy(void* data) {
					CUDA_CHECK(cudaMemcpy(ptr, (void*)data, byteSize, cudaMemcpyHostToDevice));
				}

				//! alloc and store values
				void store(void* data, size_t size) {
					assert(ptr == nullptr);
					byteSize = size;
					CUDA_CHECK(cudaMalloc((void**)&ptr, byteSize));
					CUDA_CHECK(cudaMemcpy(ptr, (void*)data, size, cudaMemcpyHostToDevice));

				}

				CUdeviceptr getPtr() {
					return (CUdeviceptr)ptr;
				}

				void getData(void *data) {
					assert(ptr != nullptr);
					CUDA_CHECK(cudaMemcpy(data, ptr,
						byteSize, cudaMemcpyDeviceToHost));
				}

				void getData(void* data, size_t s) {
					assert(ptr != nullptr);
					assert(s <= byteSize);
					CUDA_CHECK(cudaMemcpy(data, ptr,
						s, cudaMemcpyDeviceToHost));
				}
				size_t getSize() {
					return byteSize;
				}

			protected:

				size_t byteSize;
				void* ptr;
				
			};
		}
	};
};



#endif