#ifndef NAU_RT_GEOMETRY_H
#define NAU_RT_GEOMETRY_H

#include "nau/config.h"
#if NAU_RT == 1

#include <map>
#include <string>
#include <vector>

#include "nau/render/rt/rtRenderer.h"

#include <glbinding/gl/gl.h>
#include <cuda_gl_interop.h>
#include <cuda.h>


namespace nau {
	namespace render {
		namespace rt {


			class RTGeometry {

			public:
				RTGeometry();


				struct CUDABuffer {
					cudaGraphicsResource *cgr;
					CUdeviceptr memPtr;
					size_t sizeInBytes;
					int numElements;

					CUDABuffer() {
						cgr = nullptr;
						memPtr = 0;
						sizeInBytes = 0;
						numElements = 0;
					}

					CUDABuffer(const CUDABuffer &cb) {

						cgr = cb.cgr;
						memPtr = cb.memPtr;
						sizeInBytes = cb.sizeInBytes;
						numElements = cb.numElements;
					}

					CUDABuffer& operator = (const CUDABuffer& cb) {
						cgr = cb.cgr;
						memPtr = cb.memPtr;
						sizeInBytes = cb.sizeInBytes;
						numElements = cb.numElements;
						return *this;
					}
				};
				struct CUDABuffers{
					// for vertex buffers. map from vertex attribute id to cuda buffer
					std::map<int, CUDABuffer> vertexBuffers;
					// map from material name to index buffer
					std::map < std::string, CUDABuffer> indexBuffers;
				};

				void addVertexAttribute(unsigned int attr);

				const OptixTraversableHandle& getTraversableHandle();
				const std::map<std::string, CUDABuffers>& getCudaVBOS();

				bool generateAccel(const std::vector<std::string>& sceneNames);

			protected:

				std::vector<int> m_VertexAttributes;

				// map from scene object name to CUDA VBOs
				std::map<std::string, CUDABuffers> m_CudaVBOs;

				OptixTraversableHandle rtTraversableHandle;

			};
		};
	};
};



#endif

#endif
