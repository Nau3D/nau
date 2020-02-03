#if NAU_RT == 1
#ifndef NAU_RT_RENDERER_H
#define NAU_RT_RENDERER_H

#include "nau/config.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <string>
namespace nau
{
	namespace render
	{
		namespace rt {

			class RTRenderer  {


			public:

				typedef enum {
					ANY_HIT,
					CLOSEST_HIT,
					MISS
				} SHADER_TYPES;

				virtual ~RTRenderer() {};

				static bool Init();
				static bool isReady();
				static const OptixDeviceContext& getOptixContext();
				static const CUstream& getOptixStream();

				static int getProcType(const std::string& pType);

			protected:
				RTRenderer() {};
				static void Log(unsigned int level, const char* tag, 
					const char* message, void*);
				static CUstream cuStream;
				static cudaDeviceProp cuDeviceProps;
				static CUcontext cuContext;
				static OptixDeviceContext optixContext;
				static bool rtOK;
			};
		};
	};
};
#endif // PassOptix Class


#endif







