#include "nau/config.h"

#if NAU_RT == 1


#include "nau/render/rt/rtRenderer.h"

#include "nau/render/rt/rtException.h"
#include "nau/slogger.h"

#include <optix_function_table_definition.h>

using namespace nau::render::rt;

bool RTRenderer::rtOK = false;
CUstream RTRenderer::cuStream;
cudaDeviceProp RTRenderer::cuDeviceProps;
CUcontext RTRenderer::cuContext;
OptixDeviceContext RTRenderer::optixContext;



bool 
RTRenderer::Init() {

	if (rtOK == false) {

		int numDevices;
		int deviceID = 0;

		try {
			// check for available optix7 capable devices
			cudaFree(&deviceID);
			cudaGetDeviceCount(&numDevices);
			SLOG("RT: found %d CUDA devices", numDevices);

			// return if no devices are found
			if (numDevices == 0)
				return false;

			// initialize optix
			OPTIX_CHECK(optixInit());

			CUDA_CHECK(cudaSetDevice(0));
			CUDA_CHECK(cudaStreamCreate(&cuStream));

			cudaGetDeviceProperties(&cuDeviceProps, deviceID);
			SLOG("RT: running on device: %s", cuDeviceProps.name);

			cuContext = 0;  // zero means take the current context

			OptixDeviceContextOptions options = {};
			options.logCallbackFunction = &Log;
			options.logCallbackLevel = 4;
			OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &optixContext));
		}
		catch (std::exception const& e)
		{
			SLOG("Exception in RT renderer init: %s", e.what());
			rtOK = false;
		}
		rtOK = true;
	}
	return rtOK;
}


bool 
RTRenderer::isReady() {
	return rtOK;
}


const OptixDeviceContext& 
RTRenderer::getOptixContext() {
	return optixContext;
}


const CUstream& 
RTRenderer::getOptixStream() {
	return cuStream;
}


int 
RTRenderer::getProcType(const std::string& pType) {

	if (pType == "ANY_HIT")
		return RTRenderer::ANY_HIT;
	else if (pType == "CLOSEST_HIT")
		return RTRenderer::CLOSEST_HIT;
	else if (pType == "MISS")
		return RTRenderer::MISS;
	else
		return -1;
}

void 
RTRenderer::Log(unsigned int level, const char* tag, const char* message,	void*) {

	SLOG("RT: [%2d][%12s]: %s\n", (int)level, tag, message);
}



#endif