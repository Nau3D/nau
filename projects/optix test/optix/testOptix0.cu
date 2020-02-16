
#include <optix.h>
#include "LaunchParams2.h" // our launch params
#include <vec_math.h> // NVIDIAs math utils
  
extern "C" {

__constant__ LaunchParams optixLaunchParams;

// closest hit program
extern "C" __global__ void __closesthit__radiance() { 
  
}
  
// any hit program
extern "C" __global__ void __anyhit__radiance() {
	
}

// miss program 
extern "C" __global__ void __miss__radiance() {
	
}

//------------------------------------------------------------------------------
// ray gen program 
//------------------------------------------------------------------------------


extern "C" __global__ void __raygen__renderFrame()  {

	// printf only once

	const uint3 index = optixGetLaunchIndex();
	
	// compute a test pattern based on pixel ID

	uint3 aux = index - optixGetLaunchDimensions() * 0.5;
	float2 pixelCenteredCoord = make_float2(aux.x, aux.y);
	float pixelIntensity = 0.5 + 0.5 * cos((pixelCenteredCoord.x + optixLaunchParams.frame.frame) * 0.1f);

	const int r = 0;
	const int g = (pixelIntensity * 255);
	const int b = (1 - pixelIntensity * 255);

	// convert to 32-bit rgba value - explicitly set alpha to 0xff
	const uint32_t rgba = 0xff000000  | (r<<0) | (g<<8) | (b<<16);

	const unsigned int fbIndex = index.x + (index.y * optixGetLaunchDimensions().x);
	
	if (optixLaunchParams.frame.frame == 0 && index.x == 0 && index.y == 0) {

		// print info to console
		printf("===========================================\n");
		printf("Nau Ray-Tracing Hello World\n");
		printf("Lunch size: %i x %i\n", index.x, index.y);
		printf("Camera Direction: %f %f %f\n", 
				optixLaunchParams.camera.direction.x,
				optixLaunchParams.camera.direction.y,
				optixLaunchParams.camera.direction.z
			);
		printf("===========================================\n");
	}

	optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  
} // end of extern
