
#include <optix.h>

#include "LaunchParams.h"

#include <vec_math.h>
  
extern "C" {

__constant__ LaunchParams optixLaunchParams;

  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //------------------------------------------------------------------------------
  

extern "C" __global__ void __closesthit__radiance() { 
  
}
  

extern "C" __global__ void __anyhit__radiance() {
	
}


  
//------------------------------------------------------------------------------
// miss program 
// ------------------------------------------------------------------------------
  

extern "C" __global__ void __miss__radiance() {
	
}



//------------------------------------------------------------------------------
// ray gen program 
//------------------------------------------------------------------------------


extern "C" __global__ void __raygen__renderFrame()  {

	//const float4 &color = *(const float4) optixGetSbtDataPointer();
	// printf only one
	if (optixLaunchParams.frameID == 0 &&
			optixGetLaunchIndex().x == 0 &&
			optixGetLaunchIndex().y == 0) {

		printf("############################################\n");
		printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
				optixLaunchParams.fbSize.x,
				optixLaunchParams.fbSize.y);
			//	printf("Color: %f %f %f %f\n", color[0], color[1], color[2], color[3]);
		printf("############################################\n");
	}

 
	// compute a test pattern based on pixel ID
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	const int r = (ix % 256);
	const int g = (iy % 256);
	const int b = ((ix+iy) % 256);

	// convert to 32-bit rgba value - explicitly set alpha to 0xff
	const uint32_t rgba = 0xff000000  | (r<<0) | (g<<8) | (b<<16);
	const unsigned int fbIndex = ix+iy*optixLaunchParams.fbSize.x;
	optixLaunchParams.colorBuffer[fbIndex] = rgba;
}
  
}
