
#include "optixParams.h" // our launch params
  
extern "C" {
// name "optixLaunchParams" is hardcoded in Nau
__constant__ LaunchParams optixLaunchParams;

// -----------------------------------------------------------------------------
// Nau requires at least one hit and miss groups

// closest hit program
extern "C" __global__ void __closesthit__phong() { 
  
}
  
// any hit program
extern "C" __global__ void __anyhit__phong() {
	
}

// miss program 
extern "C" __global__ void __miss__phong() {
	
}

//------------------------------------------------------------------------------
// ray gen program 
//------------------------------------------------------------------------------


extern "C" __global__ void __raygen__renderFrame()  {


	const uint3 index = optixGetLaunchIndex();
	
	// compute a test pattern based on pixel ID

    // compute some color based on launch index
	uint3 aux = index - optixGetLaunchDimensions() * 0.5;
    float pixelIntensity = 0.5 + 0.5 * 
        cos((aux.x + optixLaunchParams.frame.frame * 0.1) * 0.2f);

	const int r = 0;
	const int g = (pixelIntensity * 255);
	const int b = (1 - pixelIntensity * 255);

	// convert to 32-bit rgba value - explicitly set alpha to 0xff
	const uint32_t rgba = 0xff000000  | (r<<0) | (g<<8) | (b<<16);

    // compute output buffer index based on laaunch index and dimensions
	const unsigned int fbIndex = index.x + (index.y * optixGetLaunchDimensions().x);
    
    // print only once
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
    // fill output buffer -> this will be copied to a render target
	optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  
} // end of extern
