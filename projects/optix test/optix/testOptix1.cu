
#include <optix.h>
#include "LaunchParams.h" // our launch params
#include <vec_math.h> // NVIDIAs math utils


extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}
//  a single ray type
enum { PHONG_RAY_TYPE=0, RAY_TYPE_COUNT };


// -------------------------------------------------------
// closest hit computes color based only on the triangle normal

extern "C" __global__ void __closesthit__phong() {

    // get mesh data
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // fetch vertex position for the three vertices
    const float3 &A     = make_float3(sbtData.vertexD.position[index.x]);
    const float3 &B     = make_float3(sbtData.vertexD.position[index.y]);
    const float3 &C     = make_float3(sbtData.vertexD.position[index.z]);

    // compute triangle normal:
    const float3 Ng     = normalize(cross(B-A,C-A)) * 0.5 + 0.5;

    // get the payload variable
    float3 &prd = *(float3*)getPRD<float3>();
    // output will be the normal
    prd = Ng;
}
  

// nothing to do in here
extern "C" __global__ void __anyhit__phong() {
}


// miss sets the bacgground color
extern "C" __global__ void __miss__phong() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.0f, 1.0f);
}


// ray gen program - responsible for launching primary rays
extern "C" __global__ void __raygen__renderFrame() {

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;  
    
    // ray payload
    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    // compute ray direction
    // normalized screen plane position, in [-1, 1]^2
    const float2 screen(make_float2(ix+.5f,iy+.5f)
                    / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
  
    // note: nau already takes into account the field of view when computing 
    // camera horizontal and vertival
    float3 rayDir = normalize(camera.direction
                           + screen.x  * camera.horizontal
                           + screen.y * camera.vertical);
    
    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
             camera.position,
             rayDir,
             0.f,    // tmin
             1e20f,  // tmax
             0.0f,   // rayTime
             OptixVisibilityMask( 255 ),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             PHONG_RAY_TYPE,             // SBT offset
             RAY_TYPE_COUNT,             // SBT stride
             PHONG_RAY_TYPE,             // missSBTIndex 
             u0, u1 );

    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*pixelColorPRD.x);
    const int g = int(255.0f*pixelColorPRD.y);
    const int b = int(255.0f*pixelColorPRD.z);

    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);
    // compute index
    const uint32_t fbIndex = ix+iy*optixGetLaunchDimensions().x;
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  

