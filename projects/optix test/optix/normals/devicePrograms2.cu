// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include "vec_math.h"
#include "LaunchParams2.h"


  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" {
    __constant__ LaunchParams optixLaunchParams;
  }

  // for this simple example, we have a single ray type
  enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };
  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // compute normal:
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];
    const float3 &A     = make_float3(sbtData.vertexD.position[index.x]);
    const float3 &B     = make_float3(sbtData.vertexD.position[index.y]);
    const float3 &C     = make_float3(sbtData.vertexD.position[index.z]);
    const float3 Ng     = normalize(cross(B-A,C-A)) * 0.5 + 0.5;

    const float3 rayDir = optixGetWorldRayDirection();
    //const float cosDN  = 0.2f + .8f*fabsf(dot(rayDir,Ng));
    const float cosDN  =  0.5 + 0.5 * dot(rayDir,Ng);
    float3 &prd = *(float3*)getPRD<float3>();
    prd = Ng;
    //prd = cosDN * sbtData.color;
  //  prd = make_float3(0.0,1.0,0.0);
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }


  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    float3 &prd = *(float3*)getPRD<float3>();
    // set to constant white as background color
    prd = make_float3(0.0f, 0.0f, 1.0f);
  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    float3 pixelColorPRD = make_float3(1.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );

    // normalized screen plane position, in [0,1]^2
    const float2 screen(make_float2(ix+.5f,iy+.5f)
                       / make_float2(optixLaunchParams.frame.size) * 2.0 - 1.0);
    
    // generate ray direction
    float3 rayDir = normalize(camera.direction
                             + screen.x  * camera.horizontal
                             + screen.y * camera.vertical);

    RayGenData *rgd = (RayGenData *)optixGetSbtDataPointer();
    int3 ci = rgd->color;
	if (optixLaunchParams.frame.frame == 0 &&
			optixGetLaunchIndex().x == 0 &&
			optixGetLaunchIndex().y == 0) {

        printf("Cam dir: %f %f %f\n", camera.direction.x, camera.direction.y, camera.direction.z);
        printf("Ray dir: %f %f %f\n", rayDir.x, rayDir.y, rayDir.z);

      }
         optixTrace(optixLaunchParams.traversable,
               camera.position,
               rayDir,
               0.f,    // tmin
               1e20f,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask( 255 ),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,             // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SURFACE_RAY_TYPE,             // missSBTIndex 
               u0, u1 );

    const int r = int(255.99f*pixelColorPRD.x);
    const int g = int(255.99f*pixelColorPRD.y);
    const int b = int(255.99f*pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);

    const uint32_t rgba2 = 0xff000000
      | (ci.x<<0) | (ci.y<<8) | (ci.z<<16);
    // and write to frame buffer ...
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  }
  

