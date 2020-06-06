
#include "optixParams.h" // our launch params



extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}
//  a single ray type
enum { PHONG=0, SHADOW, RAY_TYPE_COUNT };


// -------------------------------------------------------
// closest hit computes color based lolely on the triangle normal

extern "C" __global__ void __closesthit__radiance() {

    float3 &prd = *(float3*)getPRD<float3>();

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];

    // direction towards light
    float3 lDir = make_float3(-optixLaunchParams.global->lightDir);
    lDir = normalize(lDir);
    float3 nn = normalize(make_float3(n));

    float intensity = max(dot(lDir, nn),0.0f);

    // ray payload
    float shadowAttPRD = 1.0f;
    uint32_t u0, u1;
    packPointer( &shadowAttPRD, u0, u1 );  

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    //(1.f-u-v) * A + u * B + v * C;
    
    // trace shadow ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        lDir,
        0.001f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, //OPTIX_RAY_FLAG_NONE,
        SHADOW,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        SHADOW,             // missSBTIndex 
        u0, u1 );

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {  
        // get barycentric coordinates
        // compute pixel texture coordinate
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        prd = make_float3(fromTexture) * min(intensity * shadowAttPRD + 0.3, 1.0);
    }
    else
        prd = sbtData.diffuse * min(intensity * shadowAttPRD + 0.3, 1.0);
}


// any hit to ignore intersections with back facing geometry
extern "C" __global__ void __anyhit__radiance() {

}


// miss sets the background color
extern "C" __global__ void __miss__radiance() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.0f, 1.0f);
}


// -----------------------------------------------
// Shadow rays

extern "C" __global__ void __closesthit__shadow() {

}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow() {

    float &prd = *(float*)getPRD<float>();
    // set blue as background color
    prd = 0.0f;
//    We don't need this since we passed the flag OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT to rtTrace
//    optixTerminateRay();
}


// miss for shadows
extern "C" __global__ void __miss__shadow() {

    float &prd = *(float*)getPRD<float>();
    // set blue as background color
    prd = 1.0f;
}


// -----------------------------------------------
// Alpha Transparency Phong rays

// bars are dark grey. No need to complicate in here.
extern "C" __global__ void __closesthit__phong_alphaTrans() {

    float3 &prd = *(float3*)getPRD<float3>();
    prd = make_float3(0.2,0.2,0.2);
}


// any hit to ignore intersections based on alpha transparency
extern "C" __global__ void __anyhit__phong_alphaTrans() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // assume that there is a texture and texCoords
    const float4 tc
    = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
    +         u * sbtData.vertexD.texCoord0[index.y]
    +         v * sbtData.vertexD.texCoord0[index.z];
  
    // fetch texture value
    float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);

    // ignore intersection based on alpha transparency
    if (fromTexture.w < 0.25)
        optixIgnoreIntersection();
}


// miss sets the background color
extern "C" __global__ void __miss__phong_alphaTrans() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.0f, 1.0f);
}



// -----------------------------------------------
// Alpha Transparency Shadow rays

extern "C" __global__ void __closesthit__shadow_alphaTrans() {

}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow_alphaTrans() {

    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // assume that there is a texture and texCoords
    const float4 tc
    = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
    +         u * sbtData.vertexD.texCoord0[index.y]
    +         v * sbtData.vertexD.texCoord0[index.z];
  
    // fetch texture value
    float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);

    // ignore intersection based on alpha transparency
    if (fromTexture.w < 0.25)
        optixIgnoreIntersection();
    else {
        float &prd = *(float*)getPRD<float>();
        prd = 0.0f;
    }
//    We don't need this since we passed the flag OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT to rtTrace
//    optixTerminateRay();
}


// miss for shadows
extern "C" __global__ void __miss__shadow_alphaTrans() {

    float &prd = *(float*)getPRD<float>();
    // set blue as background color
    prd = 1.0f;
}



// -----------------------------------------------
// Glass Phong rays

// bars are dark grey. No need to complicate in here.
extern "C" __global__ void __closesthit__phong_glass() {

    // ray payload
    float3 afterPRD = make_float3(1.0f);
    uint32_t u0, u1;
    packPointer( &afterPRD, u0, u1 );  

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    //(1.f-u-v) * A + u * B + v * C;
    
    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        optixGetWorldRayDirection(),
        0.001f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        PHONG,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        PHONG,             // missSBTIndex 
        u0, u1 );

    float3 &prd = *(float3*)getPRD<float3>();
    prd = make_float3(0.8,0.8,0.8) * afterPRD;
}


// any hit to ignore intersections based on alpha transparency
extern "C" __global__ void __anyhit__phong_glass() {

 }


// miss sets the background color
extern "C" __global__ void __miss__phong_glass() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.0f, 1.0f);
}



// -----------------------------------------------
// Glass Shadow rays

extern "C" __global__ void __closesthit__shadow_glass() {

    // ray payload
    float afterPRD = 1.0f;
    uint32_t u0, u1;
    packPointer( &afterPRD, u0, u1 );  

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    //(1.f-u-v) * A + u * B + v * C;
    
    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        optixGetWorldRayDirection(),
        0.001f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        SHADOW,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        SHADOW,             // missSBTIndex 
        u0, u1 );

    float &prd = *(float*)getPRD<float>();
    prd = 0.8f * afterPRD;
}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow_glass() {

}


// miss for shadows
extern "C" __global__ void __miss__shadow_glass() {

    float &prd = *(float*)getPRD<float>();
    // set blue as background color
    prd = 1.0f;
}







// -----------------------------------------------
// Primary Rays


extern "C" __global__ void __raygen__renderFrame() {

    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;  
    

	if (optixLaunchParams.frame.frame == 0 && ix == 0 && iy == 0) {

		// print info to console
		printf("===========================================\n");
        printf("Nau Ray-Tracing Debug\n");
        const float4 &ld = optixLaunchParams.global->lightDir;
        printf("LightDir: %f, %f %f %f\n", ld.x,ld.y,ld.z,ld.w);
        printf("Launch dim: %u %u\n", optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
		printf("===========================================\n");
	}


    // ray payload
    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    // compute ray direction
    // normalized screen plane position, in [-1, 1]^2
    const float2 screen(make_float2(ix+.5f,iy+.5f)
                    / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
  
    // note: nau already takes into account the field of view and ratio when computing 
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
             OPTIX_RAY_FLAG_NONE,//,OPTIX_RAY_FLAG_DISABLE_ANYHIT
             PHONG,             // SBT offset
             RAY_TYPE_COUNT,               // SBT stride
             PHONG,             // missSBTIndex 
             u0, u1 );

    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*pixelColorPRD.x);
    const int g = int(255.0f*pixelColorPRD.y);
    const int b = int(255.0f*pixelColorPRD.z);

    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);
    // compute index
    const uint32_t fbIndex = ix + iy*optixGetLaunchDimensions().x;
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  

