#include "optixParams.h" // our launch params



extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}
//  a single ray type
enum { PHONG=0, SHADOW, RAY_TYPE_COUNT };


// -------------------------------------------------------
// closest hit computes color based only on the triangle normal

extern "C" __global__ void __closesthit__phong() {


    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();  

    const int   primID = optixGetPrimitiveIndex();
    const uint3 index  = sbtData.index[primID];

    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float4 n
        = (1.f-u-v) * sbtData.vertexD.normal[index.x]
        +         u * sbtData.vertexD.normal[index.y]
        +         v * sbtData.vertexD.normal[index.z];

        const float3 &A     = make_float3(sbtData.vertexD.position[index.x]);
        const float3 &B     = make_float3(sbtData.vertexD.position[index.y]);
        const float3 &C     = make_float3(sbtData.vertexD.position[index.z]);
    
    // direction towards light
    float3 lDir = normalize(make_float3(-optixLaunchParams.global->lightDir));
    float3 nn = normalize(make_float3(n));
    const float3 rayDir = optixGetWorldRayDirection();

    float intensity = max(dot(lDir, nn),0.0f);

    // ray payload
    float shadowAttPRD = 1.0f;
    uint32_t u0, u1;
    packPointer( &shadowAttPRD, u0, u1 );  

    float3 pos = (1.f-u-v) * A + u * B + v * C;
    // trace shadow ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        lDir,
        0.1f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 
        SHADOW,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        SHADOW,             // missSBTIndex 
        u0, u1 );


    float3 &prd = *(float3*)getPRD<float3>();

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {  
        // get barycentric coordinates
        // compute pixel texture coordinate
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        prd = make_float3(fromTexture) * (intensity * shadowAttPRD + 0.3);
    }
    else
        prd = sbtData.diffuse * (intensity * shadowAttPRD + 0.3);
}


// any hit to ignore intersections with back facing geometry
extern "C" __global__ void __anyhit__phong() {
}


// miss sets the bacgground color
extern "C" __global__ void __miss__phong() {

    float3 &prd = *(float3*)getPRD<float3>();
    // set blue as background color
    prd = make_float3(0.0f, 0.0f, 1.0f);
}


// -----------------------------------------------
// Shadow rays

// nothing to do in here
extern "C" __global__ void __closesthit__shadow() {

}


// any hit for shadows: object is in shadow
extern "C" __global__ void __anyhit__shadow() {

    float &prd = *(float*)getPRD<float>(); 
    prd = 0.0f;

//  We don't to explicitly terminate the ray 
//  since we passed the flag OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT to rtTrace
//  optixTerminateRay();
}


// miss for shadows: object is lit
extern "C" __global__ void __miss__shadow() {

    float &prd = *(float*)getPRD<float>();
    prd = 1.0f;
}

// -----------------------------------------------
// Primary Rays


extern "C" __global__ void __raygen__renderFrame() {

    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;  
    
    // compute ray direction
    // normalized screen plane position, in [-1, 1]^2
    const float2 screen(make_float2(ix+.5f,iy+.5f)
                    / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
  
    // note: nau already takes into account the field of view and ratio when computing 
    // camera horizontal and vertival
    float3 rayDir = normalize(camera.direction
                           + screen.x  * camera.horizontal
                           + screen.y * camera.vertical);
    
    // ray payload
    float3 pixelColorPRD = make_float3(1.f);
    uint32_t u0, u1;
    packPointer( &pixelColorPRD, u0, u1 );  

    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
             camera.position,
             rayDir,
             0.01f,    // tmin
             1e20f,  // tmax
             0.0f,   // rayTime
             OptixVisibilityMask( 255 ),
             OPTIX_RAY_FLAG_NONE,// any hit will be used to discard backface intersections
             PHONG,             // SBT offset
             RAY_TYPE_COUNT,               // SBT stride
             PHONG,             // missSBTIndex 
             u0, u1 );

    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*pixelColorPRD.x);
    const int g = int(255.0f*pixelColorPRD.y);
    const int b = int(255.0f*pixelColorPRD.z);

    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000 | (r<<0) | (g<<8) | (b<<16);
    // compute index
    const uint32_t fbIndex = ix + iy*optixGetLaunchDimensions().x;
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  

