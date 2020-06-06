#include "optixParams.h" // our launch params



extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}
//  a single ray type
enum { PHONG=0, SHADOW, RAY_TYPE_COUNT };

struct colorPRD{
    float3 color;
    unsigned int seed;
} ;

struct shadowPRD{
    float shadowAtt;
    unsigned int seed;
} ;




// -------------------------------------------------------
// closest hit computes color based lolely on the triangle normal

extern "C" __global__ void __closesthit__radiance() {

    colorPRD &prd = *(colorPRD *)getPRD<colorPRD>();

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

    // intersection position
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();

    // direction towards light
    float3 lPos = make_float3(optixLaunchParams.global->lightPos);
    float3 lDir = normalize(lPos - pos);
    float3 nn = normalize(make_float3(n));
    float intensity = max(dot(lDir, nn),0.0f);
    
    int numRays = optixLaunchParams.global->aoRays;
    float ambientOcclusion = 0;
    // Ambient Occlusion
    shadowPRD AOPRD;
    AOPRD.shadowAtt = 1.0f;
    AOPRD.seed = prd.seed;
    uint32_t u0, u1;
    packPointer( &AOPRD, u0, u1 );  

    for ( int i = 0; i < numRays; ++i) {

        const float z1 = rnd(prd.seed);
        const float z2 = rnd(prd.seed);

        float3 rayDir;
        cosine_sample_hemisphere( z1, z2, rayDir );
        Onb onb( nn );
        onb.inverse_transform( rayDir );

        optixTrace(optixLaunchParams.traversable,
            pos,
            rayDir,
            0.01f,           // tmin
            optixLaunchParams.global->aoRadius,   //tmax
            0.0f,               // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
            SHADOW,             // SBT offset
            RAY_TYPE_COUNT,     // SBT stride
            SHADOW,             // missSBTIndex 
            u0, u1 );

            ambientOcclusion += AOPRD.shadowAtt;
    }

    prd.color = make_float3(ambientOcclusion / numRays);
}


// any hit to ignore intersections with back facing geometry
extern "C" __global__ void __anyhit__radiance() {

}


// miss sets the background color
extern "C" __global__ void __miss__radiance() {

    colorPRD &prd = *(colorPRD*)getPRD<colorPRD>();
    // set blue as background color
    prd.color = make_float3(0.0f, 0.0f, 1.0f);
}


// -----------------------------------------------
// Shadow rays

extern "C" __global__ void __closesthit__shadow() {

    shadowPRD &prd = *(shadowPRD*)getPRD<shadowPRD>();
    prd.shadowAtt = 0.0f;
}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow() {

}


// miss for shadows
extern "C" __global__ void __miss__shadow() {

    shadowPRD &prd = *(shadowPRD*)getPRD<shadowPRD>();
    prd.shadowAtt = 1.0f;
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
        const float4 &ld = optixLaunchParams.global->lightPos;
        printf("LightPos: %f, %f %f %f\n", ld.x,ld.y,ld.z,ld.w);
        printf("Launch dim: %u %u\n", optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
        printf("Rays per pixel squared: %d \n", optixLaunchParams.frame.raysPerPixel);
		printf("===========================================\n");
	}


    // ray payload
    colorPRD pixelColorPRD;
    pixelColorPRD.color = make_float3(1.f);

    float raysPerPixel = float(optixLaunchParams.frame.raysPerPixel);
    // half pixel
    float2 delta = make_float2(1.0f/raysPerPixel, 1.0f/raysPerPixel);

    // compute ray direction
    // normalized screen plane position, in [-1, 1]^2
  
    float red = 0.0f, blue = 0.0f, green = 0.0f;
    for (int i = 0; i < raysPerPixel; ++i) {
        for (int j = 0; j < raysPerPixel; ++j) {

            uint32_t seed = tea<4>( ix * optixGetLaunchDimensions().x + iy, i*raysPerPixel + j );

            pixelColorPRD.seed = seed;
            uint32_t u0, u1;
            packPointer( &pixelColorPRD, u0, u1 );  
            const float2 subpixel_jitter = make_float2( i * delta.x + delta.x *  rnd( seed ), j * delta.y + delta.y * rnd( seed ) );
            const float2 screen(make_float2(ix + subpixel_jitter.x, iy + subpixel_jitter.y)
                            / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);
        
        // note: nau already takes into account the field of view and ratio when computing 
        // camera horizontal and vertival
            float3 rayDir = normalize(camera.direction
                                + (screen.x ) * camera.horizontal
                                + (screen.y ) * camera.vertical);
            
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

            red += pixelColorPRD.color.x / (raysPerPixel*raysPerPixel);
            green += pixelColorPRD.color.y / (raysPerPixel*raysPerPixel);
            blue += pixelColorPRD.color.z / (raysPerPixel*raysPerPixel);
        }
}

    //convert float (0-1) to int (0-255)
    const int r = int(255.0f*red);
    const int g = int(255.0f*green);
    const int b = int(255.0f*blue);
    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000
      | (r<<0) | (g<<8) | (b<<16);
    // compute index
    const uint32_t fbIndex = ix + iy*optixGetLaunchDimensions().x;
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  

