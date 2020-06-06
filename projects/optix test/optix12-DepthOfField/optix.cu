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

    // get mesh data
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
    const float3 nn = normalize(make_float3(n));

    // intersection position
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();

    // direction towards light
    float3 lPos = make_float3(optixLaunchParams.global->lightPos);
    float3 lDir = normalize(lPos - pos);

    // compute light intensity
    float intensity = max(dot(lDir, nn),0.0f);

    // ray payload
    shadowPRD shadowAttPRD;
    shadowAttPRD.shadowAtt = 1.0f;
    shadowAttPRD.seed = prd.seed;
    uint32_t u0, u1;
    packPointer( &shadowAttPRD, u0, u1 );  
  
    // trace shadow ray
    int squaredShadowRays = 2;
    float shadowTotal = 0.0f;
    for (int i = 0; i < squaredShadowRays; ++i) {
        for (int j = 0; j < squaredShadowRays; ++j) {

            lPos.x = -0.2 + i * 1.0/squaredShadowRays * 0.4f + rnd(prd.seed) * 1.0/squaredShadowRays * 0.4;
            lPos.z = -0.2 + j * 1.0/squaredShadowRays * 0.4f + rnd(prd.seed) * 1.0/squaredShadowRays * 0.4;
            lDir = normalize(lPos - pos);
            optixTrace(optixLaunchParams.traversable,
                pos,
                lDir,
                0.001f,               // tmin
                5.0,               // tmax
                0.0f,               // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
                SHADOW,             // SBT offset
                RAY_TYPE_COUNT,     // SBT stride
                SHADOW,             // missSBTIndex 
                u0, u1 );

                shadowTotal += shadowAttPRD.shadowAtt;
        }
    }
    shadowTotal /= (squaredShadowRays * squaredShadowRays);

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {  
        // get barycentric coordinates
        // compute pixel texture coordinate
        const float4 tc
          = (1.f-u-v) * sbtData.vertexD.texCoord0[index.x]
          +         u * sbtData.vertexD.texCoord0[index.y]
          +         v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        
        float4 fromTexture = tex2D<float4>(sbtData.texture,tc.x,tc.y);
        prd.color = make_float3(fromTexture) * min(intensity * shadowTotal + 0.0, 1.0);
    }
    else
        prd.color = sbtData.diffuse * min(intensity * shadowTotal + 0.0, 1.0);
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
// Light material


extern "C" __global__ void __closesthit__light() {

    colorPRD &prd = *(colorPRD*)getPRD<colorPRD>();
    prd.color = make_float3(1.0f);
}


extern "C" __global__ void __anyhit__light() {
}


extern "C" __global__ void __miss__light() {


}


extern "C" __global__ void __closesthit__light_shadow() {

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

    float3 intersectionPoint = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    float ndotl = max(0.0f, dot(normalize(make_float3(n)), -normalize(intersectionPoint-optixGetWorldRayOrigin())));
    shadowPRD &prd = *(shadowPRD*)getPRD<shadowPRD>();
    prd.shadowAtt = ndotl;
}



extern "C" __global__ void __anyhit__light_shadow() {
}



// -----------------------------------------------
// Metal Phong rays

extern "C" __global__ void __closesthit__phong_metal() {

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
    // ray payload

    float3 normal = normalize(make_float3(n));

    // entering glass
    //if (dot(optixGetWorldRayDirection(), normal) < 0)

    colorPRD &prd = *(colorPRD*)getPRD<colorPRD>();

    colorPRD afterPRD;
    afterPRD.color = make_float3(1.0f);
    afterPRD.seed = prd.seed;
    uint32_t u0, u1;
    packPointer( &afterPRD, u0, u1 );  

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    //(1.f-u-v) * A + u * B + v * C;
    
    float3 rayDir = reflect(optixGetWorldRayDirection(), normal);
    optixTrace(optixLaunchParams.traversable,
        pos,
        rayDir,
        0.04f,    // tmin is high to void self-intersection
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        PHONG,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        PHONG,             // missSBTIndex 
        u0, u1 );

    prd.color = make_float3(0.8,0.8,0.8) * afterPRD.color;
}





// -----------------------------------------------
// Glass Phong rays


extern "C" __global__ void __closesthit__phong_glass() {

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

    float3 normal = normalize(make_float3(n));
    const float3 normRayDir = optixGetWorldRayDirection();

    // new ray direction
    float3 rayDir;
    // entering glass
    float dotP;
    if (dot(normRayDir, normal) < 0) {
        dotP = dot(normRayDir, -normal);
        rayDir = refract(normRayDir, normal, 0.66);
    }
    // exiting glass
    else {
        dotP = 0;
        rayDir = refract(normRayDir, -normal, 1.5);
    }

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    
    // ray payload 
    colorPRD &prd = *(colorPRD*)getPRD<colorPRD>();

    colorPRD refractPRD;
    refractPRD.color = make_float3(0.0f);
    refractPRD.seed = prd.seed;
    uint32_t u0, u1;
    packPointer( &refractPRD, u0, u1 );  
    
    if (length(rayDir) > 0)
        optixTrace(optixLaunchParams.traversable,
            pos,
            rayDir,
            0.00001f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
            PHONG,             // SBT offset
            RAY_TYPE_COUNT,     // SBT stride
            PHONG,             // missSBTIndex 
            u0, u1 );

 
    colorPRD reflectPRD;
    reflectPRD.color = make_float3(0.0f);
    reflectPRD.seed = prd.seed;
    if (dotP > 0) {
        float3 reflectDir = reflect(normRayDir, normal);        
        packPointer( &reflectPRD, u0, u1 );  
        optixTrace(optixLaunchParams.traversable,
            pos,
            reflectDir,
            0.00001f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
            PHONG,             // SBT offset
            RAY_TYPE_COUNT,     // SBT stride
            PHONG,             // missSBTIndex 
            u0, u1 );
        float r0 = (1.5f - 1.0f)/(1.5f + 1.0f);
        r0 = r0*r0 + (1-r0*r0) * pow(1-dotP,5);
        prd.color =  refractPRD.color * (1-r0) + r0 * reflectPRD.color;
    }
    else
        prd.color =  refractPRD.color ;
}



extern "C" __global__ void __anyhit__phong_glass() {

}


// miss sets the background color
extern "C" __global__ void __miss__phong_glass() {

    colorPRD &prd = *(colorPRD*)getPRD<colorPRD>();
    // set blue as background color
    prd.color = make_float3(0.0f, 0.0f, 1.0f);
}



// -----------------------------------------------
// Glass Shadow rays

extern "C" __global__ void __closesthit__shadow_glass() {

    shadowPRD &prd = *(shadowPRD*)getPRD<shadowPRD>();
    // ray payload
    shadowPRD afterPRD;
    afterPRD.shadowAtt = 1.0f;
    afterPRD.seed = prd.seed;
    uint32_t u0, u1;
    packPointer( &afterPRD, u0, u1 );  

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
    
    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        optixGetWorldRayDirection(),
        0.01f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask( 255 ),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        SHADOW,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        SHADOW,             // missSBTIndex 
        u0, u1 );

    prd.shadowAtt = 0.95f * afterPRD.shadowAtt;
}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow_glass() {

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
        const float3 &ld = make_float3(optixLaunchParams.global->lightPos);
        printf("LightPos: %f, %f, %f\n", ld.x,ld.y,ld.z);
        printf("Launch dim: %u %u\n", optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
        printf("Rays per pixel squared: %d \n", optixLaunchParams.frame.raysPerPixel);
		printf("===========================================\n");
	}

    // ray payload
    colorPRD pixelColorPRD;
    pixelColorPRD.color = make_float3(1.f);

    float raysPerPixel = float(optixLaunchParams.frame.raysPerPixel);
    // half pixel

    float aperture = optixLaunchParams.global->aperture;
    float2 delta = make_float2(1.0f/raysPerPixel, 1.0f/raysPerPixel);

    // compute ray direction
    // normalized screen plane position, in [-1, 1]^2
    
    const float2 centerPixel = make_float2(delta.x * 0.5, delta.y * 0.5);
    const float2 centerPixelScreen(make_float2(ix + centerPixel.x, iy + centerPixel.y)
            / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0); 
           

    // first find a point P in the focal plane            

    // point in the image plane            
    float3 pixelOrigin = camera.position
                + (centerPixelScreen.x ) * camera.horizontal
                + (centerPixelScreen.y ) * camera.vertical;

    // ray direction starts from the image plane and goes through the center of the lens
    // need to define the lens distance. Assuming 1.0 to start.
    float lensDistance = optixLaunchParams.global->lensDistance;
    float3 lensCenter = camera.position + camera.direction * ( lensDistance);
    float3 rayDir = normalize(lensCenter - pixelOrigin);

    //define ray origin as being the center of the lens

    float cosAlpha = dot(rayDir, camera.direction);
    float focalDistance = optixLaunchParams.global->focalDistance;
    float3 P = lensCenter + rayDir * focalDistance/cosAlpha;

    float red = 0.0f, blue = 0.0f, green = 0.0f;
    for (int i = 0; i < raysPerPixel; ++i) {
        for (int j = 0; j < raysPerPixel; ++j) {

            uint32_t seed = tea<4>( ix * optixGetLaunchDimensions().x + iy, i*raysPerPixel + j );

            pixelColorPRD.seed = seed;
            uint32_t u0, u1;
            packPointer( &pixelColorPRD, u0, u1 );  

            float u = rnd(seed); float v = rnd(seed);
            float alpha = 2 * 3.14159 * u;
            float r = sqrt(v * aperture);
            float up = r * sin(alpha);
            float left = r * cos(alpha);

            float3 lensSample = lensCenter + left * camera.horizontal + up * camera.vertical;
            float3 dir = normalize(P - lensSample);

            // trace primary ray
            optixTrace(optixLaunchParams.traversable,
                    lensSample, //camera.position,
                    dir,
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
    const uint32_t fbIndex = (optixGetLaunchDimensions().x-ix) + (optixGetLaunchDimensions().y-iy)*optixGetLaunchDimensions().x;
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}
  





extern "C" __global__ void __raygen__renderFrame_1() {

    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto &camera = optixLaunchParams.camera;  
    

	if (optixLaunchParams.frame.frame == 0 && ix == 0 && iy == 0) {

		// print info to console
		printf("===========================================\n");
        printf("Nau Ray-Tracing Debug\n");
        const float3 &ld = make_float3(optixLaunchParams.global->lightPos);
        printf("LightPos: %f, %f, %f\n", ld.x,ld.y,ld.z);
        printf("Launch dim: %u %u\n", optixGetLaunchDimensions().x, optixGetLaunchDimensions().y);
        printf("Rays per pixel squared: %d \n", optixLaunchParams.frame.raysPerPixel);
		printf("===========================================\n");
	}

    // ray payload
    colorPRD pixelColorPRD;
    pixelColorPRD.color = make_float3(1.f);

    float raysPerPixel = float(optixLaunchParams.frame.raysPerPixel);
    // half pixel

    float aperture = optixLaunchParams.global->aperture;
    float2 delta = make_float2(1.0f/raysPerPixel, 1.0f/raysPerPixel);

    // compute ray direction
    // normalized screen plane position, in [-1, 1]^2
    
    const float2 centerPixel = make_float2(delta.x * 0.5, delta.y * 0.5);
    const float2 centerPixelScreen(make_float2(ix + centerPixel.x, iy + centerPixel.y)
            / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0); 
            
    float3 rayDir = normalize(camera.direction
                + (centerPixelScreen.x ) * camera.horizontal
                + (centerPixelScreen.y ) * camera.vertical);

    float3 p = camera.position + rayDir * optixLaunchParams.global->focalDistance;

    float red = 0.0f, blue = 0.0f, green = 0.0f;
    for (int i = 0; i < raysPerPixel; ++i) {
        for (int j = 0; j < raysPerPixel; ++j) {

            uint32_t seed = tea<4>( ix * optixGetLaunchDimensions().x + iy, i*raysPerPixel + j );

            pixelColorPRD.seed = seed;
            uint32_t u0, u1;
            packPointer( &pixelColorPRD, u0, u1 );  

            float u = rnd(seed); float v = rnd(seed);
            float alpha = 2 * 3.14159 * u;
            float r = sqrt(v * aperture);
            float up = r * sin(alpha);
            float left = r * cos(alpha);

            float3 cameraPoint = camera.position + left * camera.horizontal + up * camera.vertical;
            float3 dir = normalize(p - cameraPoint);

            // trace primary ray
            optixTrace(optixLaunchParams.traversable,
                    cameraPoint, //camera.position,
                    dir,
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
  

