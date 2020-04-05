
#include <stdint.h>
#include "vec_math.h"

struct vertexData {
		float4* position;
		float4* normal;
		float4* texCoord0;
		float4* tangent;
		float4* bitangent;
};

struct TriangleMeshSBTData {
    uint3 *index;
    vertexData vertexD;
	int hasTexture;
	cudaTextureObject_t texture;
    float3  color;
};

struct RayGenData {
  int3 color;
};

struct GlobalParams{
    float4 lightDir;
	cudaTextureObject_t texture;
} ;


struct LaunchParams
{
    struct {
        int frame;
        uint32_t *colorBuffer;
        int raysPerPixel;
    } frame;
  
    struct {
        float3 position;
        float3 direction;
        float3 horizontal;
        float3 vertical;
    } camera;

    OptixTraversableHandle traversable;

    GlobalParams *global;
};


// pack and unpack payload pointer from
// Ingo Wald Optix 7 course
// https://gitlab.com/ingowald/optix7course

static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1 ) {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
}

static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 ) {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD() { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}
