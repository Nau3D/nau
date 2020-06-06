#include <stdint.h>
#include <optix.h>
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
    float3  diffuse;
    float3 specular;
    float3 emission;
    float shininess;
};

struct RayGenData {
  int3 color;
};


struct Frame {
    int frame;
    int subFrame;
    uint32_t *colorBuffer;
    int raysPerPixel;
    int maxDepth;    
};

struct Camera {
    float3 position;
    float3 direction;
    float3 horizontal;
    float3 vertical;
    bool changed;
};