
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
};

struct RayGenData {
  int3 color;
};

