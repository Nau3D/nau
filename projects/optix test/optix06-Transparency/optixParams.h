#include "../launchParamsGlobal.h"
#include  "../util.h"

struct GlobalParams{
    float4 lightDir;
	cudaTextureObject_t texture;
} ;

struct LaunchParams
{
    Frame frame;
    Camera camera;
    OptixTraversableHandle traversable;

    GlobalParams *global;
};
