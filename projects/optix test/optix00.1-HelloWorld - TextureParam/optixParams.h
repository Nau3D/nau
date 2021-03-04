#include "../launchParamsGlobal.h"

struct GlobalParams{
    float4 color;
    cudaTextureObject_t tex;
};

struct LaunchParams
{
    Frame frame;
    Camera camera;
    OptixTraversableHandle traversable;

    GlobalParams *global;
};
