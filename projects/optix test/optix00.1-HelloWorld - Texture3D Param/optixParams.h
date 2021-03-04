#include "../launchParamsGlobal.h"

struct GlobalParams{
    cudaTextureObject_t tex;
    float slice;
};

struct LaunchParams
{
    Frame frame;
    Camera camera;
    OptixTraversableHandle traversable;

    GlobalParams *global;
};
