#include "../launchParamsGlobal.h"
#include  "../util.h"

struct GlobalParams{
};

struct LaunchParams
{
    Frame frame;
    Camera camera;
    OptixTraversableHandle traversable;

    GlobalParams *global;
};
