#include "../launchParamsGlobal.h"
#include  "../util.h"


struct GlobalParams{
    float4 lightDir;
} ;


struct LaunchParams
{
    Frame frame;
    Camera camera;
    OptixTraversableHandle traversable;

    GlobalParams *global;
};
