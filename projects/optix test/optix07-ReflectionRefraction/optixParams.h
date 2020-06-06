#include "../launchParamsGlobal.h"
#include  "../util.h"


struct GlobalParams{
    float4 lightPos;
} ;

struct LaunchParams
{
    Frame frame;
    Camera camera;
    OptixTraversableHandle traversable;

    GlobalParams *global;
};
