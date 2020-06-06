#include "../launchParamsGlobal.h"
#include  "../util.h"

struct GlobalParams{
    float4 lightPos;
    float4 *accumBuffer;
    int shadowRays;
    float gamma;
    float lightScale;
} ;


struct LaunchParams
{
    Frame frame;
    Camera camera;
    OptixTraversableHandle traversable;

    GlobalParams *global;
};

