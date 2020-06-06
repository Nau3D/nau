#include "../launchParamsGlobal.h"
#include  "../util.h"

struct GlobalParams{
    float4 lightPos;
    float focalDistance;
    float aperture;
    float lensDistance;
} ;

struct LaunchParams
{
    Frame frame;
    Camera camera;
    OptixTraversableHandle traversable;

    GlobalParams *global;
};
