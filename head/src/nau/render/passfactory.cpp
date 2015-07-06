#include "nau/render/passfactory.h"

#include <vector>

#include "nau/render/pass.h"

#include "nau/render/passDepthMap.h"
//#include "nau/render/depthmap4depthtexturespass.h"
//#include "nau/render/depthmaprgba32fpass.h"
//#include "nau/render/fogwithcausticspass.h"
#include "nau/render/passQuad.h"
//#include "nau/render/waterplanefogpass.h"
#include "nau/render/passProfiler.h"
#include "nau/render/passCompute.h"

#ifdef NAU_OPTIX_PRIME 
#include "nau/render/passoptixprime.h"
#endif
#ifdef NAU_OPTIX
#include "nau/render/passOptix.h"
#endif


// DAVE
//#include "nau/render/raytracerpass.h"
//#include "nau/render/shadowmapraytracerpass.h"
//END DAVE
using namespace nau::render;

Pass*
PassFactory::create (const std::string &type, const std::string &name)
{
	if ("default" == type) {
		return new Pass (name);
	}
	if ("depthmap" == type) {
		return new PassDepthMap (name);
	}
	if ("quad" ==  type) {
		return new PassQuad (name);
	}
	if ("profiler" == type) {
		return new PassProfiler(name);
	}
	if ("compute" == type) {
		return new PassCompute(name);
	}
#ifdef NAU_OPTIX
	if ("optix" == type)
		return new PassOptix(name);
#endif
#ifdef NAU_OPTIX_PRIME
#if NAU_OPENGL_VERSION >= 420
	if ("optixPrime" == type)
		return new PassOptixPrime(name);
#endif
#endif

	return 0;
}

bool
PassFactory::isClass(const std::string &name)
{
	if (("default" != name) && 
		("depthmap" != name) && ("quad" != name) && ("profiler" != name) && ("compute" != name)
#ifdef NAU_OPTIX		
		&& ("optix" != name)
#endif
#ifdef NAU_OPTIX_PRIME		
		&& ("optixPrime" != name)
#endif
		)
		return false;
	else
		return true;
}


std::vector<std::string> * 
PassFactory::getClassNames(){

	std::vector<std::string> *names = new std::vector<std::string>; 

	names->push_back("default");
	names->push_back("depthmap");
	names->push_back("quad");
	names->push_back("profiler");
	names->push_back("optix");
	names->push_back("compute");

	return names;
}
