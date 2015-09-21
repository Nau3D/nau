#include "nau/render/passFactory.h"

#include "nau/errors.h"
#include "nau/render/iAPISupport.h"
#include "nau/render/pass.h"
#include "nau/render/passDepthMap.h"
////#include "nau/render/depthmap4depthtexturespass.h"
////#include "nau/render/depthmaprgba32fpass.h"
////#include "nau/render/fogwithcausticspass.h"
//#include "nau/render/passQuad.h"
#include "nau/render/passProfiler.h"

//#include "nau/render/passCompute.h"
//
//#ifdef NAU_OPTIX 
//#include "nau/render/passOptixPrime.h"
//#endif
//#ifdef NAU_OPTIX
//#include "nau/render/passOptix.h"
//#endif


using namespace nau::render;


PassFactory *PassFactory::Instance = NULL;

PassFactory*
PassFactory::GetInstance (void) {

	if (0 == Instance) {
		Instance = new PassFactory();
	}

	return Instance;
}


PassFactory::PassFactory() {

}


void 
PassFactory::registerClass(const std::string &type, Pass *(*cb)(const std::string &)) {

	m_Creator[type] = cb;
}


Pass*
PassFactory::create (const std::string &type, const std::string &name) {

	IAPISupport *sup = IAPISupport::GetInstance();
	if (type == "compute" && !sup->apiSupport(IAPISupport::COMPUTE_SHADER))
		NAU_THROW("Compute Shader is not supported");

	if (m_Creator.count(type))
		return (*(Pass *(*)(const std::string &))(m_Creator[type]))(name);

//	if ("default" == type) {
//		return new Pass (name);
//	}
	if ("depthmap2" == type) {
		return new PassDepthMap (name);
	}
//	if ("quad" ==  type) {
//		return new PassQuad (name);
//	}
	if ("profiler" == type) {
		return new PassProfiler(name);
	}
	return NULL;
//	if ("compute" == type) {
//		return new PassCompute(name);
//	}
//#ifdef NAU_OPTIX
//	if ("optix" == type)
//		return new PassOptix(name);
//#endif
//#ifdef NAU_OPTIX
//#if NAU_OPENGL_VERSION >= 420
//	if ("optixPrime" == type)
//		return new PassOptixPrime(name);
//#endif
//#endif

	//return 0;
}

bool
PassFactory::isClass(const std::string &name) {

	IAPISupport *sup = IAPISupport::GetInstance();
	if (name == "compute" && !sup->apiSupport(IAPISupport::COMPUTE_SHADER))
		return false;

	if (m_Creator.count(name))
		return true;
	else
		return false;
}


std::vector<std::string> * 
PassFactory::getClassNames() {

	IAPISupport *sup = IAPISupport::GetInstance();
	std::vector<std::string> *names = new std::vector<std::string>; 

	for (auto s : m_Creator) {
		if (s.first == "compute" || sup->apiSupport(IAPISupport::COMPUTE_SHADER))
			names->push_back(s.first);
	}

	return names;
}
