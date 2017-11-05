#include "nau/render/iAPISupport.h"

#include "nau/config.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glAPISupport.h"
#endif


using namespace nau::render;

IAPISupport *IAPISupport::Instance = NULL;


IAPISupport *
IAPISupport::GetInstance() {

	if (Instance == NULL) {
#ifdef NAU_OPENGL
		Instance = new GLAPISupport();
#endif
	}

	return Instance;
}


bool 
IAPISupport::apiSupport(APIFeatureSupport feature) {

	//if (Instance == )
	return m_APISupport[feature];
}


unsigned int
IAPISupport::getVersion() {

	return m_Version;
}

IAPISupport::IAPISupport() {

}

IAPISupport::~IAPISupport() {

	m_APISupport.clear();
}
