#include "nau/render/optix/optixMaterial.h"
#include "nau/render/optix/optixRenderer.h"
#include "nau/render/optix/optixMaterialLib.h"
#include "nau.h"

using namespace nau::render::optixRender;


OptixMaterial::OptixMaterial() {

		m_Context = OptixRenderer::getContext();
}


void 
OptixMaterial::setMaterialProgram(unsigned int  aProgram, int rayType, std::string filename, std::string proc) {

	try {
		switch (aProgram) {
		case OptixMaterialLib::CLOSEST_HIT:
				m_ClosestHitProgram[rayType] = m_Context->createProgramFromPTXFile(filename, proc);
				break;
			case OptixMaterialLib::ANY_HIT:
				m_AnyHitProgram[rayType] = m_Context->createProgramFromPTXFile(filename, proc);
				break;
		}
	}
	catch(optix::Exception& e) {
		NAU_THROW("Optix Creating Material Program Error (%s - %s) [%s]", filename.c_str(), proc.c_str(), e.getErrorString().c_str());
	}
}


void 
OptixMaterial::applyMaterialPrograms(optix::Material omat) {

	std::map<int, optix::Program>::iterator iter;

	for (iter = m_ClosestHitProgram.begin(); iter != m_ClosestHitProgram.end(); ++iter) {
		omat->setClosestHitProgram(iter->first, iter->second);
	}
	for (iter = m_AnyHitProgram.begin(); iter != m_AnyHitProgram.end(); ++iter) {
		omat->setAnyHitProgram(iter->first, iter->second);
	}

}
