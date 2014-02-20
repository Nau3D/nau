#include <nau/render/optix/optixRenderer.h>
#include <nau.h>

using namespace nau::render::optixRender;

optix::Context OptixRenderer::p_Context; //= optix::Context::create();
unsigned int OptixRenderer::p_EntryPointCount; //= 1;
std::map<OptixRenderer::ProgramTypes, optix::Program> OptixRenderer::p_Program;
unsigned int OptixRenderer::p_RayTypeCount = OptixRenderer::init();


int
OptixRenderer::init() {

//	p_RayTypeCount = 0;
	p_EntryPointCount = 0;
	try {
		p_Context = optix::Context::create();
//		p_Context->setRayTypeCount(1);
//		p_Context->setEntryPointCount(1);
	}
	catch(optix::Exception& e) {
	
		NAU_THROW("Optix Creating Context Error [%s]", e.getErrorString().c_str());
	}
	return 0;
}


int 
OptixRenderer::getNextAvailableRayType() {

	++p_RayTypeCount;
	p_Context->setRayTypeCount(p_RayTypeCount);
	return (p_RayTypeCount-1);
}


int 
OptixRenderer::getNextAvailableEntryPoint() {

	++p_EntryPointCount;
	p_Context->setEntryPointCount(p_EntryPointCount);
	return(p_EntryPointCount-1);
}




optix::Context &
OptixRenderer::getContext() {

	return p_Context;
}


void
OptixRenderer::setProgram(ProgramTypes aType, int rayType, std::string fileName, std::string proc) {

	try {
		p_Program[aType] = p_Context->createProgramFromPTXFile( fileName, proc);
	}
	catch(optix::Exception& e) {
	
		NAU_THROW("Optix Creating Program Error (file: %s proc: %s) [%s]", fileName.c_str(), proc.c_str(), e.getErrorString().c_str());
	}

	try {
		switch(aType) {
	
			case RAY_GEN: p_Context->setRayGenerationProgram(rayType, p_Program[aType] );
				break;
			case EXCEPTION: p_Context->setExceptionProgram( rayType, p_Program[aType] );
				break;
		}
	}
	catch(optix::Exception& e) {
	
		NAU_THROW("Optix Setting Program Error (file: %s proc: %s) [%s]", fileName.c_str(), proc.c_str(), e.getErrorString().c_str());
	}
}