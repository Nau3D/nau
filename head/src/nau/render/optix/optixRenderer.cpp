#include "nau/render/optix/optixRenderer.h"
#include "nau.h"

using namespace nau::render::optixRender;

optix::Context OptixRenderer::s_Context; //= optix::Context::create();
unsigned int OptixRenderer::s_EntryPointCount; //= 1;
std::map<OptixRenderer::ProgramTypes, optix::Program> OptixRenderer::s_Program;
unsigned int OptixRenderer::s_RayTypeCount = OptixRenderer::Init();


int
OptixRenderer::Init() {

//	s_RayTypeCount = 0;
	s_EntryPointCount = 0;
	try {
		s_Context = optix::Context::create();
//		s_Context->setRayTypeCount(1);
//		s_Context->setEntryPointCount(1);
	}
	catch(optix::Exception& e) {
	
		NAU_THROW("Optix Creating Context Error [%s]", e.getErrorString().c_str());
	}
	return 0;
}


int 
OptixRenderer::GetNextAvailableRayType() {

	++s_RayTypeCount;
	s_Context->setRayTypeCount(s_RayTypeCount);
	return (s_RayTypeCount-1);
}


int 
OptixRenderer::GetNextAvailableEntryPoint() {

	++s_EntryPointCount;
	s_Context->setEntryPointCount(s_EntryPointCount);
	return(s_EntryPointCount-1);
}




optix::Context &
OptixRenderer::GetContext() {

	return s_Context;
}


void
OptixRenderer::SetProgram(ProgramTypes aType, int rayType, std::string fileName, std::string proc) {

	try {
		s_Program[aType] = s_Context->createProgramFromPTXFile( fileName, proc);
	}
	catch(optix::Exception& e) {
	
		NAU_THROW("Optix Creating Program Error (file: %s proc: %s) [%s]", fileName.c_str(), proc.c_str(), e.getErrorString().c_str());
	}

	try {
		switch(aType) {
	
			case RAY_GEN: s_Context->setRayGenerationProgram(rayType, s_Program[aType] );
				break;
			case EXCEPTION: s_Context->setExceptionProgram( rayType, s_Program[aType] );
				break;
		}
	}
	catch(optix::Exception& e) {
	
		NAU_THROW("Optix Setting Program Error (file: %s proc: %s) [%s]", fileName.c_str(), proc.c_str(), e.getErrorString().c_str());
	}
}