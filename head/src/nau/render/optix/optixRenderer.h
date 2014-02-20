#ifndef OPTIXRENDERER_H
#define OPTIXRENDERER_H

#include <map>
#include <string>

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

namespace nau {
  namespace render {
   namespace optixRender {
		
		
	class OptixRenderer {

	public:

		enum ProgramTypes{
			RAY_GEN, 
			EXCEPTION, 
		} ;

		static void setProgram(ProgramTypes aType, int rayType, 
						std::string fileName, std::string proc);
		static optix::Context &getContext();
		static int getNextAvailableRayType();
		static int getNextAvailableEntryPoint();

	private:
		OptixRenderer();

		static int init();
		static optix::Context p_Context;
		static unsigned int p_RayTypeCount;
		static unsigned int p_EntryPointCount;
		static std::map<ProgramTypes, optix::Program>  p_Program;

	};
   };
  };
};


#endif