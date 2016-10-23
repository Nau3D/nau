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

		static void SetProgram(ProgramTypes aType, int rayType, 
						std::string filename, std::string proc);
		static optix::Context &GetContext();
		static int GetNextAvailableRayType();
		static int GetNextAvailableEntryPoint();
		static void Terminate();
		static int Init();

	private:
		OptixRenderer();

		static optix::Context s_Context;
		static unsigned int s_RayTypeCount;
		static unsigned int s_EntryPointCount;
		static std::map<ProgramTypes, optix::Program>  s_Program;

	};
   };
  };
};


#endif