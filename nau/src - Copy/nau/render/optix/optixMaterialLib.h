#ifndef OPTIXMATERIALLIB_H
#define OPTIXMATERIALLIB_H

#include <map>
#include <string>

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include "nau/material/materialId.h"
#include "nau/material/material.h"

#include "nau/render/optix/optixTextureLib.h"
#include "nau/render/optix/optixMaterial.h"


namespace nau {
  namespace render {
   namespace optixRender {
		

		
	class OptixMaterialLib {

	public:

		enum MaterialPrograms {
			CLOSEST_HIT,
			ANY_HIT,
			MISS
		};

		OptixMaterialLib() {};
		void setContext(optix::Context &aContext);
		void setTextureLib(OptixTextureLib *otl);
		void setMaterialProgram(MaterialPrograms aProgram, int rayType, std::string filename, std::string proc);
		void setMaterialProgram(std::string name, MaterialPrograms aProgram, int rayType, std::string filename, std::string proc);
		void addMaterial(nau::material::MaterialID aMat);
		optix::Material &getMaterial(nau::material::MaterialID aMat);
		void applyMaterial(optix::GeometryInstance, nau::material::MaterialID aMat);
		void applyMissPrograms();
		std::string isDefined(std::string matName);
			
		void addMaterialAttribute(std::string name, nau::material::ProgramValue &p);

	private:
		
		std::map<std::string, optix::Material> m_MaterialLib;
		std::map<std::string, nau::render::optixRender::OptixMaterial> m_MaterialProcLib;
		std::map<int, optix::Program> m_ClosestHitProgram;
		std::map<int, optix::Program> m_AnyHitProgram;
		std::map<int, optix::Program> m_MissProgram;
		std::map<std::string, nau::material::ProgramValue> o_MatAttribute;
		std::string m_Name;
		optix::Context m_Context;
		OptixTextureLib *m_TextureLib;
	};
   };
  };
};


#endif