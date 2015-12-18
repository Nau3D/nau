#ifndef OPTIXTEXTURELIB_H
#define OPTIXTEXTURELIB_H

#include <map>
#include <string>

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include "nau/material/material.h"

namespace nau {
  namespace render {
   namespace optixRender {
		
		
	class OptixTextureLib {

	public:

		OptixTextureLib() {};
		void setContext(optix::Context &aContext);
		unsigned int addTextures(std::shared_ptr<nau::material::Material> &m);
		void addTexture(unsigned int glID);
		optix::TextureSampler &getTexture(unsigned int GLID);

		void applyTextures(optix::GeometryInstance, std::shared_ptr<nau::material::Material> &m);

	private:
		
		std::map<unsigned int, optix::TextureSampler> m_TextureLib;
		optix::Context m_Context;

		int translateWrapModeToOptix(int mode);
		int translateFilterModeToOptix(int mode);
	};
   };
  };
};


#endif