#ifndef GL_IMAGE_TEXTURE_H
#define GL_IMAGE_TEXTURE_H

#include "nau/config.h"

//#if NAU_OPENGL_VERSION >=  420

#include "nau/material/iImageTexture.h"
#include "nau/render/opengl/glTexture.h"



#include <string>

using namespace nau::render;



namespace nau
{
	namespace render
	{
		class GLImageTexture : public IImageTexture
		{
			friend class IImageTexture;

		public:

			~GLImageTexture(void);

			// prepare the unit texture for rendering
			virtual void prepare();
			/// restore the sampler state
			virtual void restore();

			static std::map<GLenum, GLTexture::TexIntFormats> TexIntFormat;
		protected:
			static bool InitGL();
			static bool Inited;

//			unsigned int m_Unit;

			// For loaded images
			GLImageTexture (std::string label, unsigned int unit, unsigned int texID, unsigned int level=0, unsigned int access=(int)GL_WRITE_ONLY );

			GLImageTexture() {};
		};
	};
};
#endif
//#endif