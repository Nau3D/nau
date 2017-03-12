#ifndef GL_ARRAY_IMAGE_TEXTURE_H
#define GL_ARRAY_IMAGE_TEXTURE_H

#include "nau/config.h"
#include "nau/material/materialArrayOfImageTextures.h"
#include "nau/render/opengl/glTexture.h"

#include <string>

using namespace nau::render;
using namespace nau::material;


namespace nau
{
	namespace render
	{
		class GLMaterialImageTextureArray : public MaterialArrayOfImageTextures
		{
			friend class MaterialArrayOfImageTextures;

		public:

			~GLMaterialImageTextureArray(void);

			static std::map<GLenum, GLTexture::TexIntFormats> TexIntFormat;
		protected:
			static bool InitGL();
			static bool Inited;


			// For loaded images
			GLMaterialImageTextureArray();

		};
	};
};
#endif
//#endif