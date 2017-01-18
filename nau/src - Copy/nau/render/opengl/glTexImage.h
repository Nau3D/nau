#ifndef GLTEXIMAGE_H
#define GLTEXIMAGE_H

#include "nau/render/opengl/glTexture.h"
#include "nau/material/iTexImage.h"


using namespace nau::render;

namespace nau
{
	namespace render
	{
		class GLTexImage : public nau::material::ITexImage
		{
		friend class nau::material::ITexImage;

		public:

			void update(void);
			void *getData();

			unsigned char *getRGBData();

		protected:
			GLTexImage (ITexture *t);
			~GLTexImage(void);

		};
	};
};
#endif
