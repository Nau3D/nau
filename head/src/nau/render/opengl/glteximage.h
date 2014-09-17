#ifndef GLTEXIMAGE_H
#define GLTEXIMAGE_H

#include <nau/render/opengl/gltexture.h>
#include <nau/material/teximage.h>

#include <GL/glew.h>

using namespace nau::render;

namespace nau
{
	namespace render
	{
		class GLTexImage : public nau::material::TexImage
		{
		friend class nau::material::TexImage;

		public:

			void update(void);
			void *getData();

		protected:
			GLTexImage (Texture *t);
			~GLTexImage(void);




		};
	};
};
#endif
