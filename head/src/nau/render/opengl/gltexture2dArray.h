#ifndef GLTEXTURE2DARRAY_H
#define GLTEXTURE2DARRAY_H

#include <nau/render/opengl/glTexture.h>

using namespace nau::render;
using namespace nau::material;

namespace nau
{
	namespace render
	{
		class GLTexture2DArray : public GLTexture
		{
		friend class Texture;

		public:

			
		protected:
			static bool InitGL();
			static bool Inited;

			GLTexture2DArray (std::string label, std::string internalFormat,
				int width, int height, int layers );

		public:
			~GLTexture2DArray(void);
		};
	};
};
#endif
