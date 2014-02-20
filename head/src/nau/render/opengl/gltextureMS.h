#ifndef GLTEXTUREMS_H
#define GLTEXTUREMS_H

#include <nau/render/opengl/glTexture.h>

using namespace nau::render;
using namespace nau::material;

namespace nau
{
	namespace render
	{
		class GLTextureMS : public GLTexture
		{
		friend class Texture;

		public:

			
		protected:
			static bool InitGL();
			static bool Inited;

			GLTextureMS (std::string label, std::string internalFormat,
				int width, int height, int samples );

		public:
			~GLTextureMS(void);
		};
	};
};
#endif
