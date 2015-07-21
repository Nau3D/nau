#ifndef GLIMAGETEXTURE_H
#define GLIMAGETEXTURE_H

#include "nau/config.h"

#if NAU_OPENGL_VERSION >=  420

#include "nau/render/imageTexture.h"

#include <GL/glew.h>
#include <string>

using namespace nau::render;


namespace nau
{
	namespace render
	{
		class GLImageTexture : public ImageTexture
		{
			friend class ImageTexture;

		public:

			~GLImageTexture(void);

			// prepare the unit texture for rendering
			virtual void prepare();
			/// restore the sampler state
			virtual void restore();

		protected:
			static bool InitGL();
			static bool Inited;

//			unsigned int m_Unit;

			// For loaded images
			GLImageTexture (std::string label, unsigned int unit, unsigned int texID, unsigned int level=0, unsigned int access=GL_WRITE_ONLY );

			GLImageTexture() {};
		};
	};
};
#endif
#endif