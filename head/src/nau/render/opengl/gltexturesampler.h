#ifndef GLTEXTURESAMPLER_H
#define GLTEXTURESAMPLER_H

#include "nau/material/texturesampler.h"
#include <map>

#include <GL/glew.h>


using namespace nau::material;
using namespace nau::render;

namespace nau
{
	namespace render
	{
		class GLTextureSampler : public nau::material::TextureSampler
		{
		public:
			~GLTextureSampler(void);
			GLTextureSampler(Texture *t);

			void update();

			virtual void setPrope(EnumProperty prop, int value);

			virtual void prepare(unsigned int aUnit, int aDim);
			static void restore(unsigned int aUnit, int aDim);

		protected:
			static bool Inited;
			static bool InitGL();


			GLTextureSampler(void);
		};
	};
};


#endif
