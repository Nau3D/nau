#ifndef GLTEXTURESAMPLER_H
#define GLTEXTURESAMPLER_H

#include "nau/material/texturesampler.h"

#include <map>

using namespace nau::material;

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
			virtual void restore(unsigned int aUnit, int aDim);

		protected:
			static bool InitedGL;
			static bool InitGL();


			GLTextureSampler(void);
		};
	};
};


#endif
