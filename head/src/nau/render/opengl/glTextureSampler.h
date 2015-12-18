#ifndef GLTEXTURESAMPLER_H
#define GLTEXTURESAMPLER_H

#include "nau/material/iTextureSampler.h"

#include <map>

using namespace nau::material;

namespace nau
{
	namespace render
	{
		class GLTextureSampler : public nau::material::ITextureSampler
		{
		public:
			~GLTextureSampler(void);
			GLTextureSampler(ITexture *t);

			void update();

			virtual void setPrope(EnumProperty prop, int value);
			virtual void setPropf4(Float4Property prop,  vec4 &value);

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
