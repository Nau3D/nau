#ifndef GLTEXTURESAMPLER_H
#define GLTEXTURESAMPLER_H

#include <nau/material/texturesampler.h>
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

			virtual void setProp(EnumProperty prop, int value);
			virtual void setProp(Float4Property prop, float x, float y, float z, float w);
			virtual void setProp(Float4Property prop, vec4& value);

			virtual void prepare(unsigned int aUnit, int aDim);
			static void restore(unsigned int aUnit, int aDim);

		protected:
			static bool Inited;
			static bool Init();


			GLTextureSampler(void);
		};
	};
};


#endif
