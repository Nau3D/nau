#ifndef GL_ARRAY_OF_TEXTURES
#define GL_ARRAY_OF_TEXTURES

#include "nau/material/iArrayOfTextures.h"
#include "nau/render/opengl/glTexture.h"
#include "nau/render/opengl/glTextureSampler.h"

#include "nau.h"
#include "nau/render/iAPISupport.h"

#include <vector>

using namespace gl;
using namespace nau::material;

namespace nau
{
	namespace render
	{
		class GLArrayOfTextures : public IArrayOfTextures
		{
		friend class IArrayOfTextures;

		public:

			GLArrayOfTextures(const std::string &label);
			~GLArrayOfTextures() ;

			virtual void prepare(unsigned int firstUnit, ITextureSampler *ts);
			virtual void restore(unsigned int firstUnit, ITextureSampler *ts);

			virtual void build();

			virtual void clearTextures();

			virtual void clearTexturesLevel(int l);

			virtual void generateMipmaps();


		protected:
			static bool InitGL();
			static bool Inited;

			std::vector<uint64_t> m_TexturePointers;
			struct TexIntFormats {
				GLenum format;
				GLenum type;
				std::string name;

				TexIntFormats( std::string n, GLenum f, GLenum t) :
					format(f), type(t), name(n) {}
				TexIntFormats() : format(GL_RGBA), type(GL_UNSIGNED_BYTE), name("") {}
			};
			static std::map<GLenum, GLArrayOfTextures::TexIntFormats> TexIntFormat;

		};
	};
};

#endif
