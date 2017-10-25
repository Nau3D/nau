#ifndef GLTEXTURE_H
#define GLTEXTURE_H

#include "nau/material/iTexture.h"
#include "nau/material/iTextureSampler.h"

#include <glbinding/gl/gl.h>
using namespace gl;

using namespace nau::material;

namespace nau
{
	namespace render
	{
		class GLTexture : public ITexture
		{
		friend class ITexture;

		public:

			~GLTexture(void);
			GLTexture(){};

			//! prepare a texture for rendering
			virtual void prepare(unsigned int unit, ITextureSampler *ts);
			//! restore default texture in texture unit
			virtual void restore(unsigned int unit, ITextureSampler *ts);
			//! builds a texture with the attribute parameters previously set
			virtual void build(int immutable = 0);

			virtual void clear();
			virtual void clearLevel(int l);

			virtual void generateMipmaps();
			virtual void resize(unsigned int x, unsigned int y, unsigned int z);

			static int GetCompatibleFormat(int dim, int anInternalFormat);
			static int GetCompatibleType(int dim, int anInternalFormat);
			static int GetNumberOfComponents(unsigned int format);
			static int GetElementSize(unsigned int format, unsigned int type);

			static std::map<GLenum, GLenum> TextureBound;
			struct TexIntFormats{
				GLenum format;
				GLenum type;				
				std::string name;

				TexIntFormats(std::string n, GLenum f, GLenum t):
					format(f), type(t), name(n)   {}
				TexIntFormats(): format(GL_RGBA), type(GL_UNSIGNED_BYTE), name("") {}
			};
			static std::map<GLenum, GLTexture::TexIntFormats> TexIntFormat;

		protected:
			static bool InitGL();
			static bool Inited;


			struct TexFormats{
				unsigned int numComp;				
				std::string name;

				TexFormats(std::string n, unsigned int t):
					numComp(t), name(n)   {}
				TexFormats(): numComp(0) , name(""){}
			};
			static std::map<GLenum, TexFormats> TexFormat;

			struct TexDataTypes{
				unsigned int bitDepth;				
				std::string name;

				TexDataTypes(std::string n, unsigned int t):
					bitDepth(t), name(n)  {}
				TexDataTypes(): bitDepth(0), name(""){}
			};
			static std::map<GLenum, TexDataTypes> TexDataType;

			// returns the number of channels in the texture
			virtual int getNumberOfComponents(void);
			virtual  int getElementSize();

			// for loaded images
			GLTexture (std::string label, std::string internalFormat,
				std::string aFormat, std::string aType, int width, int height, int depth,
				void* data, bool mipmap = true );

			// for empty textures
			GLTexture(std::string label, std::string anInternalFormat, int width, int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);
			GLTexture(std::string label, int anInternalFormat, int width, int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);

			// for empty textures with default parameters
			GLTexture(std::string label);

		};
	};
};
#endif
