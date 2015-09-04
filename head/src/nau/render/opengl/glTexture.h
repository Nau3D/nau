#ifndef GLTEXTURE_H
#define GLTEXTURE_H

#include "nau/material/iTexture.h"
#include "nau/material/iTextureSampler.h"

#include <GL/glew.h>


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

			static std::map<int, int> GLTexture::TextureBound;
			struct TexIntFormats{
				unsigned int format;
				unsigned int type;				
				char name[32];

				TexIntFormats(char *n, int f, unsigned int t):
					format(f), type(t)   {memcpy(name,n,32);}
				TexIntFormats(): format(0), type(0) {name[0]='\0';}
			};
			static std::map<unsigned int, GLTexture::TexIntFormats> TexIntFormat;

		protected:
			static bool InitGL();
			static bool Inited;


			struct TexFormats{
				unsigned int numComp;				
				char name[32];

				TexFormats(char *n, unsigned int t):
					numComp(t)   {memcpy(name,n,32);}
				TexFormats(): numComp(0) {name[0]='\0';}
			};
			static std::map<unsigned int, TexFormats> TexFormat;

			struct TexDataTypes{
				unsigned int bitDepth;				
				char name[32];

				TexDataTypes(char *n, unsigned int t):
					bitDepth(t)   {memcpy(name,n,32);}
				TexDataTypes(): bitDepth(0) {name[0]='\0';}
			};
			static std::map<unsigned int, TexDataTypes> TexDataType;

			// returns the number of channels in the texture
			virtual int getNumberOfComponents(void);
			virtual  int getElementSize();

			// for loaded images
			GLTexture (std::string label, std::string internalFormat,
				std::string aFormat, std::string aType, int width, int height, 
				void* data, bool mipmap = true );

			// for empty textures
			GLTexture(std::string label, std::string anInternalFormat, int width, int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);

			// for empty textures with default parameters
			GLTexture(std::string label);

		};
	};
};
#endif
