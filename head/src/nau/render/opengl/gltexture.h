#ifndef GLTEXTURE_H
#define GLTEXTURE_H

#include <nau/render/texture.h>
//#include <nau/material/textureSampler.h>
#include <nau/render/opengl/gltexturesampler.h>

#include <GL/glew.h>

using namespace nau::render;
using namespace nau::material;

namespace nau
{
	namespace render
	{
		class GLTexture : public Texture
		{
		friend class Texture;

		public:

			~GLTexture(void);

			// prepare the unit texture for rendering
			virtual void prepare(int unit, TextureSampler *ts);
			/// restore the sampler state
			virtual void restore(int unit);

			//void enableCompareToTexture (void);
			//void disableCompareToTexture (void);
			//void enableObjectSpaceCoordGen (void);
			//void generateObjectSpaceCoords (TextureCoord aCoord, float *plane);

		protected:
			static bool InitGL();
			static bool Inited;

//#if NAU_OPENGL_VERSION < 420 || NAU_OPTIX
			// for empty textures, given an internal format, a format and type are required
			// by GL
			static int GetCompatibleFormat(int anInternalFormat);
			static int GetCompatibleType(int aFormat);
//#endif
			// returns the number of channels in the texture
			virtual int getNumberOfComponents(void);
			virtual  int getElementSize();

			// For loaded images
			GLTexture (std::string label, std::string internalFormat,
				std::string aFormat, std::string aType, int width, int height, 
				void* data, bool mipmap = true );

			// For Texture Storage
			GLTexture(std::string label, std::string anInternalFormat, int width, int height, int levels = 1);

			GLTexture() {};

			// For ...
			//GLTexture (std::string label);

			//void setData(std::string internalFormat, std::string aFormat, 
			//	std::string aType, int width, int height, unsigned char * data = NULL);

			//GLenum translateCoord (TextureCoord aCoord);
		};
	};
};
#endif
