#ifndef GLTEXTURE_CUBEMAP_H
#define GLTEXTURE_CUBEMAP_H

#include <nau/render/textureCubeMap.h>
#include <nau/scene/camera.h>

#include <nau.h>

#include <GL/glew.h>

using namespace nau::render;

namespace nau
{
	namespace render
	{
		class GLTextureCubeMap : public TextureCubeMap
		{
		friend class TextureCubeMap;

		public:

			~GLTextureCubeMap(void);

			virtual void prepare(int unit, nau::material::TextureSampler *ts);
			virtual void restore(int unit);

			//void enableCompareToTexture (void);
			//void disableCompareToTexture (void);
			//void enableObjectSpaceCoordGen (void);
			//void generateObjectSpaceCoords (TextureCoord aCoord, float *plane);


		protected:
			GLTextureCubeMap (std::string label, std::vector<std::string> files, 
				std::string internalFormat,
				std::string aFormat, std::string aType, int width, unsigned char** data, bool mipmap = true );
			virtual int getNumberOfComponents(void);
			virtual int getElementSize(){return 0;};
		private:

			int getIndex(std::string StringArray[], int IntArray[], std::string aString);

			static int faces[6];

			//GLenum translateCoord (TextureCoord aCoord);
		};
	};
};
#endif
