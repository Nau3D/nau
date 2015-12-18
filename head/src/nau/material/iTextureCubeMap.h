#ifndef TEXTURECUBEMAP_H
#define TEXTURECUBEMAP_H


#include "nau/material/iState.h"
#include "nau/material/iTexture.h"

#include <string>

namespace nau {

	namespace material {
		class ITextureSampler;
	}
}

namespace nau
{
	namespace material
	{
		class ITextureCubeMap: public ITexture
		{

		protected:
			std::vector<std::string> m_Files;


		public:

			typedef enum {
				TEXTURE_CUBE_MAP_POSITIVE_X, 
				TEXTURE_CUBE_MAP_NEGATIVE_X, 
				TEXTURE_CUBE_MAP_POSITIVE_Y, 
				TEXTURE_CUBE_MAP_NEGATIVE_Y, 
				TEXTURE_CUBE_MAP_POSITIVE_Z, 
				TEXTURE_CUBE_MAP_NEGATIVE_Z
			} TextureCubeMapFaces;

			static ITextureCubeMap* Create (std::vector<std::string> files, std::string label, bool mipmap = true); 

		public:
			virtual void setData(std::string internalFormat, std::string aFormat, 
				std::string aType, int width, int height, unsigned char *data=NULL) {};

			virtual std::string& getLabel (void);
			virtual void setLabel (std::string label);

			virtual std::string &getFile (TextureCubeMapFaces i);
			virtual void setFile (std::string file, TextureCubeMapFaces i);

			virtual void prepare(unsigned int unit, nau::material::ITextureSampler *ts) = 0;
			virtual void restore(unsigned int unit, nau::material::ITextureSampler *ts) = 0;

			virtual void clear() = 0;
			virtual void clearLevel(int l) = 0;

			virtual void generateMipmaps() = 0;

			//virtual void enableCompareToTexture (void) = 0;
			//virtual void disableCompareToTexture (void) = 0;

			//virtual void enableObjectSpaceCoordGen (void) {};
			//virtual void generateObjectSpaceCoords (TextureCoord aCoord, float *plane) {}; /***MARK***/ //Maybe this should be plane class


		protected:
			ITextureCubeMap(std::string label, std::vector<std::string> files, 
				std::string internalFormat, 
				std::string aFormat, std::string aType, int width);
			//ITextureCubeMap (std::string label, std::vector<std::string> files);
		public:
			virtual ~ITextureCubeMap(void);
		};
	};
};

#endif
