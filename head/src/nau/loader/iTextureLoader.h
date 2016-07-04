#ifndef TEXTURELOADER_H
#define TEXTURELOADER_H

#include <string>

#include "nau/material/iTexture.h"
#include "nau/material/iTexImage.h"

using namespace nau::material;

namespace nau
{
	namespace loader
	{
		class ITextureLoader
		{
		public:

			typedef enum {
				HDR,
				PNG
			} FileType;
			static ITextureLoader* create (void);

			virtual int loadImage (std::string file, bool convertToRGBA = true) = 0;
			virtual unsigned char* getData (void) = 0;
			virtual int getWidth (void) = 0;
			virtual int getHeight (void) = 0;
			virtual std::string getFormat (void) = 0; 
			virtual std::string getType (void) = 0;
			virtual void freeImage (void) = 0;

			virtual void convertToFloatLuminance() = 0;
			virtual void convertToRGBA() = 0;

			virtual void save(ITexImage *ti, std::string filename) = 0;
			virtual void save(int width, int height, unsigned char *data, std::string filename) = 0;

			static void SaveRaw(ITexture *t, std::string filename);

			static const int BITMAP_SIZE = 96;

			static void Save(ITexture *t, FileType ft);
			static void Save(int width, int height, unsigned char *data, std::string filename = "");

			virtual ~ITextureLoader(void) {};
		};
	};
};

#endif //TEXTURELOADER_H
