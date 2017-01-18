#ifndef TEXTURELOADER_H
#define TEXTURELOADER_H

#include <string>

#include "nau/material/iTexture.h"
#include "nau/material/iTexImage.h"

using namespace nau::material;

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif


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

			nau_API virtual int loadImage (std::string file, bool convertToRGBA = true) = 0;
			nau_API virtual unsigned char* getData (void) = 0;
			nau_API virtual int getWidth (void) = 0;
			nau_API virtual int getHeight (void) = 0;
			nau_API virtual std::string getFormat (void) = 0;
			nau_API virtual std::string getType (void) = 0;
			nau_API virtual void freeImage (void) = 0;

			nau_API virtual void convertToFloatLuminance() = 0;
			nau_API virtual void convertToRGBA() = 0;

			nau_API virtual void save(ITexImage *ti, std::string filename) = 0;
			nau_API virtual void save(int width, int height, unsigned char *data, std::string filename) = 0;

			static nau_API void SaveRaw(ITexture *t, std::string filename);

			static nau_API const int BITMAP_SIZE = 96;

			static nau_API void Save(ITexture *t, FileType ft);
			static nau_API void Save(int width, int height, unsigned char *data, std::string filename = "");

			nau_API virtual ~ITextureLoader(void) {};
		};
	};
};

#endif //TEXTURELOADER_H
