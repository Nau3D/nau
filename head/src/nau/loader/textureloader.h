#ifndef TEXTURELOADER_H
#define TEXTURELOADER_H

#include <string>

#include "nau/render/texture.h"
#include "nau/material/teximage.h"

using namespace nau::material;

namespace nau
{
	namespace loader
	{
		class TextureLoader
		{
		public:
			static TextureLoader* create (void);

			virtual int loadImage (std::string file) = 0;
			virtual unsigned char* getData (void) = 0;
			virtual int getWidth (void) = 0;
			virtual int getHeight (void) = 0;
			virtual std::string getFormat (void) = 0; 
			virtual std::string getType (void) = 0;
			virtual void freeImage (void) = 0;

			virtual void save(TexImage *ti, std::string filename) = 0;

			virtual ~TextureLoader(void) {};

			static const int BITMAP_SIZE = 96;
		};
	};
};

#endif //TEXTURELOADER_H
