#ifndef DEVILTEXTURELOADER_H
#define DEVILTEXTURELOADER_H

#include "nau/loader/iTextureLoader.h"
#include "nau/material/iTexImage.h"

#include <IL/il.h>

#include <string>

using namespace nau::material;

namespace nau
{
	namespace loader
	{
		class DevILTextureLoader : public ITextureLoader
		{
		friend class ITextureLoader;

		protected:
			DevILTextureLoader (void);
		public:
			~DevILTextureLoader (void);

			int loadImage (std::string file);
			unsigned char* getData (void);
			int getWidth (void);
			int getHeight (void);
			std::string getFormat (void); 
			std::string getType (void);
			void freeImage (void);

			void save(ITexImage *ti, std::string filename);

			void save(int width, int height, char *data, std::string filename);


		private:
			ILuint m_IlId;
			static bool inited;	

			ILuint convertType(std::string texType);
		};
	};
};

#endif //DEVILTEXTURELOADER_H
