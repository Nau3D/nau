#ifndef VTKTEXTURELOADER_H
#define VTKTEXTURELOADER_H

#include "nau/enums.h"
#include "nau/loader/iTextureLoader.h"
#include "nau/material/iTexImage.h"

#include <string>

using namespace nau::material;

namespace nau
{
	namespace loader
	{
		class VTKTextureLoader : public ITextureLoader
		{
		friend class ITextureLoader;

		protected:
			VTKTextureLoader(const std::string &filename);
		public:
			~VTKTextureLoader(void);

			int loadImage (bool notUsed);
			unsigned char* getData (void);
			int getWidth (void);
			int getHeight (void);
			int getDepth(void);
			std::string getFormat (void); 
			std::string getSizedFormat(void);
			std::string getType (void);
			void freeImage (void);

			void convertToFloatLuminance();
			void convertToRGBA();
			void save(ITexImage *ti, std::string filename);
			void save(int width, int height, unsigned char *data, std::string filename);

		private:
			int m_Width, m_Depth, m_Height;

			enum { ASCII, BINARY };
			int m_Mode; // ascii or binary
			int m_NumPoints;
			Enums::DataType m_DataType;
			unsigned char *m_Data;

		};
	};
};

#endif //VTKTEXTURELOADER_H
