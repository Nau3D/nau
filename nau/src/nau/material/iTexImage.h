#ifndef TEXIMAGE_H
#define TEXIMAGE_H

#include "nau/material/iTexture.h"

#include <vector>

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
	namespace material
	{

		class ITexImage {

		protected:
			void *m_Data;
			int m_DataSize;
			ITexture *m_Texture;


			//typedef enum {
			//	FLOAT,
			//	INT,
			//	SHORT,
			//	BYTE,
			//	UNSIGNED_CHAR,
			//	UNSIGNED_INT,
			//	UNSIGNED_SHORT,
			//	UNISGNED_BYTE} dataTypes;

			int  m_DataType;
			unsigned int m_NumComponents;
			unsigned int m_Height, m_Width, m_Depth;
			ITexImage(ITexture *t);


		public:

			static nau_API ITexImage *create(ITexture *t);
			/// reloads the image from the texture
			nau_API virtual void update(void) = 0;
			nau_API virtual const std::string &getTextureName();
			/// returns the image data
			nau_API virtual void *getData() = 0;
			nau_API virtual unsigned char *getRGBData() = 0;
			nau_API virtual int getNumComponents();
			nau_API std::string getType();
			nau_API virtual int getWidth();
			nau_API virtual int getHeight();
			nau_API virtual int getDepth();
			nau_API virtual ~ITexImage();

			static const unsigned int IconSize = 96;

		};
	};
};

#endif //TEXIMAGE_H
