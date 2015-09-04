#ifndef TEXIMAGE_H
#define TEXIMAGE_H

#include "nau/material/iTexture.h"

#include <vector>

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

			static ITexImage *create(ITexture *t);
			/// reloads the image from the texture
			virtual void update(void) = 0;
			virtual const std::string &getTextureName();
			/// returns the image data
			virtual void *getData() = 0;
			virtual unsigned char *getRGBData() = 0;
			virtual int getNumComponents();
			std::string getType();
			virtual int getWidth();
			virtual int getHeight();
			virtual int getDepth();
			~ITexImage();

			static const unsigned int IconSize = 96;

		};
	};
};

#endif //TEXIMAGE_H
