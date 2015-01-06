#ifndef TEXIMAGE_H
#define TEXIMAGE_H

#include <vector>

#include <nau/render/texture.h>

using namespace nau::render;

namespace nau
{
	namespace material
	{

		class TexImage {

		protected:
			void *m_Data;
			Texture *m_Texture;


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
			TexImage(Texture *t);


		public:

			static TexImage *create(Texture *t);
			/// reloads the image from the texture
			virtual void update(void) = 0;
			virtual const std::string &getTextureName();
			/// returns the image data
			virtual void *getData() = 0;
			virtual int getNumComponents();
			std::string getType();
			virtual int getWidth();
			virtual int getHeight();
			virtual int getDepth();
			~TexImage();

		};
	};
};

#endif //TEXIMAGE_H
