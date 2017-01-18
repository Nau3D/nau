#ifndef ARRAY_TEXTURE_H
#define ARRAY_TEXTURE_H

#include "nau/attributeValues.h"
#include "nau/material/iBuffer.h"
#include "nau/material/iTextureSampler.h"

#include <string>

namespace nau {
	namespace material {

		class IArrayOfTextures : public AttributeValues
		{
		public:
			ENUM_PROP(DIMENSION, 0);
			ENUM_PROP(FORMAT, 1);
			ENUM_PROP(TYPE, 2);
			ENUM_PROP(INTERNAL_FORMAT, 3);

			INT_PROP(WIDTH, 1);
			INT_PROP(HEIGHT, 2);
			INT_PROP(DEPTH, 3);
			INT_PROP(LEVELS, 4);
			INT_PROP(SAMPLES, 5);
			INT_PROP(LAYERS, 6);
			INT_PROP(COMPONENT_COUNT, 7);
			INT_PROP(ELEMENT_SIZE, 8);
			INT_PROP(BUFFER_ID, 9);

			UINT_PROP(TEXTURE_COUNT, 0);

			BOOL_PROP(MIPMAP, 0);
			BOOL_PROP(CREATE_BUFFER, 1);

			INTARRAY_PROP(TEXTURE_ID_ARRAY, 0);

			static AttribSet Attribs;

			static IArrayOfTextures *Create(const std::string &label);

			virtual const std::string& getLabel(void);
			virtual void setLabel(const std::string &label);
			//! prepare buffer with texture pointers for rendering
			virtual void prepare(unsigned int firstUnit, ITextureSampler *ts) = 0;
			//! restore 
			virtual void restore(unsigned int firstUnit, ITextureSampler *ts) = 0;
			//! builds array of textures with the attribute parameters previously set
			//! creates uniform buffer and fills it with texture pointers
			virtual void build() = 0;

			virtual void clearTextures() = 0;
			virtual void clearTexturesLevel(int l) = 0;

			virtual void generateMipmaps() = 0;

			ITexture *getTexture(unsigned int i);

		protected:

			IArrayOfTextures(const std::string &label);

			static bool Init();
			static bool Inited;

			std::string m_Label;

			IBuffer *m_Buffer;
			std::vector<ITexture *> m_Textures;
		};
	};
};

#endif 