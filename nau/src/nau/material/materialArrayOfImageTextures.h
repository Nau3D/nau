#ifndef MATERIAL_ARRAY_IMAGE_TEXTURE_H
#define MATERIAL_ARRAY_IMAGE_TEXTURE_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/material/iArrayOfTextures.h"
#include "nau/material/iImageTexture.h"


#include <string>
#include <vector>

using namespace nau;

namespace nau
{
	namespace material
	{
		class MaterialArrayOfImageTextures : public AttributeValues
		{
		public:

			ENUM_PROP(ACCESS, 0);
			ENUM_PROP(INTERNAL_FORMAT, 1);

			UINT_PROP(LEVEL, 0);

			BOOL_PROP(CLEAR, 0);
			INT_PROP(FIRST_UNIT, 0);

			INTARRAY_PROP(SAMPLER_ARRAY, 0);

			static MaterialArrayOfImageTextures *
				MaterialArrayOfImageTextures::Create();

			static AttribSet Attribs;
			static AttribSet &GetAttribs() { return Attribs; }

			void setPropi(IntProperty prop, int value);
			void setPrope(EnumProperty prop, int value);
			void setPropui(UIntProperty prop, unsigned int value);
			void setPropb(BoolProperty prop, bool value);



			MaterialArrayOfImageTextures();
			MaterialArrayOfImageTextures(const MaterialArrayOfImageTextures &mt);


			std::string& getLabel(void) {
				return m_Label;
			};

			void bind();
			void unbind();

			void setArrayOfTextures(IArrayOfTextures *b);
			IArrayOfTextures *getArrayOfTextures();


			virtual ~MaterialArrayOfImageTextures(void);

		protected:

			static bool Init();
			static bool Inited;

			std::string m_Label;
			IArrayOfTextures *m_TextureArray;
			std::vector<IImageTexture *> m_ImageTextures;

		};
	};
};


#endif // IBUFFER_H