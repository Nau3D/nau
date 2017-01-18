#ifndef MATERIAL_ARRAY_TEXTURE_H
#define MATERIAL_ARRAY_TEXTURE_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/material/iArrayOfTextures.h"

#include <string>

using namespace nau;

namespace nau
{
	namespace material
	{
		class MaterialArrayOfTextures : public AttributeValues
		{
		public:

			INT_PROP(FIRST_UNIT, 0);

			INTARRAY_PROP(SAMPLER_ARRAY, 0);

			static AttribSet Attribs;
			static AttribSet &GetAttribs() { return Attribs; }

			MaterialArrayOfTextures(int unit);
			MaterialArrayOfTextures();
			MaterialArrayOfTextures(const MaterialArrayOfTextures &mt);


			void setPropi(IntProperty p, int value);

			std::string& getLabel(void) {
				return m_Label;
			};

			//NauIntArray getPropiv(IntArrayProperty prop);

			void bind();
			void unbind();

			void setArrayOfTextures(IArrayOfTextures *b);
			IArrayOfTextures *getArrayOfTextures();

			ITextureSampler *getSampler();

			virtual ~MaterialArrayOfTextures(void);

		protected:

			static bool Init();
			static bool Inited;

			std::string m_Label;
			IArrayOfTextures *m_TextureArray;
			ITextureSampler *m_Sampler;
		};
	};
};


#endif // IBUFFER_H