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
		class IMaterialArrayOfTextures : public AttributeValues
		{
		public:

			INT_PROP(BINDING_POINT, 0);

			static AttribSet Attribs;

			static IMaterialArrayOfTextures* Create(IArrayOfTextures *);

			std::string& getLabel(void) {
				return m_Label;
			};

			virtual void bind() = 0;
			virtual void unbind() = 0;
			void setTextureArray(IArrayOfTextures *b);
			IArrayOfTextures *getTextureArray();

			virtual ~IMaterialArrayOfTextures(void) {};

		protected:

			IMaterialArrayOfTextures() { registerAndInitArrays(Attribs); };

			static bool Init();
			static bool Inited;

			std::string m_Label;
			IArrayOfTextures *m_TextureArray;
		};
	};
};


#endif // IBUFFER_H