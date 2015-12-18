#ifndef MATERIALTEXTURE_H
#define MATERIALTEXTURE_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/material/iTexture.h"
#include "nau/material/iTextureSampler.h"

#include <string>

using namespace nau;

namespace nau {

	namespace material {

		class MaterialTexture : public AttributeValues
		{
		public:

			INT_PROP(UNIT, 0);

			static AttribSet Attribs;

			std::string& getLabel(void) {
				return m_Label;
			};

			MaterialTexture(int unit);
			MaterialTexture();
			MaterialTexture(const MaterialTexture &mt);

			~MaterialTexture(void);

			const MaterialTexture& operator =(const MaterialTexture &mt);

			void bind();
			void unbind();
			void setTexture(ITexture *b);
			//void setSampler(ITextureSampler *b);
			ITexture *getTexture();
			ITextureSampler *getSampler();


			void clone(MaterialTexture &mt);

		protected:

			static bool Init();
			static bool Inited;

			std::string m_Label;
			ITexture *m_Texture;
			ITextureSampler *m_Sampler;
		};
	};
};


#endif 