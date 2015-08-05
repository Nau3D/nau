#ifndef MATERIALTEXTURE_H
#define MATERIALTEXTURE_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/material/texture.h"
#include "nau/material/textureSampler.h"

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
			const MaterialTexture& operator =(const MaterialTexture &mt);

			void bind();
			void unbind();
			void setTexture(Texture *b);
			void setSampler(TextureSampler *b);
			Texture *getTexture();
			TextureSampler *getSampler();

			~MaterialTexture(void) {};

			void clone(MaterialTexture &mt);

		protected:

			static bool Init();
			static bool Inited;

			std::string m_Label;
			Texture *m_Texture;
			TextureSampler *m_Sampler;
		};
	};
};


#endif 