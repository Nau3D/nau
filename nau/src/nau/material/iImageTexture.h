#include "nau/config.h"

 //#if NAU_OPENGL_VERSION >= 420

#ifndef IMAGE_TEXTURE_H
#define IMAGE_TEXTURE_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <string>



using namespace nau;

namespace nau
{
	namespace material
	{
		class IImageTexture: public AttributeValues
		{
		public:

			ENUM_PROP(ACCESS, 0);
			ENUM_PROP(INTERNAL_FORMAT, 1);

			UINT_PROP(LEVEL, 0);
			UINT_PROP(TEX_ID, 1);

			BOOL_PROP(CLEAR, 0);

			INT_PROP(UNIT, 0);
			
			static AttribSet Attribs;
			static nau_API AttribSet &GetAttribs();

			static IImageTexture* Create(std::string label, unsigned int unit, unsigned int texID, unsigned int level, unsigned int access);
			static IImageTexture* Create(std::string label, unsigned int unit, unsigned int texID);

			virtual void prepare() = 0;
			virtual void restore() = 0;
		
			virtual ~IImageTexture(void) {};

			virtual std::string& getLabel (void);
			virtual void setLabel (std::string label);

		protected:
			IImageTexture();

			static bool Init();
			static bool Inited;

			std::string m_Label;
			unsigned int m_Format, m_Type, m_Dimension;

			//float m_Data[4];

		};
	};
};

#endif
