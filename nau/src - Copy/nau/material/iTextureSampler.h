#ifndef TEXTURE_SAMPLER
#define TEXTURE_SAMPLER

#include "nau/math/vec4.h"
#include "nau/material/iTexture.h"
#include "nau/attributeValues.h"
#include "nau/attribute.h"

#include <vector>

using namespace nau::math;


namespace nau {

	namespace material {
		class ITexture;
	}
}

namespace nau {

	namespace material {
	
		class ITextureSampler : public AttributeValues {
		
		public:

			ENUM_PROP(WRAP_S, 0);
			ENUM_PROP(WRAP_T, 1);
			ENUM_PROP(WRAP_R, 2);
			ENUM_PROP(MIN_FILTER, 3);
			ENUM_PROP(MAG_FILTER, 4);
			ENUM_PROP(COMPARE_FUNC, 5);
			ENUM_PROP(COMPARE_MODE, 6);

			FLOAT4_PROP(BORDER_COLOR, 0);

			BOOL_PROP(MIPMAP, 0);

			INT_PROP(ID, 0);	

			static AttribSet Attribs;

			//int getPrope(EnumProperty prop);
			//const vec4 &getPropf4(Float4Property prop);
			//int getPropi(IntProperty);
			//bool getPropb(BoolProperty);
			//void *getProp(int prop, Enums::DataType type);

			//virtual void setProp(EnumProperty prop, int value) = 0;
			//virtual void setProp(Float4Property prop, float x, float y, float z, float w) = 0;
			//virtual void setProp(Float4Property prop, vec4& value) = 0;
			// Note: no validation is performed!
			//void setProp(int prop, Enums::DataType type, void *value);

			static ITextureSampler* create(ITexture *t);

			virtual void prepare(unsigned int aUnit, int aDim) = 0;
			virtual void restore(unsigned int aUnit, int aDim) = 0;


			virtual ~ITextureSampler() {};
		
		protected:

			static bool Init();
			static bool Inited;

			ITextureSampler() ;	

		};
	};
};


#endif