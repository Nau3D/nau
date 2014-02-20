#ifndef __TEXTURE_SAMPLER__
#define __TEXTURE_SAMPLER__

#include <vector>

#include <nau/math/vec4.h>
#include <nau/render/texture.h>
#include <nau/attribute.h>

using namespace nau::math;

namespace nau {

	namespace render {
		class Texture;
	}
}

namespace nau {

	namespace material {
	
		class TextureSampler {
		
		public:

			typedef enum {WRAP_S, WRAP_T, WRAP_R, MIN_FILTER, MAG_FILTER, 
					COMPARE_FUNC, COMPARE_MODE, COUNT_ENUMPROPERTY} EnumProperty;
			typedef enum {BORDER_COLOR, COUNT_FLOAT4PROPERTY} Float4Property;
			typedef enum {ID, COUNT_UINTPROPERTY} UIntProperty;
			typedef enum {MIPMAP, COUNT_BOOLPROPERTY} BoolProperty;
			typedef enum {COUNT_FLOATPROPERTY} FloatProperty;
			typedef enum {COUNT_INTPROPERTY} IntProperty;

			static AttribSet Attribs;

			int getPrope(EnumProperty prop);
			const vec4 &getPropf4(Float4Property prop);
			unsigned int getPropui(UIntProperty);
			bool getPropb(BoolProperty);
			void *getProp(int prop, Enums::DataType type);

			virtual void setProp(EnumProperty prop, int value) = 0;
			virtual void setProp(Float4Property prop, float x, float y, float z, float w) = 0;
			virtual void setProp(Float4Property prop, vec4& value) = 0;
			// Note: no validation is performed!
			void setProp(int prop, Enums::DataType type, void *value);

			static TextureSampler* create(nau::render::Texture *t);

			virtual void prepare(int aUnit, int aDim) = 0;

			TextureSampler() ;		
		
		protected:
			std::map<int, vec4> m_Float4Props;
			std::map<int, int> m_EnumProps;
			std::map<int, unsigned int> m_UIntProps;
			std::map<int, bool> m_BoolProps;
			std::map<int, float> m_FloatProps;
			std::map<int, int> m_IntProps;


//			bool m_Mipmap;
		};
	};
};


#endif