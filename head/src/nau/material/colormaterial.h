#ifndef COLORMATERIAL_H
#define COLORMATERIAL_H

#include "nau/enums.h"
#include "nau/attribute.h"
#include "nau/attributeValues.h"

namespace nau
{
	namespace material
	{
		class ColorMaterial: public AttributeValues {

		public:
			FLOAT4_PROP(AMBIENT, 0);
			FLOAT4_PROP(DIFFUSE, 1);
			FLOAT4_PROP(SPECULAR, 2);
			FLOAT4_PROP(EMISSION, 3);
			
			FLOAT_PROP(SHININESS, 0);

			static AttribSet Attribs;

			//typedef enum {AMBIENT, DIFFUSE, SPECULAR, EMISSION, COUNT_FLOAT4PROPERTY} Float4Property;
			//typedef enum {SHININESS, COUNT_FLOATPROPERTY} FloatProperty;


			// Note: no validation is performed!
			//void setProp(int prop, Enums::DataType type, void *value);

			//float  getPropf(const FloatProperty prop) ;
			//const vec4 &getProp4f(Float4Property prop) ;

			//void setProp(Float4Property prop, float r, float g, float b, float a);
			//void setProp(Float4Property prop, float *v);
			//void setProp(Float4Property prop, const vec4& color);
			//void setProp(FloatProperty prop, float f);
			//void initArrays();

			ColorMaterial();
			~ColorMaterial();

			void prepare ();
			void restore();
			void clear();

			void clone( ColorMaterial &mat);

		protected:
			static bool Init();
			static bool Inited;

			//std::map<int, vec4> m_Float4Props;
			//std::map<int, float> m_FloatProps;

		};
	};
};

#endif // COLORMATERIAL_H
