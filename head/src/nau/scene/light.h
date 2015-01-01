#ifndef LIGHT_H
#define LIGHT_H

#include <nau/math/vec3.h>
#include <nau/math/vec4.h>
#include <nau/scene/sceneobject.h>
#include <nau/attribute.h>
#include <nau/attributeValues.h>

#include <string>
#include <map>

using namespace nau::math;
using namespace nau::scene;

namespace nau
{
	namespace scene
	{
		class Light: public AttributeValues, public SceneObject
		{
		public:

			FLOAT4_PROP(POSITION,0);
			FLOAT4_PROP(DIRECTION,1);
			FLOAT4_PROP(NORMALIZED_DIRECTION,2);
			FLOAT4_PROP(COLOR,3);
			FLOAT4_PROP(AMBIENT,4);
			FLOAT4_PROP(SPECULAR,5);

			FLOAT_PROP(SPOT_EXPONENT, 0);
			FLOAT_PROP(SPOT_CUTOFF, 1);
			FLOAT_PROP(CONSTANT_ATT, 2);
			FLOAT_PROP(LINEAR_ATT, 3);
			FLOAT_PROP(QUADRATIC_ATT, 4);

			BOOL_PROP(ENABLED, 0);

			INT_PROP(ID, 0);


			static AttribSet Attribs;

			Light (std::string &name);
			~Light(void);

			// returns "LIGHT"
			std::string getType();

			virtual void setPropf(FloatProperty prop, float value);
			virtual void setPropf4(Float4Property prop, float r, float g, float b, float a);
			virtual void setPropf4(Float4Property prop, vec4& aVec);
			//void setProp(BoolProperty prop, bool value);
			//void setProp(EnumProperty, int value);
			//void setProp(IntProperty, int value);
			// Note: no validation is performed!
			//void setProp(int prop, Enums::DataType type, void *value);


			//float getPropf(FloatProperty prop);
			//const vec4 &getPropf4(Float4Property prop);
			//bool getPropb(BoolProperty prop);
			//int getPrope(EnumProperty prop);
			//int getPropi(IntProperty prop);
			//void *getProp(int prop, Enums::DataType type);


		protected:

			static bool Inited;
			static bool Init();
		};
	};
};



#endif
