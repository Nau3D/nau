#ifndef _NAU_PHYSICS_MATERIAL
#define _NAU_PHYSICS_MATERIAL

#include "nau/attributeValues.h"
#include "nau/enums.h"
#include "nau/math/data.h"


namespace nau 
{
	namespace physics
	{
		class PhysicsManager;

		class PhysicsMaterial: public AttributeValues 
		{
			

		public:
			ENUM_PROP(SCENE_TYPE, 0);
			ENUM_PROP(SCENE_SHAPE, 1);
			ENUM_PROP(SCENE_CONDITION, 2);
			FLOAT_PROP(NBPARTICLES, 0);
			FLOAT_PROP(MAX_PARTICLE, 1);
			STRING_PROP(BUFFER, 0);
			FLOAT4_PROP(DIRECTION, 0);
			FLOAT4_PROP(SCENE_CONDITION_VALUE, 1);

			static AttribSet Attribs;

			PhysicsMaterial(const std::string &name);
			PhysicsMaterial();

			void setPropf(FloatProperty p, float value);
			void setPropf4(Float4Property p, vec4 &value);
			void setProps(StringProperty prop, std::string &value);

			float * getBuffer() { return buffer;}
			void setBuffer(float * b) { buffer = b; }

		protected:

			static bool Init();
			static bool Inited;

			std::string m_Name;
			float * buffer;

		};
	};
};

#endif