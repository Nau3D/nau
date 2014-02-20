#ifndef LIGHT_H
#define LIGHT_H

#include <nau/math/vec3.h>
#include <nau/math/vec4.h>
#include <nau/scene/sceneobject.h>
#include <nau/attribute.h>

#include <string>
#include <map>

using namespace nau::math;
using namespace nau::scene;

namespace nau
{
	namespace scene
	{
		class Light: public SceneObject
		{
		public:

			typedef enum {
				DIRECTIONAL,
				POSITIONAL,
				SPOT_LIGHT,
				OMNILIGHT,COUNT_LIGHTTYPE
			} LightType;

			typedef enum {POSITION, DIRECTION, NORMALIZED_DIRECTION, COLOR,
				AMBIENT,SPECULAR,
				COUNT_FLOAT4PROPERTY} Float4Property;
			typedef enum {SPOT_EXPONENT, SPOT_CUTOFF, CONSTANT_ATT, 
				LINEAR_ATT, QUADRATIC_ATT,
				COUNT_FLOATPROPERTY} FloatProperty;
			typedef enum {ENABLED, COUNT_BOOLPROPERTY} BoolProperty;
			typedef enum {TYPE, COUNT_ENUMPROPERTY} EnumProperty;
			typedef enum {ID, COUNT_INTPROPERTY} IntProperty;


			static AttribSet Attribs;

			Light (std::string &name);
			~Light(void);

			void init (nau::math::vec3 position,
				  nau::math::vec3 direction,
				  nau::math::vec4 color, int enabled, LightType type);

			void setDefault();


			// returns "LIGHT"
			std::string getType();

			void setProp(FloatProperty prop, float value);
			void setProp(Float4Property prop, float r, float g, float b, float a);
			void setProp(Float4Property prop, const vec4& aVec);
			void setProp(BoolProperty prop, bool value);
			void setProp(EnumProperty, int value);
			void setProp(IntProperty, int value);
			// Note: no validation is performed!
			void setProp(int prop, Enums::DataType type, void *value);


			float getPropf(FloatProperty prop);
			const vec4 &getPropf4(Float4Property prop);
			bool getPropb(BoolProperty prop);
			int getPrope(EnumProperty prop);
			int getPropi(IntProperty prop);
			void *getProp(int prop, Enums::DataType type);


		protected:
			std::map<int,float> m_FloatProps;
			std::map<int,vec4> m_Float4Props;
			std::map<int,bool> m_BoolProps;
			std::map<int,int> m_EnumProps;
			std::map<int,int> m_IntProps;

			static bool Inited;
			static bool Init();
		};
	};
};



#endif
