#ifndef LIGHT_H
#define LIGHT_H

#include "nau/math/vec3.h"
#include "nau/math/vec4.h"
#include "nau/scene/sceneObject.h"
#include "nau/attribute.h"
#include "nau/attributeValues.h"

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
			friend class LightFactory;
		public:

			FLOAT4_PROP(POSITION,0);
			FLOAT4_PROP(DIRECTION,1);
			FLOAT4_PROP(NORMALIZED_DIRECTION,2);
			FLOAT4_PROP(SPOT_DIRECTION, 3);
			FLOAT4_PROP(COLOR,4);
			FLOAT4_PROP(AMBIENT,5);

			FLOAT_PROP(SPOT_EXPONENT, 0);
			FLOAT_PROP(SPOT_CUTOFF, 1);
			FLOAT_PROP(CONSTANT_ATT, 2);
			FLOAT_PROP(LINEAR_ATT, 3);
			FLOAT_PROP(QUADRATIC_ATT, 4);

			BOOL_PROP(ENABLED, 0);

			INT_PROP(ID, 0);

			static AttribSet Attribs;
			static nau_API AttribSet &GetAttribs();

			~Light(void);

			void eventReceived(const std::string & sender, const std::string & eventType, 
				const std::shared_ptr<IEventData>& evt);

			// returns "LIGHT"
			std::string getType();

			virtual void setPropf(FloatProperty prop, float value);
			virtual void setPropf4(Float4Property prop, float r, float g, float b, float a);
			virtual void setPropf4(Float4Property prop, vec4& aVec);

		protected:

			Light (std::string &name);
			static bool Inited;
			static bool Init();
		};
	};
};



#endif
