#ifndef VIEWPORT_H
#define VIEWPORT_H

#include <nau/math/vec2.h>
#include <nau/math/vec4.h>
#include <nau/attribute.h>
#include <nau/attributeValues.h>
#include <nau/event/ilistener.h>
#include <nau/event/eventVec3.h>

#include <string>

using namespace nau::event_;
using namespace nau::math;
using namespace nau;

namespace nau
{
	namespace render
	{
		class Viewport: public AttributeValues, public IListener
		{
		public:

			FLOAT2_PROP(ORIGIN, 0);
			FLOAT2_PROP(SIZE, 1);
			FLOAT2_PROP(ABSOLUT_ORIGIN, 2);
			FLOAT2_PROP(ABSOLUT_SIZE, 3);

			BOOL_PROP(FULL, 0);

			FLOAT_PROP(RATIO, 0);

			FLOAT4_PROP(CLEAR_COLOR, 0);

			static AttribSet Attribs;

			Viewport(void);
			~Viewport(void);

			void eventReceived(const std::string &sender, const std::string &eventType, IEventData *evtData);

			std::string &getName();
			void setName(std::string);

			void setPropb(BoolProperty prop, bool value);
			void setPropf(FloatProperty prop, float values);
			void setPropf2(Float2Property prop, vec2 &value);
			//void setPropf4(Float4Property prop,vec4 &value);
			// Note: no validation is performed!
			//void setProp(int prop, Enums::DataType type, void *value);

			float getPropf(FloatProperty prop);

		protected:
			std::string m_Name;

			static bool Inited;
			static bool Init();
		};
	};
};
#endif
