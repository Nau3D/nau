#ifndef VIEWPORT_H
#define VIEWPORT_H

#include <nau/math/vec2.h>
#include <nau/math/vec4.h>
#include <nau/attribute.h>

#include <nau/event/ilistener.h>
#include <nau/event/eventVec3.h>

#include <map>
#include <string>

using namespace nau::event_;
using namespace nau::math;

namespace nau
{
	namespace render
	{
		class Viewport: public IListener
		{
		protected:
			//vec2 m_Origin;
			//vec2 m_Size;
			//vec2 m_RelSize;
			//vec2 m_RelOrigin;
			//vec4 m_BackgroundColor;
			//bool m_Fixed;
			//float m_Ratio;
			std::string m_Name;

			std::map<int, bool> m_BoolProps;
			std::map<int, vec2> m_Float2Props;
			std::map<int, vec4> m_Float4Props;
			std::map<int, float> m_FloatProps;
			std::map<int, int> m_IntProps;

			static bool Inited;
			static bool Init();

		public:

			typedef enum {ORIGIN, SIZE, SET_ORIGIN, SET_SIZE, COUNT_FLOAT2PROPERTY} Float2Property;
			typedef enum {CLEAR_COLOR, COUNT_FLOAT4PROPERTY} Float4Property;
			typedef enum {FULL, COUNT_BOOLPROPERTY} BoolProperty;
			typedef enum {RATIO, COUNT_FLOATPROPERTY} FloatProperty;
			typedef enum {COUNT_INTPROPERTY} IntProperty;
			static AttribSet Attribs;

			Viewport(void);
			~Viewport(void);

			void eventReceived(const std::string &sender, const std::string &eventType, IEventData *evtData);

			std::string &getName();
			void setName(std::string);

			void setProp(Float4Property prop, const vec4 &value);
			void setProp(Float2Property prop, const vec2 &value);
			void setProp(BoolProperty prop, bool value);
			void setProp(FloatProperty prop, float values);
			// Note: no validation is performed!
			void setProp(int prop, Enums::DataType type, void *value);

			const vec2 &getPropf2(Float2Property prop);
			const vec4 &getPropf4(Float4Property prop);
			bool getPropb(BoolProperty prop);
			float getPropf(FloatProperty prop);
			void *getProp(int prop, Enums::DataType type);




//			const vec2& getOrigin (void);
//			void setOrigin (float x, float y);
//			const vec2& getSize (void);
//			void setSize (float width, float height);

			//const vec4& getBackgroundColor (void);
			//void setBackgroundColor (const vec4& aColor);

			//float getRatio();
			//void setRatio(float m);

			//bool isFixed (void);
			//void setFixed (bool value);

			//bool isRelative(void);

		};
	};
};
#endif
