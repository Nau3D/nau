#ifndef LIGHTWITHSWITCH_H
#define LIGHTWITHSWITCH_H

// Marta
#include <nau/scene/light.h>
#include <nau/event/iEventData.h>
#include <nau/event/ilistener.h>

namespace nau
{
	namespace scene
	{
		class LightWithSwitch: public Light
		{
		public:

			std::string name;
			LightWithSwitch (std::string &name);
			~LightWithSwitch(void);

			std::string& getName (void);

			void lightOff(void);
			void lightOn(void);
			void addLightListener(void);
			void removeLightListener(void);
			void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);
		};
	};
};
#endif