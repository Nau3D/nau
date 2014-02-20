#ifndef EVENTFACTORY_H
#define EVENTFACTORY_H

#include <nau/event/iEventData.h>
#include <string>

namespace nau
{
	namespace event_
	{
		class EventFactory
		{
		public:
			static nau::event_::IEventData* create (std::string type);
		private:
			EventFactory(void) {};
			~EventFactory(void) {};
		};
	};
};
#endif