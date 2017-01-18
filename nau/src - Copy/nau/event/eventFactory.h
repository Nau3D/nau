#ifndef EVENTFACTORY_H
#define EVENTFACTORY_H

#include "nau/event/iEventData.h"

#include <memory>
#include <string>

namespace nau
{
	namespace event_
	{
		class EventFactory
		{
		public:
			static std::shared_ptr<IEventData> Create (std::string type);
		private:
			EventFactory(void) {};
			~EventFactory(void) {};
		};
	};
};
#endif