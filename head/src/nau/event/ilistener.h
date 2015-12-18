#ifndef LISTENER_H
#define LISTENER_H

#include "nau/event/iEventData.h"

#include <memory>
#include <string>

namespace nau
{
	namespace event_
	{
		class IListener
		{
		public:

			// Construtors
			IListener(void){};
			~IListener(void){};

			// Methods
			virtual std::string &getName () = 0;
			virtual void eventReceived(const std::string &sender, 
				const std::string &eventType, 
				const std::shared_ptr<IEventData> &evt) = 0;
		};
	};
};
#endif