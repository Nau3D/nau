#ifndef EVENTFACTORY_H
#define EVENTFACTORY_H

#include "nau/event/iEventData.h"

#include <memory>
#include <string>

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

namespace nau
{
	namespace event_
	{
		class EventFactory
		{
		public:
			static nau_API std::shared_ptr<IEventData> Create (std::string type);
		private:
			EventFactory(void) {};
			~EventFactory(void) {};
		};
	};
};
#endif