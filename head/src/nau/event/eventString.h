#ifndef EventString_H
#define EventString_H

#include "nau/event/iEventData.h"
#include <string>

namespace nau
{
	namespace event_
	{
		class EventString : public IEventData
		{
		public:
			std::string m_Str;

			//EventString(std::string *str);
			EventString(const EventString &s);
			EventString(void);
			~EventString(void);
			
			void setData(void *data);			
			void *getData(void);

		};
	};
};

#endif