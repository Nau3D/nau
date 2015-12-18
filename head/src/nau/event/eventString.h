#ifndef EventString_H
#define EventString_H

#include "nau/event/eventFactory.h"
#include "nau/event/iEventData.h"
#include <string>

namespace nau
{
	namespace event_
	{
		class EventString : public IEventData
		{
			friend class EventFactory;

		public:
			~EventString(void);
			
			void setData(void *data);			
			void *getData(void);

		protected:
			EventString(const EventString &s);
			EventString(void);

			std::string m_Str;
		};
	};
};

#endif