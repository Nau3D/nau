#ifndef EventInt_H
#define EventInt_H

#include <nau/event/iEventData.h>

namespace nau
{
	namespace event_
	{
		class EventInt : public IEventData
		{
		public:
			int flt;

			EventInt(int flt);
			EventInt(const EventInt &f);
			EventInt(void);
			~EventInt(void);
			
			void setData(void *data);
			void *getData(void);

		};
	};
};

#endif