#ifndef EventInt_H
#define EventInt_H

#include "nau/event/iEventData.h"

namespace nau
{
	namespace event_
	{
		class EventInt : public IEventData
		{
			friend class EventFactory;

		public:
			~EventInt(void);
			
			void setData(void *data);
			void *getData(void);

		protected:
			EventInt(int flt);
			EventInt(const EventInt &f);
			EventInt(void);

			int flt;
		};
	};
};

#endif