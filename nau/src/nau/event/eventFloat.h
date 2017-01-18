#ifndef EVENT_FLOAT_H
#define EVENT_FLOAT_H

#include "nau/event/iEventData.h"

namespace nau
{
	namespace event_
	{
		class EventFloat : public IEventData
		{
			friend class EventFactory;

		public:
			~EventFloat(void);
			
			void setData(void *data);
			void *getData(void);

		protected:
			EventFloat(float flt);
			EventFloat(const EventFloat &f);
			EventFloat(void);

			float flt;
		};
	};
};

#endif