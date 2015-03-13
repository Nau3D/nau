#ifndef EventFloat_H
#define EventFloat_H

#include "nau/event/iEventData.h"

namespace nau
{
	namespace event_
	{
		class EventFloat : public IEventData
		{
		public:
			float flt;

			EventFloat(float flt);
			EventFloat(const EventFloat &f);
			EventFloat(void);
			~EventFloat(void);
			
			void setData(void *data);
			void *getData(void);

		};
	};
};

#endif