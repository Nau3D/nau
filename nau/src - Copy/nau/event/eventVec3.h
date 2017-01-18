#ifndef EVENT_VEC3_H
#define EVENT_VEC3_H

#include "nau/event/iEventData.h"
#include "nau/math/vec3.h"

namespace nau
{
	namespace event_
	{
		class EventVec3: public IEventData
		{
			friend class EventFactory;

		public:
			~EventVec3(void);
			
			void setData(void *data);
			void *getData(void);

		protected:
			EventVec3(nau::math::vec3 v);
			EventVec3(const EventVec3 &c);
			EventVec3(void);

			nau::math::vec3 v;
		};
	};
};

#endif