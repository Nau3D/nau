#ifndef EventVec3_H
#define EventVec3_H

#include <nau/event/iEventData.h>
#include <nau/math/vec3.h>

namespace nau
{
	namespace event_
	{
		class EventVec3: public IEventData
		{
		public:
			nau::math::vec3 v;

			EventVec3(nau::math::vec3 v);
			EventVec3(const EventVec3 &c);
			EventVec3(void);
			~EventVec3(void);
			
			void setData(void *data);
			void *getData(void);

		};
	};
};

#endif