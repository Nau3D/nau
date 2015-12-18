#ifndef EventVec4_H
#define EventVec4_H

#include "nau/event/iEventData.h"
#include "nau/event/eventvec4.h"
#include "nau/math/vec4.h"

namespace nau
{
	namespace event_
	{
		class EventVec4: public IEventData
		{
			friend class EventFactory;

		public:
			~EventVec4(void);
			
			void setData(void *data);
			void *getData(void);

		protected:
			EventVec4(nau::math::vec4 v);
			EventVec4(const EventVec4 &c);
			EventVec4(void);

			nau::math::vec4 v;
		};
	};
};

#endif