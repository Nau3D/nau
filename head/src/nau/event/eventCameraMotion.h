#ifndef EventCameraMotion_H
#define EventCameraMotion_H

#include "nau/event/iEventData.h"
#include "nau/event/cameraMotion.h"

namespace nau
{
	namespace event_
	{
		class EventCameraMotion : public IEventData
		{
			friend class EventFactory;

		public:
			~EventCameraMotion(void);
			
			void setData(void *data);
			void *getData(void);

		protected:
			EventCameraMotion(nau::event_::CameraMotion cam);
			EventCameraMotion(const EventCameraMotion &c);
			EventCameraMotion(void);

			nau::event_::CameraMotion cam;
		};
	};
};

#endif