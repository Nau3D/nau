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
		public:
			nau::event_::CameraMotion cam;
			
			EventCameraMotion(nau::event_::CameraMotion cam);
			EventCameraMotion(const EventCameraMotion &c);
			EventCameraMotion(void);
			~EventCameraMotion(void);
			
			void setData(void *data);
			void *getData(void);			
		};
	};
};

#endif