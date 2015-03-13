#ifndef EventCameraOrientation_H
#define EventCameraOrientation_H

#include "nau/event/iEventData.h"
#include "nau/event/cameraOrientation.h"

namespace nau
{
	namespace event_
	{
		class EventCameraOrientation : public IEventData
		{
		public:
			nau::event_::CameraOrientation cam;
			
			EventCameraOrientation(nau::event_::CameraOrientation cam);
			EventCameraOrientation(const EventCameraOrientation &c);
			EventCameraOrientation(void);
			~EventCameraOrientation(void);
			
			void setData(void *data);
			void *getData(void);

		};
	};
};

#endif