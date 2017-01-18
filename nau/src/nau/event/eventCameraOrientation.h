#ifndef EVENT_CAMERA_ORIENTATION_H
#define EVENT_CAMERA_ORIENTATION_H

#include "nau/event/iEventData.h"
#include "nau/event/cameraOrientation.h"

namespace nau
{
	namespace event_
	{
		class EventCameraOrientation : public IEventData
		{
			friend class EventFactory;

		public:
			~EventCameraOrientation(void);

			void setData(void *data);
			void *getData(void);

		protected:
			EventCameraOrientation(nau::event_::CameraOrientation cam);
			EventCameraOrientation(const EventCameraOrientation &c);
			EventCameraOrientation(void);

			nau::event_::CameraOrientation cam;
		};
	};
};

#endif