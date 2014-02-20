#include <nau/event/eventCameraOrientation.h>

using namespace nau::event_;

EventCameraOrientation::EventCameraOrientation(nau::event_::CameraOrientation cam)
{
	this->cam=cam;
}

EventCameraOrientation::EventCameraOrientation(const EventCameraOrientation &c)
{
	cam=c.cam;
}



EventCameraOrientation::EventCameraOrientation(void)
{
	cam.setCameraOrientation(0,0,0,0,0,0,0);
}

EventCameraOrientation::~EventCameraOrientation(void)
{
}

void EventCameraOrientation::setData(void *data)
{
	nau::event_::CameraOrientation *d=(nau::event_::CameraOrientation *)data;
	this->cam=*d;
}


void *EventCameraOrientation::getData(void)
{
	return &cam;
}