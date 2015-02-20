#include "nau/event/eventCameraMotion.h"

using namespace nau::event_;

EventCameraMotion::EventCameraMotion(nau::event_::CameraMotion cam)
{
	this->cam=cam;	
}

EventCameraMotion::EventCameraMotion(const EventCameraMotion &c)
{
	cam=c.cam;
}



EventCameraMotion::EventCameraMotion(void)
{
	cam.setCameraMotion("",0);
}

EventCameraMotion::~EventCameraMotion(void)
{
}

void EventCameraMotion::setData(void *data)
{
	nau::event_::CameraMotion *d=(nau::event_::CameraMotion *)data;
	this->cam=*d;
}


void *EventCameraMotion::getData(void)
{
	return &cam;
}