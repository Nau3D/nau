#include <nau/event/eventVec3.h>


using namespace nau::event_;

EventVec3::EventVec3(nau::math::vec3 v)
{
	this->v = v;
}


EventVec3::EventVec3(const EventVec3 &c)
{
	v=c.v;
}


EventVec3::EventVec3(void)
{
	v.set(0.0f, 0.0f, 0.0f);
}

EventVec3::~EventVec3(void)
{
}


void EventVec3::setData(void *data) {
	
	nau::math::vec3 *ve=(nau::math::vec3 *)data;
	this->v=*ve;
}

void *EventVec3::getData(void) {
	return &v;
}



