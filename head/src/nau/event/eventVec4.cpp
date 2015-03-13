#include "nau/event/eventVec4.h"


using namespace nau::event_;

EventVec4::EventVec4(nau::math::vec4 v)
{
	this->v = v;
}


EventVec4::EventVec4(const EventVec4 &c)
{
	v=c.v;
}


EventVec4::EventVec4(void)
{
	v.set(0.0f, 0.0f, 0.0f, 0.0f);
}


EventVec4::~EventVec4(void)
{
}


void EventVec4::setData(void *data) {
	
	nau::math::vec4 *ve=(nau::math::vec4 *)data;
	this->v=*ve;
}


void *EventVec4::getData(void) {
	return &v;
}



