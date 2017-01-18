#include "nau/event/eventFloat.h"

using namespace nau::event_;

EventFloat::EventFloat(float flt)
{
	this->flt=flt;
}

EventFloat::EventFloat(const EventFloat &s)
{
	flt=s.flt;
}

EventFloat::EventFloat(void)
{
	flt=0;
}

EventFloat::~EventFloat(void)
{
}

void EventFloat::setData(void *data)
{
	float *d=(float*)data;
	this->flt=*d;
}


void *EventFloat::getData(void)
{
	return &flt;
}