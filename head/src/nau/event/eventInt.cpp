#include "nau/event/eventInt.h"

using namespace nau::event_;

EventInt::EventInt(int flt)
{
	this->flt=flt;
}

EventInt::EventInt(const EventInt &s)
{
	flt=s.flt;
}

EventInt::EventInt(void)
{
	flt=0;
}

EventInt::~EventInt(void)
{
}

void EventInt::setData(void *data)
{
	int *d=(int*)data;
	this->flt=*d;
}


void *EventInt::getData(void)
{
	return &flt;
}