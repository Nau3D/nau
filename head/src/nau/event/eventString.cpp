#include <nau/event/eventString.h>

using namespace nau::event_;

EventString::EventString(std::string *str)
{
	this->str = str;
}

EventString::EventString(const EventString &s)
{
	str = s.str;
}

EventString::EventString(void)
{
	str = new std::string("");
}

EventString::~EventString(void)
{
}

void EventString::setData(void *data)
{
	str = (std::string *)data;
}

void *EventString::getData(void)
{
	return str;
}