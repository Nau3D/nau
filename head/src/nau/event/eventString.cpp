#include "nau/event/eventString.h"

using namespace nau::event_;

EventString::EventString(const EventString &s) {
	m_Str = s.m_Str;
}


EventString::EventString(void) {

	m_Str = "";
}


EventString::~EventString(void) {

}


void EventString::setData(void *data) {

	m_Str = *(std::string *)data;
}


void *EventString::getData(void) {

	return &m_Str;
}