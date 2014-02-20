#include <nau/event/listenerType.h>

using namespace nau::event_;

ListenerType::ListenerType(std::string eventType, vector<IListener *> *lsts)
{
	this->eventType=eventType;
	this->lsts=*lsts;
}

ListenerType::ListenerType(const ListenerType &e)
{
	eventType=e.eventType;
	lsts=e.lsts;
}

ListenerType::ListenerType(void)
{
	eventType="";
	lsts.clear();
}

ListenerType::~ListenerType(void)
{
}

void ListenerType::setEventType(std::string eventType)
{
	this->eventType=eventType;
}


std::string *ListenerType::getEventType(void)
{
	return &eventType;
}

void ListenerType::setListeners(vector<IListener *> *lsts)
{
	this->lsts=*lsts;
}

vector<IListener *> *ListenerType::getListeners(void)
{
	return &lsts;
}