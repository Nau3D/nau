#include "nau/event/eventManager.h"
#include "nau/event/sensorFactory.h"
#include "nau/event/interpolatorFactory.h"
#include <string.h>

using namespace nau::event_;

EventManager::EventManager(vector<ListenerType *> *listeners)
{
	this->listeners=*listeners;
}

EventManager::EventManager(const EventManager &a)
{
	listeners=a.listeners;
}

EventManager::EventManager(void)
{
	 listeners.clear();
}

EventManager::~EventManager(void) {

	clear();
}

void EventManager::setListeners(vector<ListenerType *> *listeners)
{
	this->listeners=*listeners;
}

vector<ListenerType *> *EventManager::getAllListeners(void)
{
	return &listeners;
}

vector<IListener*> *EventManager::getListeners(std::string eventType)
{
	vector<ListenerType *>::iterator it;

	for(it=listeners.begin();it != listeners.end();it++)
		if((*it)->eventType == eventType)
			return &(*it)->lsts;
	return 0;
}

void EventManager::addListener(std::string eventType, IListener *lst)
{
	vector<ListenerType *>::iterator it;

	for(it=listeners.begin();it != listeners.end();it++)
		if((*it)->eventType == eventType){
			(*it)->lsts.push_back(lst);
			return;
		}
	
	// if eventType is not in vector listeners
	ListenerType *l = new ListenerType();
	listeners.push_back(l);
	l->eventType=eventType;
	l->lsts.push_back(lst);

}

void EventManager::removeListener(std::string eventType, IListener *lst)
{
	vector<ListenerType *>::iterator it;
	vector<IListener *>::iterator itL;

	if (listeners.empty())
		return;

	for(it=listeners.begin();it != listeners.end();it++)
		if((*it)->eventType == eventType)
			for(itL=(*it)->lsts.begin();itL != (*it)->lsts.end();itL++){
				if (*itL == lst) {
					(*it)->lsts.erase(itL);
					return;
				}
		}
}

void EventManager::eraseAllListenersType(std::string eventType)
{
	vector<ListenerType *>::iterator it;

	for(it=listeners.begin();it != listeners.end();it++)
		if((*it)->eventType == eventType){
			listeners.erase(it);
			return;
		}
}


void EventManager::eraseAllListeners(void)
{
	clear();
}


void EventManager::clear(void)
{
	while (!listeners.empty()) {

		delete(*listeners.begin());
		listeners.erase(listeners.begin());
	}
}


void 
EventManager::notify(std::string eventType, std::string sender, std::string receiver, 
	const std::shared_ptr<IEventData> &evt, vector<IListener *> lsts)
{
	vector<IListener *>::iterator it;

	for(it=lsts.begin();it != lsts.end();it++){
		std::string n=(*it)->getName();
		if(receiver == n || receiver == "")
			(*it)->eventReceived(sender,eventType, evt);
	}

}

void EventManager::notifyEvent(std::string eventType, std::string sender, 
	std::string receiver, const std::shared_ptr<IEventData> &evt)
{
	vector<ListenerType *>::iterator itL;
	
	for(itL=listeners.begin();itL != listeners.end();itL++){
		if((*itL)->eventType == eventType){
			notify(eventType, sender, receiver, evt, (*itL)->lsts);
			return;
		}
	}
	
}
bool 
EventManager::hasSensor (std::string sensorName)
{
	if (m_Sensors.count (sensorName) > 0)
		return true;
	
	return false;
}

Sensor* 
EventManager::getSensor (std::string sensorName, std::string sClass)
{
	if (false == hasSensor (sensorName)) {
		m_Sensors[sensorName] = nau::event_::SensorFactory::create(sClass);
		m_Sensors[sensorName]->setName(sensorName);
	}
	return m_Sensors[sensorName];
}

bool
EventManager::hasInterpolator (std::string interpolatorName)
{
	if (m_Interpolators.count (interpolatorName) > 0)
		return true;
	
	return false;
}

Interpolator* 
EventManager::getInterpolator (std::string interpolatorName, std::string sClass)
{
	if (false == hasInterpolator (interpolatorName)) {
		m_Interpolators[interpolatorName] = nau::event_::InterpolatorFactory::create(sClass);
		m_Interpolators[interpolatorName]->setName(interpolatorName);
	}
	return m_Interpolators[interpolatorName];
}

bool 
EventManager::hasRoute (std::string routeName)
{
	if (m_Routes.count (routeName) > 0)
		return true;
	
	return false;
}

Route* 
EventManager::getRoute (std::string routeName)
{
	if (false == hasRoute (routeName)) {
		m_Routes[routeName] = new Route;
	}
	return m_Routes[routeName];
}