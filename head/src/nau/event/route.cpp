#include "nau/event/route.h"
#include "nau/event/eventFactory.h"
#include "nau.h"

using namespace nau::event_;

Route::Route(std::string name, std::string sender, std::string receiver, std::string eventIn, std::string eventOut){
	this->m_Name = name;
	this->sender=sender;
	this->receiver=receiver;
	this->eventIn=eventIn;
	this->eventOut=eventOut;
	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->addListener(&this->eventIn[0],lst);
}

Route::Route(void){
	this->m_Name="";
	this->sender="";
	this->receiver="";
	this->eventIn="";
	this->eventOut="";
}

Route::~Route(void){
	//nau::event_::Listener *lst;
	//lst = this;
	//EVENTMANAGER->removeListener(&eventIn[0],lst);
}

std::string &Route::getEventIn(void){
	return eventIn;
}
std::string &Route::getEventOut(void){
	return eventOut;
}
std::string &Route::getSender(void){
	return sender;
}
std::string &Route::getReceiver(void){
	return receiver;
}
std::string &Route::getName(void){
	return m_Name;
}

void Route::eventReceived(const std::string &aSender, const std::string &eventType, nau::event_::IEventData *evt){
	if(eventType==eventIn && sender == aSender){
		EVENTMANAGER->notifyEvent(eventOut,sender,receiver,evt);
	}
}

void Route::init(std::string name, std::string sender, std::string receiver, std::string eventIn, std::string eventOut){
	this->m_Name=name;
	this->sender=sender;
	this->receiver=receiver;
	this->eventIn=eventIn;
	this->eventOut=eventOut;
	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->addListener(this->eventIn,lst);
}
