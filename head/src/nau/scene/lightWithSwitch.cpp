#include "nau/scene/lightWithSwitch.h"

#include "nau.h"

using namespace nau::scene;
using namespace nau::math;
using namespace nau;

LightWithSwitch::LightWithSwitch(std::string &name): Light(name) {

	this->name=name;
	this->addLightListener(); 
}


LightWithSwitch::~LightWithSwitch(void) {

}


std::string& LightWithSwitch::getName (void) {
	
	return name;
}


void LightWithSwitch::lightOff() {

	this->m_BoolProps[ENABLED] = false;
}


void LightWithSwitch::lightOn() {

	this->m_BoolProps[ENABLED] = true;
}


void LightWithSwitch::addLightListener(void) {

	EVENTMANAGER->addListener("LIGHT_ON",(nau::event_::IListener *)this);
	EVENTMANAGER->addListener("LIGHT_OFF",(nau::event_::IListener *)this);
}


void LightWithSwitch::removeLightListener(void) {

	EVENTMANAGER->removeListener("LIGHT_ON",(nau::event_::IListener *)this);
	EVENTMANAGER->addListener("LIGHT_OFF",(nau::event_::IListener *)this);
}


void LightWithSwitch::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<IEventData> &evt) {

	if (eventType == "LIGHT_ON")
		lightOn();
	if (eventType == "LIGHT_OFF")
		lightOff();
}