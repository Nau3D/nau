#include <nau/scene/lightWithSwitch.h>
#include <nau.h>

using namespace nau::scene;
using namespace nau::math;
using namespace nau;

LightWithSwitch::LightWithSwitch(std::string &name): Light(name)
{
	this->name=name;
	this->addLightListener(); 
}

LightWithSwitch::~LightWithSwitch(void)
{
}

std::string& LightWithSwitch::getName (void){
	
	return name;

}

void LightWithSwitch::lightOff(){

	this->m_BoolProps[ENABLED] = false;
}

void LightWithSwitch::lightOn(){

	this->m_BoolProps[ENABLED] = true;
}

void LightWithSwitch::addLightListener(void)
{
	//nau::event_::IListener *lst;
	//lst = this;
	EVENTMANAGER->addListener("LIGHT_ON",(nau::event_::IListener *)this);
	EVENTMANAGER->addListener("LIGHT_OFF",(nau::event_::IListener *)this);
}

void LightWithSwitch::removeLightListener(void)
{
	//nau::event_::IListener *lst;
	//lst = this;
	EVENTMANAGER->removeListener("LIGHT_ON",(nau::event_::IListener *)this);
	EVENTMANAGER->addListener("LIGHT_OFF",(nau::event_::IListener *)this);
}

void LightWithSwitch::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt){
	//if (evt->getReceiver()==this->getName()){
		if (eventType == "LIGHT_ON")
			lightOn();
		if (eventType == "LIGHT_OFF")
			lightOff();
	//}
}