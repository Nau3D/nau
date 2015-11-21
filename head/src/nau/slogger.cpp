#include "nau/slogger.h"

#include "nau.h"
#include "nau/event/eventString.h"

#include <memory>

SLogger * SLogger::Instance = 0;


SLogger::SLogger(void) {

}


SLogger::~SLogger(void) {

}


SLogger* 
SLogger::GetInstance() {

	if (0 == Instance){
		Instance = new SLogger;
	}
	return Instance;
}


void
SLogger::DeleteInstance() {

	if (Instance) {
		delete Instance;
		Instance = NULL;
	}
}


void 
SLogger::log (std::string m) {
	
	std::shared_ptr<EventString> m_Evt =
		dynamic_pointer_cast<EventString>(EventFactory::Create("String"));

	m_Evt->setData((void *)&m);
	EVENTMANAGER->notifyEvent("LOG", "SLogger","",m_Evt);
}

