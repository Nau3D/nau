#include "nau/slogger.h"

#include "nau.h"
#include "nau/event/eventString.h"


SLogger * SLogger::Instance = 0;


SLogger::SLogger(void) {

	m_Evt = new nau::event_::EventString();
}


SLogger::~SLogger(void) {

	delete m_Evt;
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
	
	m_Evt->setData((void *)&m);
	EVENTMANAGER->notifyEvent("LOG", "SLogger","",m_Evt);
}

