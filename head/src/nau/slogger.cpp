#include "nau/slogger.h"

#include "nau.h"
#include "nau/event/eventString.h"


static SLogger *instance = 0;

SLogger::SLogger(void) 
{
	m_Evt = new nau::event_::EventString();
}

SLogger::~SLogger(void)
{
}

SLogger& 
SLogger::getInstance()
{
	if (0 == instance){
		instance = new SLogger;
	}
	return *instance;
}



void 
SLogger::log (std::string m)
{
	m_Evt->setData((void *)&m);
	EVENTMANAGER->notifyEvent("LOG", "SLogger","",m_Evt);
	

}

