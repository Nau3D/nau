#include "nau/event/positionInterpolator.h"
#include "nau/event/eventFactory.h"
#include "nau.h"

using namespace nau::event_;


PositionInterpolator::PositionInterpolator(const PositionInterpolator &c) {
	
	m_Name=c.m_Name;
	m_KeyFrames = c.m_KeyFrames;
	this->fraction=fraction;
}


PositionInterpolator::PositionInterpolator(void) {
	
	m_Name="";
	m_KeyFrames.clear();
	this->addPositionListener();
}


PositionInterpolator::~PositionInterpolator(void) {
		
	m_KeyFrames.clear();
	this->removePositionListener();
}


void PositionInterpolator::removePositionListener(void) {
	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->removeListener("SET_INTERPOLATOR_FRACTION",lst);
}


void PositionInterpolator::addPositionListener(void) {
	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->addListener("SET_INTERPOLATOR_FRACTION",lst);
}


void PositionInterpolator::eventReceived(const std::string &sender, 
	const std::string &eventType, const std::shared_ptr<IEventData> &evt) {
	
	float *f=(float *)evt->getData();
	float fc=*f;

	std::map<float, vec4>::iterator keyNext, keyIter = m_KeyFrames.begin();

	for( ; keyIter != --(m_KeyFrames.end()); ++keyIter){
		keyNext = keyIter; keyNext++;
		float k0=(*keyIter).first;
		float k1=(*(keyNext)).first;
		if(fc>=k0 && fc<=k1){
			vec4 kv0((*keyIter).second);
			vec4 kv1((*keyNext).second);
			vec4 kv;
			kv.x=kv0.x+((fc-k0)*(kv1.x-kv0.x)/(k1-k0));
			kv.y=kv0.y+((fc-k0)*(kv1.y-kv0.y)/(k1-k0));
			kv.z=kv0.z+((fc-k0)*(kv1.z-kv0.z)/(k1-k0));
			kv.w=kv0.w+((fc-k0)*(kv1.w-kv0.w)/(k1-k0));

			std::shared_ptr<IEventData> e = EventFactory::Create("Vec4");
			e->setData(&kv);

			EVENTMANAGER->notifyEvent("INTERPOLATOR_POSITION", &m_Name[0], "", e);
			return;
		}
	}			
}

