#include <nau/event/proximitySensor.h>
#include <nau/event/eventFactory.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <time.h>
#include <nau.h>
#include <string>

using namespace nau::event_;

const std::string ProximitySensor::Vec3PropNames[] = {"center","size"};
const std::string ProximitySensor::BoolPropNames[] = {"enabled"};


ProximitySensor::ProximitySensor(std::string aname, bool enabled, vec3 center, vec3 size):
			BoolProps(COUNT_BOOLPROP),
			Vec3Props(COUNT_VEC3PROP)
{
	m_Name = aname;
	BoolProps[ENABLED] = enabled;
	Vec3Props[CENTER].set(center.x, center.y, center.z);
	Vec3Props[SIZE].set(size.x, size.y, size.z);

	this->min=getMin();
	this->max=getMax();
	this->previous = false;

	this->addProximityListener();
}


ProximitySensor::ProximitySensor(const ProximitySensor &c):

			BoolProps(COUNT_BOOLPROP),
			Vec3Props(COUNT_VEC3PROP)
{
	m_Name = c.m_Name;
	BoolProps[ENABLED] = c.enabled;
	vec3 v; 
	v.set(c.center.x, c.center.y, c.center.z);
	Vec3Props[CENTER].set(v.x, v.y, v.z);
	v.set(c.size.x, c.size.y, c.size.z);
	Vec3Props[SIZE].set(v.x, v.y, v.z);
	min=c.min;
	max=c.max;
	this->previous = c.previous;

}


ProximitySensor::ProximitySensor(void):
			BoolProps(COUNT_BOOLPROP),
			Vec3Props(COUNT_VEC3PROP)
{
	m_Name = "";
	BoolProps[ENABLED] = false;
	Vec3Props[CENTER].set(0,0,0);
	Vec3Props[SIZE].set(0,0,0);
	min.set(0,0,0);
	max.set(0,0,0);
	previous = false;
}


ProximitySensor::~ProximitySensor(void){
	this->removeProximityListener();
}


bool ProximitySensor::getEnabled(void){
	return enabled;
}


vec3 ProximitySensor::getCenter(void){
	return center;
}


vec3 ProximitySensor::getSize(void){
	return size;
}


bool ProximitySensor::isNear(vec3 p){

	if( abs(Vec3Props[CENTER].x-p.x) <= Vec3Props[SIZE].x * 0.5f && 
		abs(Vec3Props[CENTER].y-p.y) <= Vec3Props[SIZE].y * 0.5f && 
		abs(Vec3Props[CENTER].z-p.z) <= Vec3Props[SIZE].z * 0.5f)
		return true;
	else
		return false;		
}


void ProximitySensor::addProximityListener(void){
	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->addListener("ACTIVATE",lst);
	EVENTMANAGER->addListener("DEACTIVATE",lst);
	EVENTMANAGER->addListener("PROXIMITY",lst);
}


void ProximitySensor::removeProximityListener(void){
	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->removeListener("ACTIVATE",lst);
	EVENTMANAGER->removeListener("DEACTIVATE",lst);
	EVENTMANAGER->removeListener("PROXIMITY",lst);
}


vec3 ProximitySensor::getMin(void){
	vec3 v;
	v.set(Vec3Props[CENTER].x - Vec3Props[SIZE].x * 0.5f,
		  Vec3Props[CENTER].y - Vec3Props[SIZE].y * 0.5f, 
		  Vec3Props[CENTER].z - Vec3Props[SIZE].z * 0.5f);
	return v;
}


vec3 ProximitySensor::getMax(void){
	vec3 v;
	v.set(Vec3Props[CENTER].x + Vec3Props[SIZE].x * 0.5f,
		  Vec3Props[CENTER].y + Vec3Props[SIZE].y * 0.5f, 
		  Vec3Props[CENTER].z + Vec3Props[SIZE].z * 0.5f);
	return v;
}


void ProximitySensor::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt){

	//if(strcmp(evt->getReceiver(), &name[0])!=0 && strcmp(evt->getReceiver(), "")!=0) return;

	if ("ACTIVATE" == eventType)
		BoolProps[ENABLED] = true;
	else if ("DEACTIVATE" == eventType)
		BoolProps[ENABLED] = false;

	else if ("PROXIMITY" == eventType && true == BoolProps[ENABLED]) {

		vec3 *ve=(vec3 *)evt->getData();
		vec3 v=*ve;

		bool b = isNear(v);

		if(b && !previous) {
			time_t t;
			t=time(NULL);
			nau::event_::IEventData *e= nau::event_::EventFactory::create("Float");
			e->setData(&t);
			EVENTMANAGER->notifyEvent("ENTER_TIME",m_Name,"", e);
			delete e;
		}
		else if (!b && previous) {
			time_t t;
			t=time(NULL);
			nau::event_::IEventData *e= nau::event_::EventFactory::create("Float");
			e->setData(&t);
			EVENTMANAGER->notifyEvent("EXIT_TIME",m_Name, "", e);
			delete e;
		}
		previous = b;
	}
}


const std::string &
ProximitySensor::getBoolPropNames(unsigned int i) 
{
	if (i < COUNT_BOOLPROP)
		return BoolPropNames[i];
	else
		return(emptyString);
}


const std::string &
ProximitySensor::getVec3PropNames(unsigned int i) 
{
	if (i < COUNT_VEC3PROP)
		return Vec3PropNames[i];
	else
		return(emptyString);
}


void 
ProximitySensor::setBool(unsigned int prop, bool value)
{
	if (prop < COUNT_BOOLPROP)
		BoolProps[prop] = value;	
}


void 
ProximitySensor::setVec3(unsigned int prop, nau::math::vec3 &value)
{
	if (prop < COUNT_VEC3PROP)
		Vec3Props[prop].set(value.x, value.y, value.z);	
}


nau::math::vec3 & 
ProximitySensor::getVec3(unsigned int prop)
{
	assert(prop < COUNT_VEC3PROP);

	return (Vec3Props[prop]);
}


bool 
ProximitySensor::getBool(unsigned int prop)
{
	assert(prop < COUNT_BOOLPROP);

	return (BoolProps[prop]);	
}





void ProximitySensor::init(){
	//this->name=name;
	//this->enabled=enabled;

	//std::string sub;
	//float v[3];
	//char *c;	
	//int pos=0;
	//int pos1,pos2;

	//pos1=str.find("center:",pos);
	//if(pos1!=-1){
	//	pos1=pos1+7;
	//	for(int i=0; i<3;i++){
	//		pos2=str.find(",",pos1);
	//		sub=str.substr(pos1,pos2-pos1);
	//		c=&sub[0];
	//		v[i]=atof(c);
	//		pos1=pos2+1;
	//	}
	//	this->center.set(v[0],v[1],v[2]);
	//}

	//pos1=str.find("size:",pos);
	//if(pos1!=-1){
	//	pos1=pos1+5;
	//	for(int i=0; i<3;i++){
	//		pos2=str.find(",",pos1);
	//		sub=str.substr(pos1,pos2-pos1);
	//		c=&sub[0];
	//		v[i]=atof(c);
	//		pos1=pos2+1;
	//	}
	//	this->size.set(v[0],v[1],v[2]);
	//}

	//this->min=getMin();
	//this->max=getMax();
	this->addProximityListener();
}