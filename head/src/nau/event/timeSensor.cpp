#include <nau/event/timeSensor.h>
#include <nau/event/eventFactory.h>
#include <math.h>
#include <time.h>
#include <nau.h>

using namespace nau::event_;

const std::string TimeSensor::FloatPropNames[] = {"secondsToStart","cycleInterval"};
const std::string TimeSensor::BoolPropNames[] = {"enabled", "loop"};


TimeSensor::TimeSensor(std::string aname, bool enabled, int secondsToStart, int cycleInterval, bool loop): 
		BoolProps(COUNT_BOOLPROP),
		FloatProps(COUNT_FLOATPROP)
	
{
	// time is defined in seconds
	float t;

	m_Name = aname;

	t=(float)clock()/CLOCKS_PER_SEC;

	BoolProps[ENABLED] = enabled;
	BoolProps[LOOP] = loop;

	FloatProps[SECONDS_TO_START] = secondsToStart;
	FloatProps[CYCLE_INTERVAL] = cycleInterval;

	this->startTime = t + secondsToStart;
	this->stopTime = startTime+cycleInterval;
	this->fraction = 0;

	this->addTimeListener();
}


TimeSensor::TimeSensor(const TimeSensor &c): 
		BoolProps(COUNT_BOOLPROP),
		FloatProps(COUNT_FLOATPROP),
		startTime(c.startTime),
		stopTime(c.stopTime),
		fraction(c.fraction)

{
	m_Name = c.m_Name;
	BoolProps[ENABLED] = c.BoolProps[ENABLED];
	BoolProps[LOOP] = c.BoolProps[LOOP];

	FloatProps[SECONDS_TO_START] = c.FloatProps[SECONDS_TO_START];
	FloatProps[CYCLE_INTERVAL] = c.FloatProps[CYCLE_INTERVAL];
}


TimeSensor::TimeSensor(void): 

		BoolProps(COUNT_BOOLPROP),
		FloatProps(COUNT_FLOATPROP),
		startTime(0),
		stopTime(0),
		fraction(0)
{
	m_Name = "";
	BoolProps[ENABLED] = false;
	BoolProps[LOOP] = false;

	FloatProps[SECONDS_TO_START] = 0.0f;
	FloatProps[CYCLE_INTERVAL] = 0.0f;
}


TimeSensor::~TimeSensor(void){

	removeTimeListener();
}


bool 
TimeSensor::getEnabled(void){

	return BoolProps[ENABLED];
}


int 
TimeSensor::getSecondsToStart(void){

	return FloatProps[SECONDS_TO_START];
}


bool 
TimeSensor::getLoop(void){

	return BoolProps[LOOP];
}


float 
TimeSensor::getCycleInterval(void){

	return FloatProps[CYCLE_INTERVAL];
}


void 
TimeSensor::removeTimeListener(void){

	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->removeListener("ACTIVATE",lst);
	EVENTMANAGER->removeListener("DEACTIVATE",lst);
	EVENTMANAGER->removeListener("FRAME_BEGIN",lst);

}


void TimeSensor::addTimeListener(void){
	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->addListener("ACTIVATE",lst);
	EVENTMANAGER->addListener("DEACTIVATE",lst);
	EVENTMANAGER->addListener("FRAME_BEGIN",lst);
}


void TimeSensor::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt){
	
	float t;
	t = (float)clock()/CLOCKS_PER_SEC;

	if (eventType == "ACTIVATE") {

		BoolProps[ENABLED] = true;
		//nau::event_::IEventData *e = nau::event_::EventFactory::create("Float");
		e.setData(&t);
		startTime = t + FloatProps[SECONDS_TO_START];
		this->stopTime = startTime + FloatProps[CYCLE_INTERVAL];
		EVENTMANAGER->notifyEvent("TIMESENSOR_IS_ACTIVE", m_Name, "", &e);
	}
	else if(eventType == "DEACTIVATE") {

		BoolProps[ENABLED] = false;
		//nau::event_::IEventData *e = nau::event_::EventFactory::create("Float");
		e.setData(&t);
		EVENTMANAGER->notifyEvent("TIMESENSOR_IS_FINISHED", m_Name, "", &e);
	}

	if(BoolProps[ENABLED]){

		if(t > stopTime){

			if(BoolProps[LOOP]){

				startTime = t;
				stopTime = startTime + FloatProps[CYCLE_INTERVAL];
			}
			else {
				BoolProps[ENABLED] = false;
				//nau::event_::IEventData *e = nau::event_::EventFactory::create("Float");
				e.setData(&t);
				EVENTMANAGER->notifyEvent("TIMESENSOR_IS_FINISHED", m_Name, "", &e);
				return;
			}
		}
		if(t > startTime){

			//nau::event_::IEventData *e = nau::event_::EventFactory::create("Float");
			
			float fr = (float)(t - startTime) / FloatProps[CYCLE_INTERVAL];
			e.setData(&fr);
			EVENTMANAGER->notifyEvent("TIMESENSOR_FRACTION_CHANGED", m_Name, "", &e);
		}
	}
}


const std::string &
TimeSensor::getBoolPropNames(unsigned int i) 
{
	if (i < COUNT_BOOLPROP)
		return BoolPropNames[i];
	else
		return(emptyString);
}


const std::string &
TimeSensor::getFloatPropNames(unsigned int i) 
{
	if (i < COUNT_FLOATPROP)
		return FloatPropNames[i];
	else
		return(emptyString);
}


void 
TimeSensor::setBool(unsigned int prop, bool value)
{
	if (prop < COUNT_BOOLPROP)
		BoolProps[prop] = value;	
}


void 
TimeSensor::setFloat(unsigned int prop, float value)
{
	if (prop < COUNT_FLOATPROP)
		FloatProps[prop] = value;	
}


float 
TimeSensor::getFloat(unsigned int prop)
{
	assert(prop < COUNT_FLOATPROP);

	return (FloatProps[prop]);
}


bool 
TimeSensor::getBool(unsigned int prop)
{
	assert(prop < COUNT_BOOLPROP);

	return (BoolProps[prop]);	
}


void 
TimeSensor::init(){

	float t;
	t=(float)clock()/CLOCKS_PER_SEC;

	startTime = t + FloatProps[SECONDS_TO_START];
	stopTime = startTime + FloatProps[CYCLE_INTERVAL];

	addTimeListener();
}