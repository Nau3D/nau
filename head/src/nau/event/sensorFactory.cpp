#include "nau/event/sensorFactory.h"
//#include "nau/event/touchSensor.h"
#include "nau/event/proximitySensor.h"
#include "nau/event/timeSensor.h"

using namespace nau::event_;
using namespace std;

nau::event_::Sensor* 
SensorFactory::create (std::string type)
{
	if ("ProximitySensor" == type) {
		return new nau::event_::ProximitySensor;
	}

/*	if("TouchSensor" == type) {
		return new nau::event_::TouchSensor;
	}
*/
	if("TimeSensor" == type) {
		return new nau::event_::TimeSensor;
	}

	return 0;
}

bool
SensorFactory::validate(std::string type)
{
	if (("ProximitySensor" == type) || ("TimeSensor" == type))
		return true;
	else
		return false;

}