#ifndef SENSORFACTORY_H
#define SENSORFACTORY_H

#include <nau/event/sensor.h>
#include <string>

namespace nau
{
	namespace event_
	{
		class SensorFactory
		{
		public:
			static nau::event_::Sensor* create (std::string type);
			static bool validate(std::string type);
		private:
			SensorFactory(void) {};
			~SensorFactory(void) {};
		};
	};
};
#endif