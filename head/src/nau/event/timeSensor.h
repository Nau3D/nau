#ifndef TIMESENSOR_H
#define TIMESENSOR_H

#include "nau/event/eventFloat.h"
#include "nau/event/sensor.h"
#include <vector>

namespace nau
{
	namespace event_
	{
			class TimeSensor: public Sensor
			{

			public:

				typedef enum {ENABLED, LOOP, COUNT_BOOLPROP} BoolProp;
				typedef enum {SECONDS_TO_START, CYCLE_INTERVAL, COUNT_FLOATPROP} FloatProp;
				static const std::string FloatPropNames[COUNT_FLOATPROP];
				static const std::string BoolPropNames[COUNT_BOOLPROP];

			protected:

				std::vector<bool> BoolProps;
				std::vector<float> FloatProps;

			private:
				float startTime;
				float stopTime;
				float fraction;

			public:
				TimeSensor(std::string name, bool enabled, int secondsToStart, 
					int cycleInterval, bool loop); 
				TimeSensor(const TimeSensor &c);
				TimeSensor(void);
				~TimeSensor(void);

				bool getEnabled(void);
				int getSecondsToStart(void);
				bool getLoop(void);
				float getCycleInterval(void);

				void removeTimeListener(void);
				void addTimeListener(void);
				void eventReceived(const std::string &sender, const std::string &eventType, 
					const std::shared_ptr<IEventData> &evt);

				void init();

				const std::string &getBoolPropNames(unsigned int i);
				const std::string &getFloatPropNames(unsigned int i);

				void setBool(unsigned int prop, bool value);
				void setFloat(unsigned int prop, float value);

				float getFloat(unsigned int prop);
				bool getBool(unsigned int prop);

				unsigned int getFloatPropCount() {return COUNT_FLOATPROP;};
				unsigned int getBoolPropCount() {return COUNT_BOOLPROP;};

			};
	};
};
#endif