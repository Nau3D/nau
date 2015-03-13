#ifndef TOUCHSENSOR_H
#define TOUCHSENSOR_H

#include "nau/event/iEventData.h"
#include "nau/event/sensor.h"
#include "nau/math/vec3.h"
#include "nau/scene/sceneobject.h"


using namespace nau::math;
using namespace nau::scene;

namespace nau
{
	namespace event_
	{
			class TouchSensor: public Sensor
			{
			public:
				std::string name;
				ISceneObject *object;
				bool enabled;
				nau::math::vec3 colorSensitive;


				TouchSensor(std::string name, ISceneObject *object, bool enabled, nau::math::vec3 colorSensitive); 
				TouchSensor(const TouchSensor &c);
				TouchSensor(void);
				~TouchSensor(void);

				void setTouchSensor(std::string name, ISceneObject *object, bool enabled, nau::math::vec3 colorSensitive);
				std::string &getName(void);
				ISceneObject *getObject(void);
				bool getEnabled(void);
				nau::math::vec3 getColorSensitive(void);
				bool isSelected(int x, int y);
				void removeTouchListener(void);
				void addTouchListener(void);
				void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);
				void init(std::string name, bool enabled, std::string str);
			};
	};
};
#endif