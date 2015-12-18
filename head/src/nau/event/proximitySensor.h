#ifndef PROXIMITYSENSOR_H
#define PROXIMITYSENSOR_H

#include "nau/event/iEventData.h"
#include "nau/event/sensor.h"
#include "nau/math/vec3.h"

#include <vector>

using namespace nau::math;

namespace nau
{
	namespace event_
	{
			class ProximitySensor: public Sensor
			{

			public:
				typedef enum {ENABLED, COUNT_BOOLPROP} BoolProp;
				typedef enum {CENTER, SIZE, COUNT_VEC3PROP} Vec3Prop;
				static const std::string Vec3PropNames[COUNT_VEC3PROP];
				static const std::string BoolPropNames[COUNT_BOOLPROP];

			protected:
				std::vector<bool> BoolProps;
				std::vector<nau::math::vec3> Vec3Props;

				bool enabled;
				nau::math::vec3 center;
				nau::math::vec3 size;
			private:
				nau::math::vec3 min;
				nau::math::vec3 max;
				bool previous;

			public:
				ProximitySensor(std::string name, bool enabled, nau::math::vec3 center, 
					nau::math::vec3 size); 
				ProximitySensor(const ProximitySensor &c);
				ProximitySensor(void);
				~ProximitySensor(void);

				bool getEnabled(void);
				nau::math::vec3 getCenter(void);
				nau::math::vec3 getSize(void);
				bool isNear(nau::math::vec3 p);
				void removeProximityListener(void);
				void addProximityListener(void);
				nau::math::vec3 getMin(void);
				nau::math::vec3 getMax(void);
				void eventReceived(const std::string &sender, const std::string &eventType, 
					const std::shared_ptr<IEventData> &evt);
				void init();

				const std::string &getBoolPropNames(unsigned int i);
				const std::string &getVec3PropNames(unsigned int i);
				void setBool(unsigned int prop, bool value);
				void setVec3(unsigned int prop, nau::math::vec3 &value);
				nau::math::vec3 &getVec3(unsigned int prop);
				bool getBool(unsigned int prop);
				unsigned int getBoolPropCount() {return COUNT_BOOLPROP;};
				unsigned int getVec3PropCount() {return COUNT_VEC3PROP;};
			};
	};
};
#endif