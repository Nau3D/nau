#ifndef ISENSOR_H
#define ISENSOR_H

#include "nau/math/vec3.h"
#include "nau/event/iListener.h"

#include <string>

namespace nau
{
	namespace event_
	{
		class Sensor: public IListener
		{

		protected:
			nau::math::vec3 emptyVec3;
			std::string emptyString;
			std::string m_Name;

		public:

			typedef enum {BOOL, FLOAT, VEC3, COUNT_PROPTYPE} PropTypes;


			Sensor(void){};
			~Sensor(void){};

			//virtual void setProximitySensor(std::string name, bool enabled, nau::math::vec3 center, nau::math::vec3 size){};
			//virtual void setTimeSensor(std::string name, bool enabled, int secondsToStart, int cicleInterval, bool loop){};
			//virtual void setTouchSensor(std::string name, ISceneObject *object, bool enabled, nau::math::vec3 colorSensitive){};
			
			void setName(std::string &name) {m_Name = name;};
			std::string &getName() {return m_Name;}
			virtual void init(){};

			virtual const std::string &getBoolPropNames(unsigned int i) {return emptyString;};
			virtual const std::string &getFloatPropNames(unsigned int i) {return emptyString;};
			virtual const std::string &getVec3PropNames(unsigned int i) {return emptyString;};

			virtual void setBool(unsigned int prop, bool value) {};
			virtual void setVec3(unsigned int prop, nau::math::vec3 &value) {};
			virtual void setFloat(unsigned int prop, float value) {};

			virtual float getFloat(unsigned int prop) {return 0.0f;};
			virtual bool getBool(unsigned int prop) {return false;};
			virtual nau::math::vec3 &getVec3(unsigned int prop) {return emptyVec3;}

			virtual unsigned int getFloatPropCount() {return 0;};
			virtual unsigned int getBoolPropCount() {return 0;};
			virtual unsigned int getVec3PropCount() {return 0;};

		};

	};
};


#endif