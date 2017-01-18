#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "nau/event/iListener.h"
#include "nau/math/vec3.h"
#include "nau/math/vec4.h"

#include <map>

using namespace nau::math;


namespace nau
{
	namespace event_
	{
		class Interpolator: public IListener
		{

		public:
			typedef enum {BOOL, FLOAT, VEC3, COUNT_PROPTYPE} PropTypes;

		protected:
			nau::math::vec3 emptyVec3;
			std::string emptyString;
			std::string m_Name;

			std::map<float, vec4> m_KeyFrames;

		public:
			Interpolator(void): m_KeyFrames() {};
			~Interpolator(void){};

			void setName(std::string &name) {m_Name = name;};
			std::string &getName() {return m_Name;}

			virtual void addKeyFrame(float key, vec4 &value) {m_KeyFrames[key] = value;};

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