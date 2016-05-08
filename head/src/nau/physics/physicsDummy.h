#ifndef _NAU_PHYSICS_DUMMY_H
#define _NAU_PHYSICS_DUMMY_H


#include "nau/math/data.h"
#include "nau/physics/iPhysics.h"

#include <map>
#include <string>

namespace nau 
{
	namespace physics 
	{
		class PhysicsDummy: public IPhysics 
		{
		public:
			virtual void update();
			virtual void build();
			
			virtual void setSceneType(const std::string &scene, SceneType type);

			virtual void applyFloatProperty(const std::string &scene, const std::string &property, float value);
			virtual void applyVec4Property(const std::string &scene, const std::string &property, float *value);

			virtual void applyGlobalFloatProperty(const std::string &property, float value);
			virtual void applyGlobalVec4Property(const std::string &property, float *value);

			virtual void setScene(const std::string &scene, float *vertices, unsigned int *indices, float *transform);
			
			virtual float *getSceneTransform(const std::string &scene);
			virtual void setSceneTransform(const std::string &scene, float *transform);

			std::map < std::string, Prop> &getGlobalProperties();
			std::map < std::string, Prop> &getMaterialProperties();
		};
	};
};

#endif