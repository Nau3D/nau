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
			
			virtual void setSceneType(std::string &scene, SceneType type);

			virtual void applyProperty(std::string &property, nau::math::Data *value);
			
			virtual void setSceneVertices(std::string &scene, float *vertices);
			virtual void setSceneIndices(std::string &scene, unsigned int *indices);
			
			virtual float *getSceneTransform(std::string &scene);
			virtual void setSceneTransform(std::string &scene, float *transform);

		};
	};
};

#endif