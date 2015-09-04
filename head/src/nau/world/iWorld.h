#ifndef IWORLD_H
#define IWORLD_H

#include "nau/scene/iScene.h"
#include "nau/math/vec3.h"

namespace nau
{
	namespace world
	{
		class IWorld
		{
		public:
		        
		    virtual ~IWorld() {};
		  
	        virtual void update (void) = 0;
			virtual void build (void) = 0;
			virtual void setScene (nau::scene::IScene *aScene) = 0;

			virtual void _add (float mass, nau::scene::SceneObject *aObject, std::string name, nau::math::vec3 aVec) = 0;
			virtual void setKinematic (std::string name) = 0;
			virtual void setDynamic (std::string name) = 0;

			virtual void disableObject (std::string name) = 0;
			virtual void enableObject (std::string name) = 0;


			virtual void setVelocity (std::string name, nau::math::vec3 vel) = 0;
		};
	};
};
#endif
