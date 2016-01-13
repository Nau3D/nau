#ifndef BULLETWORLD_H
#define BULLETWORLD_H

#include <btBulletDynamicsCommon.h>
#include "nau/world/iWorld.h"
#include "nau/scene/iScene.h"

namespace nau
{
	namespace world
	{
		class BulletWorld :
			public nau::world::IWorld
		{
		private:
			static const int maxProxies = 32766;

			nau::scene::IScene *m_pScene;
			btDynamicsWorld *m_pDynamicsWorld;

			std::map <std::string, btRigidBody*> m_RigidBodies;

		public:
			BulletWorld(void);

			void update (void);
			void build (void);
			void setScene (nau::scene::IScene *aScene);

			void _add (float mass, std::shared_ptr<nau::scene::SceneObject> &aObject, std::string name, nau::math::vec3 aVec);
			void setKinematic (std::string name);
			void setDynamic (std::string name);

			void disableObject (std::string name);
			void enableObject (std::string name);

			void setVelocity (std::string name, nau::math::vec3 vel);
		
		public:
			~BulletWorld(void);
		};
	};
};
#endif
