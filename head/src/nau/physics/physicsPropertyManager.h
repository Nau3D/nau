#ifndef _NAU_PHYSICS_PROP_MANAGER_H
#define _NAU_PHYSICS_PROP_MANAGER_H

#include "nau/physics/iPhysicsPropertyManager.h"
#include "nau/physics/physicsManager.h"

#include <string>


namespace nau
{
	namespace physics
	{
		class PhysicsPropertyManager: public IPhysicsPropertyManager
		{
		public:

			static PhysicsPropertyManager* GetInstance();
			PhysicsPropertyManager::~PhysicsPropertyManager();

			float getMaterialFloatProperty(const std::string &material, const std::string &property);
			float *getMaterialVec4Property(const std::string &material, const std::string & property);

			void setMaterialFloatProperty(const std::string &material, const std::string &property, float value);
			void setMaterialVec4Property(const std::string &material, const std::string &property, float *value);

			float getGlobalFloatProperty(const std::string &property);
			float *getGlobalVec4Property(const std::string &property);

			void setGlobalFloatProperty(const std::string &property, float value);
			void setGlobalVec4Property(const std::string &property, float *value);

		protected:
			PhysicsPropertyManager::PhysicsPropertyManager();
			PhysicsManager *m_PhysicsManager;
			static PhysicsPropertyManager *Instance;
		};
	};
};
#endif