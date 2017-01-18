#ifndef _NAU_PHYSICSMANAGER_H
#define _NAU_PHYSICSMANAGER_H

#include "nau/physics/physicsMaterial.h"
#include "nau/scene/iScene.h"

namespace nau 
{
	namespace physics 
	{
		class PhysicsManager: public AttributeValues
		{
		
		public:
		
			FLOAT4_PROP(GRAVITY, 0);
			
			static PhysicsManager* GetInstance();
			
			void registerPlugin(IPhysics *p);
			void update();
			void build();
			
			// to allow pluggins to add properties
			void addGlobalProperty(std::string &property, Enums::Data dataType);
			void addMaterialProperty(std::string &material, std::string &property, Enums::Data dataType);
			
			void setMaterialProperty(std::string &material, std::string &property, Data *value);
			void *getMaterialProperty(std::string &material, st::string &property);
			
			void addScene(nau::scene::IScene *aScene);
			
			void createMaterial(const std::string &name);
			physicsMaterial &getMaterial(const std::string &name);
			
		protected:
		
			PhysicsManager::PhysicsManager();
			~PhysicsManager::PhysicsManager();
			
			
			PhysicsManager *PhysManInst;
			IPhysics *PhysInst;
			
			std::map<std::string, physicsMaterial> matLib;
			
			float *getSceneTransform(std::string &sceneName);
			void setSceneTransform(std::string 6sceneName, float *transform);
		}
	}
}

#endif