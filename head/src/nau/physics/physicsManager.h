#ifndef _NAU_PHYSICSMANAGER_H
#define _NAU_PHYSICSMANAGER_H

#include "nau/attributeValues.h"
#include "nau/enums.h"
#include "nau/math/data.h"
#include "nau/physics/iPhysics.h"
#include "nau/physics/physicsMaterial.h"
#include "nau/scene/iScene.h"

#include <map>
#include <string>
#include <vector>

namespace nau 
{
	namespace physics 
	{
		class PhysicsManager: public AttributeValues
		{
		
		public:
		
			FLOAT4_PROP(GRAVITY, 0);


			
			static AttribSet Attribs;

			static PhysicsManager* GetInstance();
			
			void updateProps();
			void update();
			void build();

			void clear();
			
			void addScene(nau::scene::IScene *aScene, const std::string &matName);
			
			PhysicsMaterial &getMaterial(const std::string &name);
			void getMaterialNames(std::vector<std::string> *);
			
			PhysicsManager::~PhysicsManager();

		protected:
		
			PhysicsManager::PhysicsManager();
			
			IPhysics *loadPlugin();

			static PhysicsManager *PhysManInst;
			IPhysics *m_PhysInst;
			
			static bool Init();
			static bool Inited;
			bool m_Built;

			std::map<std::string, PhysicsMaterial> m_MatLib;
			// map from scenes to material
			std::map<nau::scene::IScene *, std::string> m_Scenes;
		};
	};
};

#endif