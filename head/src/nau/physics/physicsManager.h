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
			friend class PhysicsMaterial;
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


			void setPropf(FloatProperty p, float value);
			void setPropf4(Float4Property p, vec4 &value);


			
			PhysicsManager::~PhysicsManager();

		protected:
		
			PhysicsManager::PhysicsManager();
			
			void applyMaterialFloatProperty(const std::string &matName, const std::string &property, float value);
			void applyMaterialVec4Property(const std::string &matName, const std::string &property, float *value);

			IPhysics *loadPlugin();

			void applyGlobalFloatProperty(const std::string &property, float value);
			void applyGlobalVec4Property(const std::string &property, float *value);

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