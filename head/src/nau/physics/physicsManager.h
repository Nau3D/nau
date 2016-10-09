/* 
Main class for physics in NAU.
*/

#ifndef _NAU_PHYSICSMANAGER_H
#define _NAU_PHYSICSMANAGER_H

#include "nau/attributeValues.h"
#include "nau/enums.h"
#include "nau/math/data.h"
#include "nau/physics/iPhysics.h"
#include "nau/physics/physicsMaterial.h"
#include "nau/scene/iScene.h"
#include "nau/material/iBuffer.h"

#include <map>
#include <string>
#include <vector>

namespace nau 
{
	namespace physics 
	{
		class PhysicsPropertyManager;

		class PhysicsManager: public AttributeValues, public nau::event_::IListener
		{
			friend class PhysicsMaterial;
		public:
		
			FLOAT_PROP(TIME_STEP, 0);

			FLOAT4_PROP(CAMERA_POSITION, 0);
			FLOAT4_PROP(CAMERA_DIRECTION, 1); 
			FLOAT4_PROP(CAMERA_UP, 2);

			FLOAT_PROP(CAMERA_RADIUS, 1);
			FLOAT_PROP(CAMERA_HEIGHT, 2);

			static AttribSet Attribs;

			static PhysicsManager* GetInstance();
			
			void updateProps();
			void update();
			void build();

			void clear();
			
			void addScene(nau::scene::IScene *aScene, const std::string &matName);

			void cameraAction(Camera * camera, std::string action, float * value);

			PhysicsMaterial &getMaterial(const std::string &name);
			void getMaterialNames(std::vector<std::string> *);

			void setPropf(FloatProperty p, float value);
			void setPropf4(Float4Property p, vec4 &value);
			
			void eventReceived(const std::string &sender, const std::string &eventType,	const std::shared_ptr<IEventData> &evt);
			std::string& getName();

			PhysicsManager::~PhysicsManager();

		protected:
		
			PhysicsManager::PhysicsManager();

			void applyMaterialFloatProperty(const std::string &matName, const std::string &property, float value);
			void applyMaterialVec4Property(const std::string &matName, const std::string &property, float *value);

			void applyGlobalFloatProperty(const std::string &property, float value);
			void applyGlobalVec4Property(const std::string &property, float *value);

			IPhysics *loadPlugin();

			static PhysicsManager *PhysManInst;

			PhysicsPropertyManager *m_PropertyManager;

			IPhysics *m_PhysInst;
			
			static bool Init();
			static bool Inited;
			bool m_Built;
			bool hasCamera;

			std::map<std::string, PhysicsMaterial> m_MatLib;
			std::map<nau::scene::IScene *, std::string> m_Scenes;
		};
	};
};

#endif