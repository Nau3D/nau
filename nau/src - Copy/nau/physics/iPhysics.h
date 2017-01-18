/*
To develop a plugin this interface must be implemented
*/

#ifndef PHYSICS_H
#define PHYSICS_H

#include "nau/math/data.h"
#include "nau/physics/iPhysicsPropertyManager.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nau 
{
	namespace physics 
	{
		class IPhysics 
		{
		public:

			typedef enum {
				STATIC,
				RIGID,
				CLOTH,
				PARTICLES,
				CHARACTER,
				DEBUG
			} SceneType;

			// Bounding shapes
			typedef enum {
				CUSTOM,
				BOX,
				SPHERE,
				CAPSULE
			} SceneShape;

			// data types allowed for communication between Nau3D and the plugin
			typedef enum {
				FLOAT,
				VEC4
			} PropTypes;

			typedef struct Props{
				PropTypes propType;
				float x, y, z, w;

				Props() {
					propType = FLOAT;
					x = 0.0f; y = 00.f; z = 0.0f; w = 0.0f;
				};

				Props(PropTypes pt, float xx, float yy, float zz, float ww) {
					propType = pt;
					x = xx; y = yy; z = zz; w = ww;
				};

				Props(PropTypes pt, float value) {
					propType = pt;
					x = value;
				};

			} Prop;

			typedef struct {
				SceneShape sceneShape;
				float * max;
				float * min;
			} BoundingVolume;

			typedef enum {
				GT,
				LT,
				EGT,
				ELT,
				EQ,
				NONE
			} SceneCondition;

			virtual void setPropertyManager(IPhysicsPropertyManager *pm) = 0;

			virtual void update() = 0;
			virtual void build() = 0;
			
			void setSceneType(const std::string &scene, SceneType type) { m_Scenes[scene].sceneType = type; };

			void setSceneShape(const std::string &scene, SceneShape shape, float * min, float * max) {
				m_Scenes[scene].boundingVolume.sceneShape = shape;
				m_Scenes[scene].boundingVolume.min = min;
				m_Scenes[scene].boundingVolume.max = max;
			};

			void setSceneCondition(const std::string &scene, SceneCondition condition) { m_Scenes[scene].sceneCondition = condition; };

			virtual void applyFloatProperty(const std::string &scene, const std::string &property, float value) = 0;
			virtual void applyVec4Property(const std::string &scene, const std::string &property, float *value) = 0;
			
			virtual void applyGlobalFloatProperty(const std::string &property, float value) = 0;
			virtual void applyGlobalVec4Property(const std::string &property, float *value) = 0;

			virtual void setScene(const std::string &scene, const std::string &material, int numVertices, float *vertices, int numIndices, unsigned int *indices, float *transform) = 0;

			virtual float *getSceneTransform(const std::string &scene) = 0;
			virtual void setSceneTransform(const std::string &scene, float *transform) = 0;
			
			virtual void setCameraAction(const std::string &scene, const std::string &action, float * value) = 0;
			virtual std::map<std::string, float*> * getCameraPositions() = 0;

			virtual std::map<std::string, nau::physics::IPhysics::Prop> &getGlobalProperties() = 0;
			virtual std::map<std::string, nau::physics::IPhysics::Prop> &getMaterialProperties() = 0;

			virtual std::vector<float> * getDebug() = 0;

		protected:

			typedef struct {
				SceneType sceneType;
				int nbVertices;
				float * vertices;
				int nbIndices;
				unsigned int * indices;
				float * transform;
				std::string material;
				BoundingVolume boundingVolume;
				SceneCondition sceneCondition;
			} SceneProps;

			std::map<std::string, SceneProps> m_Scenes;
			std::map<std::string, Prop> m_GlobalProps;
			std::map<std::string, Prop> m_MaterialProps;

			IPhysicsPropertyManager *m_PropertyManager;
		};
	};
};

#endif