#ifndef _NAU_PHYSICS_H
#define _NAU_PHYSICS_H

#include "nau/math/data.h"

#include <map>
#include <string>


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
				PARTICLES
			} SceneType;

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

			virtual void update() = 0;
			virtual void build() = 0;
			
			virtual void setSceneType(const std::string &scene, SceneType type) = 0;

			virtual void applyFloatProperty(const std::string &scene, const std::string &property, float value) = 0;
			virtual void applyVec4Property(const std::string &scene, const std::string &property, float *value) = 0;
			
			virtual void applyGlobalFloatProperty(const std::string &property, float value) = 0;
			virtual void applyGlobalVec4Property(const std::string &property, float *value) = 0;

			virtual void setScene(const std::string &scene, float *vertices, unsigned int *indices, float *transform) = 0;
			
			virtual float *getSceneTransform(const std::string &scene) = 0;
			virtual void setSceneTransform(const std::string &scene, float *transform) = 0;

			virtual void getGlobalProperties(std::map < std::string, Prop> *) = 0;
			virtual void getMaterialProperties(std::map < std::string, Prop> *) = 0;

		protected:

			typedef struct {
				SceneType sceneType;
				float *vertices;
				unsigned int *indices;
				float *transform;
			} SceneProps;

			std::map<std::string, SceneProps> m_Scenes;
		};
	};
};

#endif