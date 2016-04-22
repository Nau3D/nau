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

			virtual void update() = 0;
			virtual void build() = 0;
			
			virtual void setSceneType(std::string &scene, SceneType type) = 0;

			virtual void applyProperty(std::string &property, nau::math::Data *value) = 0;
			
			virtual void setSceneVertices(std::string &scene, float *vertices) = 0;
			virtual void setSceneIndices(std::string &scene, unsigned int *indices) = 0;
			
			virtual float *getSceneTransform(std::string &scene) = 0;
			virtual void setSceneTransform(std::string &scene, float *transform) = 0;

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