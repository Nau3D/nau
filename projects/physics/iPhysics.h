#ifndef _NAU_PHYSICS_H
#define _NAU_PHYSICS_H


namespace nau 
{
	namespace physics 
	{
		class IPhysics 
		{
		
			IPhysics* create();

			void update() = 0;
			void build() = 0;
			
			void applyProperty(std::string &property, Data *value) = 0;
			
			void setSceneVertices(std::string &scene, float *vertices) = 0;
			void setSceneIndex(std::string &scene, unsigned int *indices) = 0;
			
			float *getSceneTransform(std::string &sceneName);
			void setSceneTransform(std::string 6sceneName, float *transform);
		}
	}
}

#endif