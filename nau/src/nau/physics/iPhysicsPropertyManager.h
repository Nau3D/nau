/*
This interface allows the plugin to receive an instance to read/write physics material properties. 

The plugin receives an instance of PropertyManager from NAU, a subclass of IPhysicsPropertyManager with 
a proper implementation to allow properties to move backwards and forwards
*/

#ifndef NAU_IPHYSICS_PROP_MANAGER_H
#define NAU_IPHYSICS_PROP_MANAGER_H

#include <string>


namespace nau
{
	namespace physics
	{
		class IPhysicsPropertyManager
		{
		public:

			virtual float getMaterialFloatProperty(const std::string &material, const std::string &property) { return 0.0f; };
			virtual float *getMaterialVec4Property(const std::string &material, const std::string &property) { return NULL; };

			virtual void setMaterialFloatProperty(const std::string &material, const std::string &property, float value) {};
			virtual void setMaterialVec4Property(const std::string &material, const std::string &property, float *value) {};

			virtual float getGlobalFloatProperty(const std::string &property) { return 0.0f; };
			virtual float *getGlobalVec4Property(const std::string &property) { return NULL; };

			virtual void setGlobalFloatProperty(const std::string &property, float value) {};
			virtual void setGlobalVec4Property(const std::string &property, float *value) {};
		};
	};
};

#endif