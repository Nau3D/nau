#ifndef _NAU_IPHYSICS_PROP_MANAGER_H
#define _NAU_IPHYSICS_PROP_MANAGER_H

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