#ifndef _PHYSICS_H
#define _PHYSICS_H



#include "nau/physics/iPhysics.h"

#include <map>
#include <string>


class Physics : public nau::physics::IPhysics
{
protected:

public:

	static Physics *Create();
	Physics();
	~Physics(void);

	void update();
	void build();

	void setSceneType(const std::string &scene, SceneType type);

	void applyFloatProperty(const std::string &scene, const std::string &property, float value);
	void applyVec4Property(const std::string &scene, const std::string &property, float *value);

	void applyGlobalFloatProperty(const std::string &property, float value);
	void applyGlobalVec4Property(const std::string &property, float *value);

	void setScene(const std::string &scene, float *vertices, unsigned int *indices, float *transform);

	float *getSceneTransform(const std::string &scene);
	void setSceneTransform(const std::string &scene, float *transform);

	std::map < std::string, Prop> &getGlobalProperties();
	std::map < std::string, Prop> &getMaterialProperties();
};

extern "C" {

	__declspec(dllexport) void *createPhysics();
	__declspec(dllexport) void init();
	__declspec(dllexport) char *getClassName();
}

#endif //DEPTHMAPPASS_H
