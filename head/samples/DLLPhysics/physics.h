#ifndef _PHYSICS_H
#define _PHYSICS_H



#include "nau/physics/iPhysics.h"
#include "nau/physics/iPhysicsPropertyManager.h"

#include <map>
#include <string>


class Physics : public nau::physics::IPhysics
{
protected:

public:

	static Physics *Create();
	Physics();
	~Physics(void);

	virtual void setPropertyManager(nau::physics::IPhysicsPropertyManager *pm);

	void update();
	void build();

	void setSceneType(const std::string &scene, SceneType type);

	void applyFloatProperty(const std::string &scene, const std::string &property, float value);
	void applyVec4Property(const std::string &scene, const std::string &property, float *value);

	void applyGlobalFloatProperty(const std::string &property, float value);
	void applyGlobalVec4Property(const std::string &property, float *value);

	void setScene(const std::string &scene, const std::string &material, int nbVertices, float *vertices, int nbIndices, unsigned int *indices, float *transform);

	float *getSceneTransform(const std::string &scene);
	void setSceneTransform(const std::string &scene, float *transform);

	void setCameraAction(const std::string &scene, const std::string &action, float * value);
	std::map<std::string, float*> * getCameraPositions();

	std::map<std::string, nau::physics::IPhysics::Prop> &getGlobalProperties();
	std::map<std::string, nau::physics::IPhysics::Prop> &getMaterialProperties();

	virtual std::vector<float> * getDebug();

};

extern "C" {

	__declspec(dllexport) void *createPhysics();
	__declspec(dllexport) void init();
	__declspec(dllexport) char *getClassName();
	__declspec(dllexport) void deletePhysics();
}

#endif //DEPTHMAPPASS_H
