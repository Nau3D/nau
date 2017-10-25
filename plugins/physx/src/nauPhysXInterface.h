#ifndef _NAUPHYSXINTERFACE_H
#define _NAUPHYSXINTERFACE_H

#include "nau/physics/iPhysics.h"
#include <map>
#include <string>
#include "physXWorldManager.h"

class NauPhysXInterface : public nau::physics::IPhysics {

private:
	PhysXWorldManager * worldManager;

public:

	static NauPhysXInterface *Create();
	NauPhysXInterface();
	~NauPhysXInterface();

	virtual void setPropertyManager(nau::physics::IPhysicsPropertyManager *pm);

	void update();
	void build();

	void applyFloatProperty(const std::string &scene, const std::string &property, float value);
	void applyVec4Property(const std::string &scene, const std::string &property, float *value);

	void applyGlobalFloatProperty(const std::string &property, float value);
	void applyGlobalVec4Property(const std::string &property, float *value);

	void setScene(const std::string &scene, const std::string & material, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform);

	float *getSceneTransform(const std::string &scene);
	void setSceneTransform(const std::string &scene, float *transform);

	void setCameraAction(const std::string &scene, const std::string &action, float * value);
	std::map<std::string, float*> * getCameraPositions() { return worldManager->getCameraPositions(); };

	std::vector<float> * getDebug(); 

	std::map<std::string, nau::physics::IPhysics::Prop> &getGlobalProperties();
	std::map<std::string, nau::physics::IPhysics::Prop> &getMaterialProperties();
};

extern "C" {

	__declspec(dllexport) void *createPhysics();
	__declspec(dllexport) void init();
	__declspec(dllexport) char *getClassName();
	__declspec(dllexport) void deletePhysics();
}

#endif