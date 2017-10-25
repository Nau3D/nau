#ifndef _PHYSXCHARACTERMANAGER_H
#define _PHYSXCHARACTERMANAGER_H

#include <map>
#include <vector>
#include "PxPhysicsAPI.h"
#include "physXScene.h"

class PhysXCharacterManager : public physx::PxUserControllerHitReport {

public:
	static int nextControllerIndex;

	PhysXCharacterManager(physx::PxScene * world);
	~PhysXCharacterManager();

	void update(physx::PxVec3 gravity = physx::PxVec3(0.0f));
	void addCharacter(const std::string &scene, physx::PxMaterial * material, physx::PxVec3 up = physx::PxVec3(0.0f, 1.0f, 0.0f));
	void addCamera(const std::string & scene, physx::PxVec3 position, physx::PxVec3 up, float pace, float minPace, float hitMagnitude, float timeStep, float stepOffset, float mass, float radius, float height, physx::PxMaterial * material);
	void move(const std::string & scene, physx::PxVec3 gravity = physx::PxVec3(0.0f));
	void move(const std::string & scene, float * transform, physx::PxVec3 gravity = physx::PxVec3(0.0f));
	void createInfo(const std::string &scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform);

	void setDirection(std::string scene, physx::PxVec3 dir);
	void setPace(std::string scene, float pace);
	void setMinPace(std::string scene, float pace);
	void setHitMagnitude(std::string scene, float hitMagnitude);
	void setMass(std::string scene, float value);
	void setFriction(std::string scene, float value);
	void setRestitution(std::string scene, float value);
	void setHeight(std::string scene, float value); 
	void setRadius(std::string scene, float value); 
	void setStepOffset(std::string scene, float value);
	void setTimeStep(std::string scene, float value);

	std::map<std::string, float *> * getCameraPositions() { return cameraPositions; };
	bool hasCamera(std::string name) { return (cameraPositions->find(name) != cameraPositions->end()); };

protected:

	typedef struct {
		bool isCamera;
		externalInfo extInfo;
		int index;
		physx::PxVec3 * direction;
		float pace;
		float minPace;
		float hitMagnitude;
		float timeStep;
		physx::PxMat44 initialTrans;
	} PhysXController;

	std::map<std::string, PhysXController> controllers;
	std::map<std::string, float *> * cameraPositions;
	physx::PxControllerManager * manager;

	void createCharacter(const std::string & scene, physx::PxVec3 position, physx::PxVec3 up, physx::PxMaterial * material, bool isCamera = false, float radius = 1.0f, float height = 1.0f);
	void updateCameraPosition(std::string cameraName, physx::PxVec3 position);

	// Inherited via PxUserControllerHitReport
	virtual void onShapeHit(const physx::PxControllerShapeHit & hit) override;
	virtual void onControllerHit(const physx::PxControllersHit & hit) override;
	virtual void onObstacleHit(const physx::PxControllerObstacleHit & hit) override;
};

#endif