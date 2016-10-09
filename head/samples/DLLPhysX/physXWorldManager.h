#ifndef _PHYSXWORLDMANAGER_H
#define _PHYSXWORLDMANAGER_H

#include "PxPhysicsAPI.h"
#include "nau/physics/iPhysics.h"
#include "physXScene.h"
#include "physXRigidManager.h"
#include "physXSoftManager.h"
#include "physXParticleManager.h"
#include "physXCharacterManager.h"

class PhysXWorldManager {

private:
	physx::PxScene *world;
	physx::PxDefaultAllocator gAllocator;
	physx::PxDefaultErrorCallback gErrorCallback;
	physx::PxVisualDebuggerConnection* gConnection = NULL;
	physx::PxCooking* mCooking;
	float timeStep;

	PhysXRigidManager * rigidManager;
	PhysXSoftManager * softManager;
	PhysXParticleManager * particleManager;
	PhysXCharacterManager * characterManager;

public:
	PhysXWorldManager();
	~PhysXWorldManager();

	void update();
	void setGravity(float x, float y, float z);
	physx::PxMaterial * createMaterial(float dynamicFrction, float staticFriction, float restitution);

	void addRigid(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform, physx::PxMaterial * material, nau::physics::IPhysics::BoundingVolume shape, bool isStatic = false);
	void setRigidProperty(std::string scene, std::string propName, float value);
	void setRigidProperty(std::string scene, std::string propName, float * value);
	void moveRigid(std::string scene, float * transform);

	void addCloth(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform, nau::physics::IPhysics::SceneCondition condition, float * conditionValue);
	void setSoftProperty(std::string scene, std::string propName, float value);
	void setSoftProperty(std::string scene, std::string propName, float * value);
	void moveSoft(std::string scene, float * transform);

	void addParticles(const std::string &scene, const std::string &material, float maxParticles, float * positions, float *transform);
	std::map<std::string, int> * getMaterialParticleNb();
	
	void addCharacter(const std::string &scene, int nbVertices, float *vertices, int nbIndices, unsigned int *indices, float *transform, physx::PxMaterial * material, float * up);
	void setCharacterProperty(std::string scene, std::string propName, float * value);
	void moveCharacter(std::string scene, float * transform);

	void addCamera(const std::string & scene, float * position, float * up, float pace, float minPace, float hitMagnitude, float timeStep, float stepOffset, float mass, float radius, float height, physx::PxMaterial * material);
	bool hasCamera(std::string cameraName) { return characterManager->hasCamera(cameraName); };
	std::map<std::string, float *> * getCameraPositions();

	void setTimeStep(float tStep) { timeStep = tStep; };

};

#endif