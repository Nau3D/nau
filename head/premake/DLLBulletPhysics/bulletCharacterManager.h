#ifndef _BULLETCHARACTERMANAGER_H
#define _BULLETCHARACTERMANAGER_H

#include <map>
#include <vector>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>
#include <BulletCollision\CollisionDispatch\btGhostObject.h>
#include <BulletDynamics\Character\btKinematicCharacterController.h>
#include "bulletScene.h"

class BulletCharacterManager : btCollisionWorld::ContactResultCallback {
public:
	static int nextControllerIndex;

	std::string currentCharacter;

	BulletCharacterManager();
	~BulletCharacterManager();

	void update(btSoftRigidDynamicsWorld * world);
	void addCharacter(btSoftRigidDynamicsWorld * world, const std::string &scene, float height, float radius, float stepHeight);
	void addCamera(btSoftRigidDynamicsWorld * world, const std::string &scene, btVector3 position, float height, float radius, float stepHeight, float pace, float minPace, float hitMagnitude);
	void move(const std::string &scene);
	void move(const std::string &scene, float * transform);
	void createInfo(const std::string &scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform);

	void setDirection(std::string scene, btVector3 dir);
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

	btScalar addSingleResult(btManifoldPoint& cp, const btCollisionObjectWrapper* colObj0, int partId0, int index0, const btCollisionObjectWrapper* colObj1, int partId1, int index1);

protected:

	typedef struct {
		bool isCamera;
		BulletScene sceneInfo;
		int index;
		btVector3 * direction;
		float pace;
		float hitMagnitude;
		float timeStep;
		float minPace;
		btKinematicCharacterController * controller;
		//ContactSensorCallback * callBack;
	} BulletController;

	std::map<std::string, BulletController> controllers;
	std::map<std::string, float *> * cameraPositions;


	void createCharacter(btSoftRigidDynamicsWorld * world, const std::string &scene, btTransform transform, float height, float radius, float stepHeight);
	btKinematicCharacterController * BulletCharacterManager::getKinematicController(const std::string & scene);
	btGhostObject * BulletCharacterManager::getGhostObject(const std::string & scene);
	bool isPresent(const std::string &scene);
	void updateCameraPosition(std::string scene, btVector3 position);
};

#endif