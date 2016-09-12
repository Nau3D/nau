#ifndef _BULLETSOFTMANAGER_H
#define _BULLETSOFTMANAGER_H

#include <map>
#include <vector>
#include "bulletScene.h"
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>
#include <BulletSoftBody/btDefaultSoftBodySolver.h>
#include <BulletSoftBody/btSoftBodyHelpers.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>

#define CLOTH_CONDITION_NONE -1
#define CLOTH_CONDITION_GT 0
#define CLOTH_CONDITION_LT 1
#define CLOTH_CONDITION_EGT 2
#define CLOTH_CONDITION_ELT 3
#define CLOTH_CONDITION_EQ 4

class BulletSoftManager {

public:
	BulletSoftManager();
	~BulletSoftManager();

	void update();
	void createInfo(const std::string &scene, int nbVertices, float *vertices, int nbIndices, unsigned int *indices, float *transform, int condition, float * conditionValue);
	btSoftBody * addSoftBody(btSoftBodyWorldInfo & worldInfo, const std::string &scene);
	void setViterations(const std::string &scene, float value);
	void setPiteration(const std::string &scene, float value);
	void move(std::string scene, float * transform);

protected:

	typedef struct {
		BulletScene info;
		int condition;
		btStaticPlaneShape * contidionPlane;
	} SoftScene;

	std::map<std::string, SoftScene> softBodies;
	bool isLocked(const std::string  &scene, btVector3 vertex);
	float distance(const std::string &scene, btVector3 p);
	bool contains(const std::string &scene, btVector3 p);
	btSoftBody * getCloth(const std::string &scene);
};

#endif