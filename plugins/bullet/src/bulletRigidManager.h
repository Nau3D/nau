#ifndef _BULLETRIGIDMANAGER_H
#define _BULLETRIGIDMANAGER_H

#include <map>
#include <vector>
#include <nau\physics\iPhysics.h>
#include "bulletScene.h"
#include "bulletMotionState.h"
#include "btBulletDynamicsCommon.h"
#include "BulletCollision/Gimpact/btGImpactShape.h"
#include "BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h"

class BulletRigidManager {

public:
	BulletRigidManager();
	~BulletRigidManager();

	void update();
	void createInfo(const std::string &scene, int nbVertices, float *vertices, int nbIndices, unsigned int *indices, float *transform);
	btCollisionShape * createCollisionShape(const std::string & scene, nau::physics::IPhysics::BoundingVolume shape, bool isStatic = true);
	btRigidBody * addRigid(const std::string &scene, btCollisionShape * shape, float mass, bool isStatic);
	void setMass(std::string name, float value);
	void setLocalInertia(std::string name, float * value);
	void setFriction(std::string name, float value);
	void setRollingFriction(std::string name, float value);
	void setRestitution(std::string name, float value);
	void addImpulse(std::string name, float * value);
	void move(std::string scene, float * transform);

protected:
	std::map<std::string, BulletScene> rigidBodies;

};

#endif