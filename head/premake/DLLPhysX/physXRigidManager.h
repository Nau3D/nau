#ifndef _PHYSXRIGIDMANAGER_H
#define _PHYSXRIGIDMANAGER_H

#include <map>
#include <vector>
#include "nau/physics/iPhysics.h"
#include "PxPhysicsAPI.h"
#include "physXScene.h"

class PhysXRigidManager {

public:
	PhysXRigidManager();
	~PhysXRigidManager();

	void update(const physx::PxActiveTransform * activeTransforms, physx::PxU32 nbActiveTransforms, float timeStep, physx::PxVec3 gravity);
	void createInfo(const std::string &scene, int nbVertices, float *vertices, int nbIndices, unsigned int *indices, float *transform);
	void addStaticBody(const std::string & scene, physx::PxScene * world, physx::PxCooking * mCooking, nau::physics::IPhysics::BoundingVolume shape, physx::PxMaterial * material);
	void addDynamicBody(const std::string & scene, physx::PxScene * world, physx::PxCooking * mCooking, nau::physics::IPhysics::BoundingVolume shape, physx::PxMaterial * material);
	void setMass(std::string name, float value);
	void setInertiaTensor(std::string name, float * value);
	void setStaticFriction(std::string name, float value);
	void setDynamicFriction(std::string name, float value);
	void setRollingFriction(std::string name, float value);
	void setRestitution(std::string name, float value);
	void move(std::string scene, float * transform);
	void setForce(std::string scene, float * force);
	void setImpulse(std::string scene, float * impulse);

protected:
	
	std::vector<physx::PxShape*> getShapes(physx::PxRigidActor * actor);
	std::vector<physx::PxMaterial*> getMaterials(physx::PxShape * shape);
	float getScalingFactor(float * trans);

	typedef struct {
		PhysXScene info;
		float rollingFriction;
		float scalingFactor;
		int rollingFrictionTimeStamp;
	} RigidObject;

	std::map<std::string, RigidObject> rigidBodies;
	physx::PxInputStream * getTriangleMeshGeo(physx::PxScene *world, physx::PxCooking* mCooking, ExternalInfo externInfo, bool isStatic);
};

#endif
