#include "PhysXRigidManager.h"

using namespace physx;

PhysXRigidManager::PhysXRigidManager() {
}


PhysXRigidManager::~PhysXRigidManager() {
}

void 
PhysXRigidManager::update(const physx::PxActiveTransform * activeTransforms, physx::PxU32 nbActiveTransforms, float timeStep, physx::PxVec3 gravity) {
	for (PxU32 i = 0; i < nbActiveTransforms; ++i) {
		if (activeTransforms[i].userData != NULL) {
			std::string *n = static_cast<std::string*>(activeTransforms[i].userData);
			if (rigidBodies.find(*n) != rigidBodies.end()) {
				PxRigidDynamic * actor = rigidBodies[*n].info.actor->is<PxRigidDynamic>();
				if (rigidBodies[*n].rollingFriction >= 0.0f) {
					float mass = actor->getMass();
					float force = mass * gravity.magnitude() * rigidBodies[*n].rollingFriction;
					float actorMotion = (mass * actor->getLinearVelocity().magnitude()) / ((float)rigidBodies[*n].rollingFrictionTimeStamp * timeStep);
					if (force <= actorMotion) {
						rigidBodies[*n].rollingFrictionTimeStamp++;
						PxVec3 forceVec = -(actor->getLinearVelocity().getNormalized() * force);
						actor->addForce(forceVec);
					}
					else {
						rigidBodies[*n].rollingFrictionTimeStamp = 1;
						//actor->setLinearVelocity(actor->getLinearVelocity() / 1.5f);
						//actor->setAngularVelocity(actor->getAngularVelocity() / 1.5f);
						//actor->setLinearVelocity(PxVec3(0.0f));
						//actor->setAngularVelocity(PxVec3(0.0f));
					}
				}
				PxMat44 mat(activeTransforms[i].actor2World);
				mat.scale(PxVec4(PxVec3(rigidBodies[*n].scalingFactor), 1.0f));
				//getMatFromPhysXTransform(activeTransforms[i].actor2World, rigidBodies[*n].extInfo.transform);
				getMatFromPhysXTransform(mat, rigidBodies[*n].info.extInfo.transform);
			}
		}
	}
}

void 
PhysXRigidManager::createInfo(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform) {
	rigidBodies[scene].info.extInfo = externalInfo(nbVertices, vertices, nbIndices, indices, transform);
}

void 
PhysXRigidManager::addStaticBody(const std::string & scene, physx::PxScene * world, physx::PxCooking * mCooking, nau::physics::IPhysics::BoundingVolume shape, physx::PxMaterial * material) {
	PxPhysics *gPhysics = &(world->getPhysics());
	PxRigidStatic * staticActor;
	PxTransform trans = PxTransform(PxMat44(rigidBodies[scene].info.extInfo.transform));
	switch (shape.sceneShape)
	{
	case nau::physics::IPhysics::BOX:
	{
		staticActor = PxCreateStatic(
			world->getPhysics(),
			trans,
			PxBoxGeometry(shape.max[0], shape.max[1], shape.max[2]),
			*material
		);
	}
	break;
	case nau::physics::IPhysics::SPHERE:
	{
		staticActor = PxCreateStatic(
			world->getPhysics(),
			trans,
			PxSphereGeometry(shape.max[0]),
			*material
		);
	}
	break;
	case nau::physics::IPhysics::CAPSULE:
	{
		staticActor = PxCreateStatic(
			world->getPhysics(),
			trans,
			PxCapsuleGeometry(
				shape.max[0],
				shape.max[1]
			),
			*material
		);
	}
	break;
	default:
	{
		if (scene.compare("plane") == 0) {
			staticActor = PxCreatePlane(
				world->getPhysics(),
				PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
				*material
			);
		}
		else {
			staticActor = gPhysics->createRigidStatic(trans);
			PxTriangleMeshGeometry triGeom(gPhysics->createTriangleMesh(*getTriangleMeshGeo(world, mCooking, rigidBodies[scene].info.extInfo, true)));
			//triGeom.triangleMesh = gPhysics->createTriangleMesh(*getTriangleMeshGeo(world, mCooking, rigidBodies[scene].extInfo, true));
			staticActor->createShape(triGeom, *material);
		}
	}
	break;
	}
	staticActor->userData = static_cast<void*> (new std::string(scene));
	world->addActor(*staticActor);
	rigidBodies[scene].info.actor = staticActor;
}

void
PhysXRigidManager::addDynamicBody(const std::string & scene, physx::PxScene * world, physx::PxCooking * mCooking, nau::physics::IPhysics::BoundingVolume shape, physx::PxMaterial * material) {
	PxPhysics *gPhysics = &(world->getPhysics());
	PxRigidDynamic * dynamic;

	rigidBodies[scene].scalingFactor = getScalingFactor(rigidBodies[scene].info.extInfo.transform);
	PxMeshScale scale = PxMeshScale(rigidBodies[scene].scalingFactor);
	PxMat44 aux(rigidBodies[scene].info.extInfo.transform);
	PxVec3 pos(aux.getPosition());
	aux *= (1.0f / rigidBodies[scene].scalingFactor);
	PxMat44 mat(
		PxVec3(aux.column0.x, aux.column0.y, aux.column0.z),
		PxVec3(aux.column1.x, aux.column1.y, aux.column1.z),
		PxVec3(aux.column2.x, aux.column2.y, aux.column2.z),
		pos
	);

	//PxTransform trans = PxTransform(PxMat44(rigidBodies[scene].extInfo.transform));
	PxTransform trans = PxTransform(mat);

	switch (shape.sceneShape)
	{
	case nau::physics::IPhysics::BOX:
	{
		dynamic = gPhysics->createRigidDynamic(trans);
		dynamic->createShape(
			PxBoxGeometry(
				shape.max[0] * rigidBodies[scene].scalingFactor,
				shape.max[1] * rigidBodies[scene].scalingFactor,
				shape.max[2] * rigidBodies[scene].scalingFactor),
			*material);
	}
	break;
	case nau::physics::IPhysics::SPHERE:
	{
		dynamic = gPhysics->createRigidDynamic(trans);
		dynamic->createShape(PxSphereGeometry(shape.max[0] * rigidBodies[scene].scalingFactor), *material);
	}
	break;
	case nau::physics::IPhysics::CAPSULE:
	{
		dynamic = gPhysics->createRigidDynamic(trans);
		dynamic->createShape(
			PxCapsuleGeometry(
				shape.max[0] * rigidBodies[scene].scalingFactor,
				shape.max[1] * rigidBodies[scene].scalingFactor),
			*material);
	}
	break;
	default:
	{
		dynamic = gPhysics->createRigidDynamic(trans);
		PxConvexMesh * convexMesh = gPhysics->createConvexMesh(*getTriangleMeshGeo(world, mCooking, rigidBodies[scene].info.extInfo, false));
		dynamic->createShape(PxConvexMeshGeometry(convexMesh, scale), *material);
	}
	break;
	}

	dynamic->userData = static_cast<void*> (new std::string(scene));
	rigidBodies[scene].rollingFriction = -1.0f;
	rigidBodies[scene].rollingFrictionTimeStamp = 1;
	world->addActor(*dynamic);
	rigidBodies[scene].info.actor = dynamic;
}

physx::PxInputStream*
PhysXRigidManager::getTriangleMeshGeo(PxScene *world, physx::PxCooking* mCooking, ExternalInfo externInfo, bool isStatic) {
	PxPhysics *gPhysics = &(world->getPhysics());
	PxDefaultMemoryOutputStream *writeBuffer = new PxDefaultMemoryOutputStream();

	bool status;
	if (isStatic) {
		PxTriangleMeshDesc meshDesc;
		meshDesc.points.count = externInfo.nbVertices;
		meshDesc.points.stride = 4 * sizeof(float);
		meshDesc.points.data = externInfo.vertices;

		meshDesc.triangles.count = static_cast<int> (externInfo.nbIndices / 3);
		meshDesc.triangles.stride = 3 * sizeof(unsigned int);
		meshDesc.triangles.data = externInfo.indices;
		
		status = mCooking->cookTriangleMesh(meshDesc, *writeBuffer);
	}
	else {
		PxConvexMeshDesc meshDesc2;

		meshDesc2.points.count = externInfo.nbVertices;
		meshDesc2.points.stride = 4 * sizeof(float);
		meshDesc2.points.data = externInfo.vertices;
		meshDesc2.flags |= PxConvexFlag::eCOMPUTE_CONVEX | PxConvexFlag::eINFLATE_CONVEX;
		meshDesc2.vertexLimit = 256;

		status = mCooking->cookConvexMesh(meshDesc2, *writeBuffer);

	}
	return new PxDefaultMemoryInputData(writeBuffer->getData(), writeBuffer->getSize());
}

void 
PhysXRigidManager::setMass(std::string name, float value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		PxRigidDynamic * dyn = rigidBodies[name].info.actor->is<PxRigidDynamic>();
		if (dyn) {
			dyn->setMass(value);
		}
	}
}

void PhysXRigidManager::setInertiaTensor(std::string name, float * value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		PxRigidDynamic * dyn = rigidBodies[name].info.actor->is<PxRigidDynamic>();
		if (dyn) {
			dyn->setMassSpaceInertiaTensor(PxVec3(value[0], value[1], value[2]));
		}
	}
}

void 
PhysXRigidManager::setStaticFriction(std::string name, float value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		PxRigidActor * actor = rigidBodies[name].info.actor->is<PxRigidActor>();
		if (actor) {
			for (PxShape* shape : getShapes(actor)) {
				for (PxMaterial * mat : getMaterials(shape)) {
					mat->setStaticFriction(value);
				}
			}
		}
	}
}

void 
PhysXRigidManager::setDynamicFriction(std::string name, float value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		PxRigidActor * actor = rigidBodies[name].info.actor->is<PxRigidActor>();
		if (actor) {
			for (PxShape* shape : getShapes(actor)) {
				for (PxMaterial * mat : getMaterials(shape)) {
					mat->setDynamicFriction(value);
				}
			}
		}
	}
}

void PhysXRigidManager::setRollingFriction(std::string name, float value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		PxRigidDynamic * dyn = rigidBodies[name].info.actor->is<PxRigidDynamic>();
		if (dyn) {
			rigidBodies[name].rollingFriction = value;
			rigidBodies[name].rollingFrictionTimeStamp = 1;
		}
	}
}

void 
PhysXRigidManager::setRestitution(std::string name, float value) {
	if (rigidBodies.find(name) != rigidBodies.end()) {
		PxRigidActor * actor = rigidBodies[name].info.actor->is<PxRigidActor>();
		if (actor) {
			for (PxShape* shape : getShapes(actor)) {
				for (PxMaterial * mat : getMaterials(shape)) {
					mat->setRestitution(value);
				}
			}
		}
	}
}


void 
PhysXRigidManager::move(std::string scene, float * transform) {
	if (rigidBodies.find(scene) != rigidBodies.end()) {
		PxRigidActor * actor = rigidBodies[scene].info.actor->is<PxRigidActor>();
		if (actor) {
			rigidBodies[scene].info.extInfo.transform = transform;
			actor->setGlobalPose(PxTransform(PxMat44(transform)));
		}
	}
}

void 
PhysXRigidManager::setForce(std::string scene, float * force) {
	if (rigidBodies.find(scene) != rigidBodies.end()) {
		PxRigidDynamic * actor = rigidBodies[scene].info.actor->is<PxRigidDynamic>();
		if (actor) {
			actor->addForce(PxVec3(force[0], force[1], force[2]));
		}
	}
}

void PhysXRigidManager::setImpulse(std::string scene, float * impulse) {
	if (rigidBodies.find(scene) != rigidBodies.end()) {
		PxRigidDynamic * actor = rigidBodies[scene].info.actor->is<PxRigidDynamic>();
		if (actor) {
			actor->addForce(PxVec3(impulse[0], impulse[1], impulse[2]), PxForceMode::eIMPULSE);
		}
	}
}

std::vector<physx::PxShape*> 
PhysXRigidManager::getShapes(physx::PxRigidActor * actor) {
	PxU32 nbShapes = actor->getNbShapes();
	std::vector<PxShape *> list(nbShapes);
	actor->getShapes(&list.at(0), nbShapes);
	return list;
}

std::vector<physx::PxMaterial*>
PhysXRigidManager::getMaterials(physx::PxShape * shape) {
	PxU32 nbMat = shape->getNbMaterials();
	std::vector<PxMaterial *> matList(nbMat);
	shape->getMaterials(&matList.at(0), nbMat);
	return matList;
}

float 
PhysXRigidManager::getScalingFactor(float * trans) {
	float m00 = trans[0];
	float m01 = trans[4];
	float m02 = trans[8];
	return (sqrt(m00 * m00 + m01 * m01 + m02 * m02));
}
