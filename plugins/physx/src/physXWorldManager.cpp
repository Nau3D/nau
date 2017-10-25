#include "physXWorldManager.h"

using namespace physx;

PhysXWorldManager::PhysXWorldManager() {
	PxTolerancesScale scale = PxTolerancesScale();
	PxFoundation* gFoundation = PxCreateFoundation(PX_FOUNDATION_VERSION, gAllocator, gErrorCallback);
   //PxProfileZoneManager* profileZoneManager = &PxProfileZoneManager::createProfileZoneManager(gFoundation);

	/*PxCudaContextManagerDesc cudaContextManagerDesc;
	PxCudaContextManager* mCudaContextManager = PxCreateCudaContextManager(*gFoundation, cudaContextManagerDesc, profileZoneManager);
	if (mCudaContextManager)
	{
		if (!mCudaContextManager->contextIsValid())
		{
			mCudaContextManager->release();
			mCudaContextManager = NULL;
		}
	}*/

	PxPhysics*	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, scale, true, NULL);
// ARF if (gPhysics->getPvdConnectionManager()) {
//		gPhysics->getVisualDebugger()->setVisualizeConstraints(true);
//		gPhysics->getVisualDebugger()->setVisualDebuggerFlag(PxVisualDebuggerFlag::eTRANSMIT_CONTACTS, true);
//		gPhysics->getVisualDebugger()->setVisualDebuggerFlag(PxVisualDebuggerFlag::eTRANSMIT_SCENEQUERIES, true);
//		//gPhysics->getVisualDebugger()->setVisualDebuggerFlag(PxVisualDebuggerFlag::eTRANSMIT_CONSTRAINTS, true);
////		gConnection = PxVisualDebuggerExt::createConnection(gPhysics->getPvdConnectionManager(), "127.0.0.1", 5425, 100);
//	}
	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	//sceneDesc.gpuDispatcher = mCudaContextManager->getGpuDispatcher();
	PxDefaultCpuDispatcher* gDispatcher = PxDefaultCpuDispatcherCreate(2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	sceneDesc.flags |= PxSceneFlag::eENABLE_ACTIVETRANSFORMS;
	//sceneDesc.frictionType = PxFrictionType::eTWO_DIRECTIONAL;
	world = gPhysics->createScene(sceneDesc);
	//m_pDynamicsWorld->setFlag(PxSceneFlag::eENABLE_ACTIVETRANSFORMS, true);

	mCooking = PxCreateCooking(PX_PHYSICS_VERSION, *gFoundation, PxCookingParams(scale));

	rigidManager = new PhysXRigidManager();
	softManager = new PhysXSoftManager();
	particleManager = new PhysXParticleManager();
	characterManager = new PhysXCharacterManager(world);
}

PhysXWorldManager::~PhysXWorldManager() {
	// ARFif (gConnection)
	//	gConnection->release();
	world->release();
	delete &world;
	delete rigidManager;
}

void PhysXWorldManager::update() {
	if (world) {
		world->simulate(timeStep);
		world->fetchResults(true);

		//RIGID BODIES UPDATE
		PxU32 nbActiveTransforms;
		const PxActiveTransform* activeTransforms = world->getActiveTransforms(nbActiveTransforms);
		rigidManager->update(activeTransforms, nbActiveTransforms, timeStep, world->getGravity());
		
		//SOFT BODIES UPDATE
		softManager->update();

		//PARTICLE UPDATE
		particleManager->update();

		//CHARACTER UPDATE
		characterManager->update(world->getGravity());
	}
}

void PhysXWorldManager::setGravity(float x, float y, float z) {
		world->setGravity(PxVec3(x, y, z));
}

physx::PxMaterial * PhysXWorldManager::createMaterial(float dynamicFrction, float staticFriction, float restitution) {
	return (&(world->getPhysics()))->createMaterial(staticFriction, dynamicFrction, restitution);
}

void PhysXWorldManager::addRigid(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform, physx::PxMaterial * material, nau::physics::IPhysics::BoundingVolume shape, bool isStatic) {
	rigidManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform);
	if (isStatic) {
		rigidManager->addStaticBody(
			scene,
			world,
			mCooking,
			shape,
			material
		);
	}
	else {
		rigidManager->addDynamicBody(
			scene,
			world,
			mCooking,
			shape,
			material
		);
	}
}

void PhysXWorldManager::setRigidProperty(std::string scene, std::string propName, float value) {
	if (propName.compare("MASS") == 0)
		rigidManager->setMass(scene, value);
	else if (propName.compare("STATIC_FRICTION") == 0)
		rigidManager->setStaticFriction(scene, value);
	else if (propName.compare("DYNAMIC_FRICTION") == 0)
		rigidManager->setDynamicFriction(scene, value);
	else if (propName.compare("ROLLING_FRICTION") == 0)
		rigidManager->setRollingFriction(scene, value);
	else if (propName.compare("RESTITUTION") == 0)
		rigidManager->setRestitution(scene, value);
}

void PhysXWorldManager::setRigidProperty(std::string scene, std::string propName, float * value) {
	if (propName.compare("FORCE") == 0)
		rigidManager->setForce(scene, value);
	else if (propName.compare("IMPULSE") == 0)
		rigidManager->setImpulse(scene, value);
	else if (propName.compare("INERTIA") == 0)
		rigidManager->setInertiaTensor(scene, value);
}

void PhysXWorldManager::moveRigid(std::string scene, float * transform) {
	rigidManager->move(scene, transform);
}

void PhysXWorldManager::addCloth(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform, nau::physics::IPhysics::SceneCondition condition, float * conditionValue) {
	switch (condition)
	{
	case nau::physics::IPhysics::GT:
		softManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform, CLOTH_CONDITION_GT, conditionValue);
		break;
	case nau::physics::IPhysics::LT:
		softManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform, CLOTH_CONDITION_LT, conditionValue);
		break;
	case nau::physics::IPhysics::EGT:
		softManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform, CLOTH_CONDITION_EGT, conditionValue);
		break;
	case nau::physics::IPhysics::ELT:
		softManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform, CLOTH_CONDITION_ELT, conditionValue);
		break;
	case nau::physics::IPhysics::EQ:
		softManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform, CLOTH_CONDITION_EQ, conditionValue);
		break;
	default:
		softManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform, CLOTH_CONDITION_NONE, conditionValue);
		break;
	}
	softManager->addSoftBody(world, scene);
}

void PhysXWorldManager::setSoftProperty(std::string scene, std::string propName, float value) {
	if (propName.compare("SOLVER_FREQUENCY") == 0)
		softManager->setSolverFrequency(scene, value);
	else if (propName.compare("INERTIA_SCALE") == 0)
		softManager->setInertiaScale(scene, value);
	else if (propName.compare("FRICTION_COEFFICIENT") == 0)
		softManager->setFrictionCoefficient(scene, value);
	else if (propName.compare("COLLISION_MASS_SCALE") == 0)
		softManager->setCollisionMassScale(scene, value);
	else if (propName.compare("SELF_COLLISION_DISTANCE") == 0)
		softManager->setSelfCollisionDistance(scene, value);
	else if (propName.compare("SELF_COLLISION_STIFFNESS") == 0)
		softManager->setSelfCollisionStiffness(scene, value);
}

void PhysXWorldManager::setSoftProperty(std::string scene, std::string propName, float * value) {
	if (propName.compare("VERTICAL_STRETCH") == 0)
		softManager->setVerticalStretch(scene, value);
	else if (propName.compare("HORIZONTAL_STRETCH") == 0)
		softManager->setHorizontalStretch(scene, value);
	else if (propName.compare("SHEARING") == 0)
		softManager->setShearing(scene, value);
	else if (propName.compare("BENDING") == 0)
		softManager->setBending(scene, value);
}

void PhysXWorldManager::moveSoft(std::string scene, float * transform) {
	softManager->move(scene, transform);
}

void PhysXWorldManager::addParticles(const std::string &scene, const std::string &material, float maxParticles, float * positions, float *transform) {
	particleManager->addParticleSystem(world, scene, material, maxParticles, positions, transform);
}

std::map<std::string, int>* PhysXWorldManager::getMaterialParticleNb() {
	return particleManager->getParticleSystemsParticleNb();
}

void PhysXWorldManager::addCharacter(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform, physx::PxMaterial * material, float * up) {
	characterManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform);
	characterManager->addCharacter(scene, material, PxVec3(up[0], up[1], up[2]));
}

void PhysXWorldManager::addCamera(const std::string & scene, float * position, float * up, float pace, float minPace, float hitMagnitude, float timeStep, float stepOffset, float mass, float radius, float height, physx::PxMaterial * material) {
	characterManager->addCamera(
		scene,
		PxVec3(position[0], position[1], position[2]),
		PxVec3(up[0], up[1], up[2]),
		pace,
		minPace,
		hitMagnitude,
		timeStep,
		stepOffset,
		mass,
		radius,
		height,
		material
	);
}

void PhysXWorldManager::setCharacterProperty(std::string scene, std::string propName, float * value) {
	if (propName.compare("DIRECTION") == 0)
		characterManager->setDirection(scene, PxVec3(value[0], value[1], value[2]));
	else if (propName.compare("PACE") == 0)
		characterManager->setPace(scene, *value);
	else if (propName.compare("HIT_MAGNITUDE") == 0)
		characterManager->setHitMagnitude(scene, *value);
	else if (propName.compare("HEIGHT") == 0)
		characterManager->setHeight(scene, *value);
	else if (propName.compare("RADIUS") == 0)
		characterManager->setRadius(scene, *value);
	else if (propName.compare("STEP_OFFSET") == 0)
		characterManager->setStepOffset(scene, *value);
	else if (propName.compare("MASS") == 0)
		characterManager->setMass(scene, *value);
	else if (propName.compare("FRICTION") == 0)
		characterManager->setFriction(scene, *value);
	else if (propName.compare("RESTITUTION") == 0)
		characterManager->setRestitution(scene, *value);
}

void PhysXWorldManager::moveCharacter(std::string scene, float * transform) {
	characterManager->move(scene, transform, world->getGravity());
}

std::map<std::string, float *> * PhysXWorldManager::getCameraPositions() {
	return characterManager->getCameraPositions();
}
