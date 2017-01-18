#include "bulletWorldManager.h"



BulletWorldManager::BulletWorldManager() {
	btSoftBodyRigidBodyCollisionConfiguration * collisionConfiguration = new btSoftBodyRigidBodyCollisionConfiguration();// btDefaultCollisionConfiguration();
	btCollisionDispatcher* dispatcher = new	btCollisionDispatcher(collisionConfiguration);

	btBroadphaseInterface* broadphase = new btDbvtBroadphase();
	btSequentialImpulseConstraintSolver* solver = new btSequentialImpulseConstraintSolver;

	btGImpactCollisionAlgorithm::registerAlgorithm(dispatcher);
	btDefaultSoftBodySolver * softbodySolver = new btDefaultSoftBodySolver();
	world = new btSoftRigidDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration, softbodySolver);
	//world->getPairCache()->setInternalGhostPairCallback(new btGhostPairCallback());

	rigidManager = new BulletRigidManager();
	softManager = new BulletSoftManager();
	characterManager = new BulletCharacterManager();
	debugDrawer = NULL;
}


BulletWorldManager::~BulletWorldManager() {
	delete debugDrawer;
	delete rigidManager;
	delete softManager;
	delete world;
}

void BulletWorldManager::update() {
	if (world) {
		world->stepSimulation(timeStep);
		rigidManager->update();
		softManager->update();
		characterManager->update(world);
		if (debugDrawer) {
			debugDrawer->clear();
			world->debugDrawWorld();
		}
	}
}

void BulletWorldManager::setGravity(float x, float y, float z) {
	world->setGravity(btVector3(x, y, z));
}

void BulletWorldManager::addRigid(const std::string &scene, int nbVertices, float *vertices, int nbIndices, unsigned int *indices, float *transform, nau::physics::IPhysics::BoundingVolume shape, float mass, bool isStatic) {
	rigidManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform);
	world->addRigidBody(
		rigidManager->addRigid(
			scene,
			rigidManager->createCollisionShape(scene, shape, isStatic),
			mass,
			isStatic
		)
	);
}

void BulletWorldManager::setRigidProperty(std::string scene, std::string propName, float value) {
	if (propName.compare("MASS") == 0)
		rigidManager->setMass(scene, value);
	else if (propName.compare("FRICTION") == 0 || propName.compare("DYNAMIC_FRICTION") == 0 || propName.compare("STATIC_FRICTION") == 0) 
		rigidManager->setFriction(scene, value);
	else if (propName.compare("ROLLING_FRICTION") == 0) 
		rigidManager->setRollingFriction(scene, value);
	else if (propName.compare("RESTITUTION") == 0) 
		rigidManager->setRestitution(scene, value);
}

void BulletWorldManager::setRigidProperty(std::string scene, std::string propName, float * value) {
	if (propName.compare("IMPULSE") == 0)
		rigidManager->addImpulse(scene, value);
	else if (propName.compare("INERTIA") == 0)
		rigidManager->setLocalInertia(scene, value);
}

void BulletWorldManager::moveRigid(std::string scene, float * transform) {
	rigidManager->move(scene, transform); 
}

void BulletWorldManager::addCloth(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform, nau::physics::IPhysics::SceneCondition condition, float * conditionValue) {
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
	world->addSoftBody(softManager->addSoftBody(world->getWorldInfo(), scene));
}

void BulletWorldManager::setSoftProperty(std::string scene, std::string propName, float value) {
	if (propName.compare("VITERATIONS") == 0)
		softManager->setViterations(scene, value);
	else if (propName.compare("PITERATIONS") == 0)
		softManager->setPiteration(scene, value);
}

void BulletWorldManager::moveSoft(std::string scene, float * transform) {
	softManager->move(scene, transform);
}

void BulletWorldManager::setDebug() {
	if (!debugDrawer) {
		debugDrawer = new BulletDebugger();
		debugDrawer->setDebugMode(btIDebugDraw::DBG_DrawWireframe);
		//debugDrawer->setDebugMode(btIDebugDraw::DBG_MAX_DEBUG_DRAW_MODE);
		world->setDebugDrawer(debugDrawer);
	}
}

void BulletWorldManager::addCharacter(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform, float height, float radius, float stepOffset) {
	characterManager->createInfo(scene, nbVertices, vertices, nbIndices, indices, transform);
	characterManager->addCharacter(world, scene, height, radius, stepOffset);
}

void BulletWorldManager::setCharacterProperty(std::string scene, std::string propName, float * value) {
	if (propName.compare("DIRECTION") == 0)
		characterManager->setDirection(scene, btVector3(value[0], value[1], value[2]));
	else if (propName.compare("PACE") == 0)
		characterManager->setPace(scene, *value);
	else if (propName.compare("HIT_MAGNITUDE") == 0)
		characterManager->setHitMagnitude(scene, *value);
}

void BulletWorldManager::moveCharacter(std::string scene, float * transform)
{
}

void BulletWorldManager::addCamera(const std::string & scene, float * position, float pace, float minPace, float hitMagnitude, float stepOffset, float radius, float height) {
	characterManager->addCamera(
		world,
		scene,
		btVector3(position[0], position[1], position[2]),
		height,
		radius,
		stepOffset,
		pace,
		minPace,
		hitMagnitude
	);
}

std::map<std::string, float*>* BulletWorldManager::getCameraPositions() {
	return characterManager->getCameraPositions();
}
