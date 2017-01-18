#include "bulletCharacterManager.h"
#include "bulletMotionState.h"

int BulletCharacterManager::nextControllerIndex = 0;

BulletCharacterManager::BulletCharacterManager() {
	cameraPositions = new std::map<std::string, float *>();
	currentCharacter = "";
}


BulletCharacterManager::~BulletCharacterManager() {
	delete &controllers;
	delete &cameraPositions;
}

void BulletCharacterManager::update(btSoftRigidDynamicsWorld * world) {
	for (auto scene : controllers) {
		currentCharacter = scene.first;
		//world->contactTest(scene.second.sceneInfo.object, *scene.second.callBack);
		world->contactTest(scene.second.sceneInfo.object, *this);
		btGhostObject * ghost = btGhostObject::upcast(scene.second.sceneInfo.object);
		btTransform trans = ghost->getWorldTransform();

		if (scene.second.isCamera) {
			updateCameraPosition(scene.first, trans.getOrigin());
			setPace(scene.first, controllers[scene.first].pace * 0.9f);
		}
		else {

			btVector3 up = world->getGravity().normalize() * -1.0f;
			btVector3 dir = *controllers[scene.first].direction;
			btVector3 right = up.cross(dir).normalize() * -1.0f;

			btMatrix3x3 mat = btMatrix3x3(
				right.getX(), up.getX(), dir.getX(),
				right.getY(), up.getY(), dir.getY(),
				right.getZ(), up.getZ(), dir.getZ()
			);

			btTransform tr = btTransform(trans.getBasis() * mat, trans.getOrigin());
			tr.getOpenGLMatrix(scene.second.sceneInfo.extInfo.transform);
		}
	}
}

void BulletCharacterManager::addCharacter(btSoftRigidDynamicsWorld * world, const std::string & scene, float height, float radius, float stepHeight) {
	//INFO: Radius, Height and StepHeight have to be set in intitialization and cannot be change
	//TODO: (remains to be tested by removing the character from the world and make the changes)
	btTransform startTransform;
	startTransform.setFromOpenGLMatrix(controllers[scene].sceneInfo.extInfo.transform);
	createCharacter(world, scene, startTransform, height, radius, stepHeight);
}

void BulletCharacterManager::addCamera(btSoftRigidDynamicsWorld * world, const std::string &scene, btVector3 position, float height, float radius, float stepHeight, float pace, float minPace, float hitMagnitude) {
	btTransform trans = btTransform();
	trans.setIdentity();
	trans.setOrigin(position);
	createCharacter(world, scene, trans, height, radius, stepHeight);
	controllers[scene].isCamera = true;
	updateCameraPosition(scene, position);
	setMinPace(scene, minPace);
	setDirection(scene, btVector3(0.0f, 0.0f, 0.0f));
	setPace(scene, pace);
	setHitMagnitude(scene, hitMagnitude);
}

void BulletCharacterManager::move(const std::string & scene) {

}

void BulletCharacterManager::move(const std::string & scene, float * transform) {

}

void BulletCharacterManager::createInfo(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform) {
	controllers[scene].sceneInfo.extInfo = externalInfo(nbVertices, vertices, nbIndices, indices, transform);
}

void BulletCharacterManager::setDirection(std::string scene, btVector3 dir) {
	btKinematicCharacterController * controller = getKinematicController(scene);
	if (controller) {
		if (controllers[scene].direction) {
			controllers[scene].direction->setX((btScalar)dir.getX());
			controllers[scene].direction->setY((btScalar)dir.getY());
			controllers[scene].direction->setZ((btScalar)dir.getZ());
		}
		else
			controllers[scene].direction = new btVector3(dir);
		controller->setWalkDirection(dir * controllers[scene].pace);
	}
}

void BulletCharacterManager::setPace(std::string scene, float pace) {
	btKinematicCharacterController * controller = getKinematicController(scene);
	if (controller) {
		float minPace = controllers[scene].minPace ? controllers[scene].minPace : 0.01f;
		controllers[scene].pace = pace > minPace ? pace : 0.0f;
		controller->setWalkDirection(*controllers[scene].direction * pace);
	}
}

void BulletCharacterManager::setMinPace(std::string scene, float pace) {
	if (isPresent(scene))
		controllers[scene].minPace = pace;
}

void BulletCharacterManager::setHitMagnitude(std::string scene, float hitMagnitude) {
	if (isPresent(scene))
		controllers[scene].hitMagnitude = hitMagnitude;
}

void BulletCharacterManager::setMass(std::string scene, float value) {
}

void BulletCharacterManager::setFriction(std::string scene, float value)
{
}

void BulletCharacterManager::setRestitution(std::string scene, float value)
{
}

void BulletCharacterManager::setHeight(std::string scene, float value) {

}

void BulletCharacterManager::setRadius(std::string scene, float value) {

}

void BulletCharacterManager::setStepOffset(std::string scene, float value)
{
}

void BulletCharacterManager::setTimeStep(std::string scene, float value)
{
}

btScalar BulletCharacterManager::addSingleResult(btManifoldPoint & cp, const btCollisionObjectWrapper * colObj0, int partId0, int index0, const btCollisionObjectWrapper * colObj1, int partId1, int index1) {
	bool isFirstBody = colObj0->m_collisionObject == controllers[currentCharacter].sceneInfo.object;
	std::string *n = static_cast<std::string*>(isFirstBody ? colObj0->m_collisionObject->getUserPointer() : colObj1->m_collisionObject->getUserPointer());
	bool isStatic = colObj1->m_collisionObject->getCollisionFlags() == btCollisionObject::CF_STATIC_OBJECT;

	btScalar direction = isFirstBody ? btScalar(-1.0) : btScalar(1.0);

	if (!isStatic) {
		const btVector3& ptA = cp.getPositionWorldOnA();
		const btVector3& ptB = cp.getPositionWorldOnB();
		const btVector3& normalOnB = cp.m_normalWorldOnB;
		(btRigidBody::upcast((btCollisionObject*)colObj1->m_collisionObject))->applyCentralForce(
			(normalOnB * direction) * controllers[*n].hitMagnitude
		);
	}
	return 0;
}

void BulletCharacterManager::createCharacter(btSoftRigidDynamicsWorld * world, const std::string & scene, btTransform transform, float height, float radius, float stepHeight) {
	//INFO: Radius, Height and StepHeight have to be set in intitialization and cannot be change
	//TODO: (remains to be tested by removing the character from the world and make the changes)
	btConvexShape* capsule = new btCapsuleShape(radius, height);

	btPairCachingGhostObject* m_ghostObject = new btPairCachingGhostObject();
	m_ghostObject->setWorldTransform(transform);
	world->getBroadphase()->getOverlappingPairCache()->setInternalGhostPairCallback(new btGhostPairCallback());
	m_ghostObject->setCollisionShape(capsule);
	m_ghostObject->setCollisionFlags(btCollisionObject::CF_CHARACTER_OBJECT);
	m_ghostObject->setUserPointer(static_cast<void *>(new std::string(scene)));

	btKinematicCharacterController* charCon = new btKinematicCharacterController(m_ghostObject, capsule, stepHeight);
//	charCon->setGravity(-world->getGravity().getY());
	charCon->setGravity(world->getGravity());
	controllers[scene].controller = charCon;

	world->addCollisionObject(m_ghostObject, btBroadphaseProxy::CharacterFilter, btBroadphaseProxy::AllFilter);
	world->addAction(charCon);
	controllers[scene].sceneInfo.object = m_ghostObject;
	//controllers[scene].callBack = new ContactSensorCallback(*m_ghostObject);
}

btKinematicCharacterController * BulletCharacterManager::getKinematicController(const std::string & scene) {
	if (isPresent(scene))
		return controllers[scene].controller;
	return NULL;
}

btGhostObject * BulletCharacterManager::getGhostObject(const std::string & scene) {
	if (isPresent(scene))
		return btGhostObject::upcast(controllers[scene].sceneInfo.object);
	return NULL;
}

bool BulletCharacterManager::isPresent(const std::string & scene) {
	return controllers.find(scene) != controllers.end();
}

void BulletCharacterManager::updateCameraPosition(std::string scene, btVector3 position) {
	if (!hasCamera(scene))
		(*cameraPositions)[scene] = new float[4]();
	(*cameraPositions)[scene][0] = position.getX();
	(*cameraPositions)[scene][1] = position.getY();
	(*cameraPositions)[scene][2] = position.getZ();
	(*cameraPositions)[scene][3] = 1.0f;

}


