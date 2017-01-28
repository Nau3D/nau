#include "physXCharacterManager.h"

int PhysXCharacterManager::nextControllerIndex = 0;

using namespace physx;

PhysXCharacterManager::PhysXCharacterManager(physx::PxScene * world) {
	manager = PxCreateControllerManager(*world);
	cameraPositions = new std::map<std::string, float*>();
}


PhysXCharacterManager::~PhysXCharacterManager() {
	manager->release();
	delete &manager;
	delete &controllers;
	delete &cameraPositions;
}

void PhysXCharacterManager::update(physx::PxVec3 gravity) {
	for (auto scene : controllers)
		move(scene.first, gravity);
}

void
PhysXCharacterManager::createCharacter(const std::string & scene, physx::PxVec3 position, physx::PxVec3 up, physx::PxMaterial * material, bool isCamera, float radius, float height) {
	PxCapsuleControllerDesc desc;
	desc.height = height;
	desc.radius = radius;
	desc.position = PxExtendedVec3(position.x, position.y, position.z);
	desc.material = material;
	desc.reportCallback = this;
	desc.climbingMode = PxCapsuleClimbingMode::eCONSTRAINED;
	//desc.stepOffset = stepHeight;
	desc.upDirection = up.getNormalized();
	//desc.slopeLimit = cosf(DegToRad(80.0f));
	desc.userData = static_cast<void*> (new std::string(scene));
	manager->createController(desc);
	controllers[scene].isCamera = isCamera;
	controllers[scene].index = PhysXCharacterManager::nextControllerIndex++;
}

void PhysXCharacterManager::updateCameraPosition(std::string cameraName, physx::PxVec3 position) {
	if (!hasCamera(cameraName))
		(*cameraPositions)[cameraName] = new float[4]();

	(*cameraPositions)[cameraName][0] = position.x;
	(*cameraPositions)[cameraName][1] = position.y;
	(*cameraPositions)[cameraName][2] = position.z;
	(*cameraPositions)[cameraName][3] = 1.0f;
}


void PhysXCharacterManager::addCharacter(const std::string & scene, physx::PxMaterial * material, physx::PxVec3 up) {
	createCharacter(
		scene,
		PxMat44(controllers[scene].extInfo.transform).getPosition(),
		up,
		material
	);
	PxMat44 initTrans = PxMat44(controllers[scene].extInfo.transform);
	initTrans.setPosition(PxVec3(0.0f));
	controllers[scene].initialTrans = initTrans;
}

void PhysXCharacterManager::addCamera(const std::string & scene, physx::PxVec3 position, physx::PxVec3 up, float pace, float minPace, float hitMagnitude, float timeStep, float stepOffset, float mass, float radius, float height, physx::PxMaterial * material) {
	createCharacter(scene, position, up, material, true, radius, height);
	updateCameraPosition(scene, position);
	setDirection	(scene, PxVec3(0.0f));
	setPace			(scene, pace);
	setMinPace		(scene, minPace);
	setHitMagnitude	(scene, hitMagnitude);
	setTimeStep		(scene, timeStep);
	setMass			(scene, mass);
}

void PhysXCharacterManager::move(const std::string & scene, physx::PxVec3 gravity) {
	if (controllers.find(scene) != controllers.end()) {
		PxController * controller = manager->getController(controllers[scene].index);
		PxExtendedVec3 auxPosition = controller->getPosition();
		PxVec3 previousPosition = PxVec3((PxReal)auxPosition.x, (PxReal)auxPosition.y, (PxReal)auxPosition.z);
		PxVec3 movement = (*controllers[scene].direction * controllers[scene].pace);
		PxVec3 movementAndGravity = movement + gravity;
		controller->move(movementAndGravity, controllers[scene].minPace, controllers[scene].timeStep, NULL);
		auxPosition = controller->getPosition();
		PxVec3 actualPosition = PxVec3((PxReal)auxPosition.x, (PxReal)auxPosition.y, (PxReal)auxPosition.z);
		if (previousPosition != actualPosition) {
			if (controllers[scene].isCamera) {
				updateCameraPosition(scene, actualPosition);
				controllers[scene].pace *= 0.9f;
			}
			else {
				PxTransform trans;
				if (movement.magnitude() > 0.0f) {
					PxVec3 up = controller->getUpDirection();
					PxVec3 dir = controllers[scene].direction->getNormalized();
					PxVec3 right = up.cross(dir).getNormalized();
					PxMat44 mat = PxMat44(
						PxVec3(right.x, up.x, dir.x),
						PxVec3(right.y, up.y, dir.y),
						PxVec3(right.z, up.z, dir.z),
						PxVec3((PxReal)controller->getPosition().x, (PxReal)controller->getPosition().y, (PxReal)controller->getPosition().z)
					);
					trans = PxTransform(mat * controllers[scene].initialTrans);
					
				}
				else {
					PxMat44 mat(controllers[scene].extInfo.transform);
					mat.setPosition(actualPosition);
					trans = PxTransform(mat);
				}
				getMatFromPhysXTransform(trans, controllers[scene].extInfo.transform);
			}
		}
	}
}

void PhysXCharacterManager::move(const std::string & scene, float * transform, physx::PxVec3 gravity) {
	if (controllers.find(scene) != controllers.end()) {
		PxController * controller = manager->getController(controllers[scene].index);
		controllers[scene].extInfo.transform = transform;
		PxMat44 initTrans = PxMat44(controllers[scene].extInfo.transform);
		PxVec3 pos = PxMat44(controllers[scene].extInfo.transform).getPosition();
		initTrans.setPosition(PxVec3(0.0f));
		controllers[scene].initialTrans = initTrans;
		controller->setPosition(PxExtendedVec3(pos.x, pos.y, pos.z));
	}
}

void PhysXCharacterManager::createInfo(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform) {
	controllers[scene].extInfo = externalInfo(nbVertices, vertices, nbIndices, indices, transform);
}

void PhysXCharacterManager::setDirection(std::string scene, physx::PxVec3 dir) {
	if (controllers[scene].direction) {
		controllers[scene].direction->x = dir.x;
		controllers[scene].direction->y = dir.y;
		controllers[scene].direction->z = dir.z;
	}
	else 
		controllers[scene].direction = new PxVec3(dir);
}

void PhysXCharacterManager::setPace(std::string scene, float pace) {
	controllers[scene].pace = pace;
}

void PhysXCharacterManager::setMinPace(std::string scene, float minPace) {
	controllers[scene].minPace = minPace;
}

void PhysXCharacterManager::setHitMagnitude(std::string scene, float hitMagnitude) {
	controllers[scene].hitMagnitude = hitMagnitude;
}

void PhysXCharacterManager::setMass(std::string scene, float value) {
	manager->getController(controllers[scene].index)->getActor()->setMass(value);
}

void PhysXCharacterManager::setFriction(std::string scene, float value) {
	PxRigidDynamic * actor = manager->getController(controllers[scene].index)->getActor();
	PxU32 nbShapes = actor->getNbShapes();
	std::vector<PxShape *> list(nbShapes);
	actor->getShapes(&list.at(0), nbShapes);
	for (PxShape* shape : list) {
		PxU32 nbMat = shape->getNbMaterials();
		std::vector<PxMaterial *> matList(nbMat);
		shape->getMaterials(&matList.at(0), nbMat);
		for (PxMaterial * mat : matList) {
			mat->setDynamicFriction(value);
			mat->setStaticFriction(value);
		}
	}
}

void PhysXCharacterManager::setRestitution(std::string scene, float value) {
	PxRigidDynamic * actor = manager->getController(controllers[scene].index)->getActor();
	PxU32 nbShapes = actor->getNbShapes();
	std::vector<PxShape *> list(nbShapes);
	actor->getShapes(&list.at(0), nbShapes);
	for (PxShape* shape : list) {
		PxU32 nbMat = shape->getNbMaterials();
		std::vector<PxMaterial *> matList(nbMat);
		shape->getMaterials(&matList.at(0), nbMat);
		for (PxMaterial * mat : matList) {
			mat->setRestitution(value);
		}
	}
}

void PhysXCharacterManager::setHeight(std::string scene, float value) {
	((PxCapsuleController *)manager->getController(controllers[scene].index))->setHeight(value);
}

void PhysXCharacterManager::setRadius(std::string scene, float value) {
	((PxCapsuleController *)manager->getController(controllers[scene].index))->setRadius(value);
}

void PhysXCharacterManager::setStepOffset(std::string scene, float value) {
	manager->getController(controllers[scene].index)->setStepOffset(value);
}

void PhysXCharacterManager::setTimeStep(std::string scene, float value) {
	controllers[scene].timeStep = value;
}

void PhysXCharacterManager::onShapeHit(const physx::PxControllerShapeHit & hit) {
	if (hit.actor->getType() == PxActorType::eRIGID_DYNAMIC) {
		std::string *n = static_cast<std::string*>(hit.controller->getUserData());
		hit.actor->is<PxRigidDynamic>()->addForce(PxVec3(hit.dir) * controllers[*n].hitMagnitude);
	}
}

void PhysXCharacterManager::onControllerHit(const physx::PxControllersHit & hit) {
}

void PhysXCharacterManager::onObstacleHit(const physx::PxControllerObstacleHit & hit) {
}
