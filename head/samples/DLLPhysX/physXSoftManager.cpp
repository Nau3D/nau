#include "physXSoftManager.h"

using namespace physx;

PhysXSoftManager::PhysXSoftManager() {
}


PhysXSoftManager::~PhysXSoftManager() {
}

void PhysXSoftManager::update() {
	for (auto scene : softBodies) {
		PxCloth * cloth = scene.second.info.actor->is<PxCloth>();
		if (cloth) {
			getMatFromPhysXTransform(cloth->getGlobalPose(), scene.second.info.extInfo.transform);
			PxClothParticleData* pData = cloth->lockParticleData();
			PxClothParticle* pParticles = pData->particles;
			float * points = scene.second.info.extInfo.vertices;

			for (int i = 0; i < scene.second.info.extInfo.nbVertices; i++) {
				points[4 * i] = pParticles[i].pos.x;
				points[(4 * i) + 1] = pParticles[i].pos.y;
				points[(4 * i) + 2] = pParticles[i].pos.z;
			}
			pData->unlock();
		}
	}
}

void PhysXSoftManager::createInfo(const std::string & scene, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform, int condition, float * conditionValue) {
	softBodies[scene].info.extInfo = externalInfo(nbVertices, vertices, nbIndices, indices, transform);
	softBodies[scene].condition = condition;
	softBodies[scene].contidionPlane = PxPlane(conditionValue[0], conditionValue[1], conditionValue[2], conditionValue[3]);
}

PxClothParticle PhysXSoftManager::createClothParticle(std::string scene, physx::PxVec3 vertice) {
	float invMass;
	switch (softBodies[scene].condition)
	{
	case CLOTH_CONDITION_GT:
		invMass = softBodies[scene].contidionPlane.distance(vertice) > 0.0f ? 0.0f : 1.0f;
		break;
	case CLOTH_CONDITION_LT:
		invMass = softBodies[scene].contidionPlane.distance(vertice) < 0.0f ? 0.0f : 1.0f;
		break;
	case CLOTH_CONDITION_EGT:
		invMass = softBodies[scene].contidionPlane.contains(vertice) || softBodies[scene].contidionPlane.distance(vertice) > 0.0f ? 0.0f : 1.0f;
		break;
	case CLOTH_CONDITION_ELT:
		invMass = softBodies[scene].contidionPlane.contains(vertice) || softBodies[scene].contidionPlane.distance(vertice) < 0.0f ? 0.0f : 1.0f;
		break;
	case CLOTH_CONDITION_EQ:
		invMass = softBodies[scene].contidionPlane.contains(vertice) ? 0.0f : 1.0f;
		break;
	default:
		invMass = 1.0f;
		break;
	}
	return PxClothParticle(vertice, invMass);
}

void PhysXSoftManager::addSoftBody(physx::PxScene * world, const std::string & scene) {
	int nbVert = softBodies[scene].info.extInfo.nbVertices;
	float * verts = softBodies[scene].info.extInfo.vertices;
	PxPhysics *gPhysics = &(world->getPhysics());

	PxDefaultMemoryOutputStream writeBuffer;

	int stride = 4 * sizeof(float);
	PxClothParticle *particles = new PxClothParticle[nbVert];

	PxClothMeshDesc meshDesc;
	PxClothParticle *ptls = particles;
	for (int i = 0; i < nbVert; i++)
		ptls[i] = createClothParticle(scene, PxVec3(verts[4*i], verts[(4*i)+1], verts[(4*i)+2]));

	meshDesc.points.data = reinterpret_cast<const unsigned char *>(verts);
	meshDesc.points.count = nbVert;
	meshDesc.points.stride = 4 * sizeof(float);

	meshDesc.invMasses.data = &particles->invWeight;
	meshDesc.invMasses.count = nbVert;
	meshDesc.invMasses.stride = sizeof(PxClothParticle);

	meshDesc.triangles.data = reinterpret_cast<const unsigned char *>(softBodies[scene].info.extInfo.indices);
	meshDesc.triangles.count = softBodies[scene].info.extInfo.nbIndices / 3;
	meshDesc.triangles.stride = 3 * sizeof(unsigned int);

	PxClothFabric* fabric = PxClothFabricCreate(*gPhysics, meshDesc, world->getGravity().getNormalized());
	PxTransform pose = PxTransform(PxMat44(softBodies[scene].info.extInfo.transform));
	PxClothFlags flags = PxClothFlags();

	flags |= PxClothFlag::eGPU;
	flags |= PxClothFlag::eSCENE_COLLISION;
	//flags |= PxClothFlag::eSWEPT_CONTACT;
	
	PxCloth* cloth = gPhysics->createCloth(pose, *fabric, particles, flags);
	cloth->userData = static_cast<void*> (new std::string(scene));

	world->addActor(*cloth);
	
	softBodies[scene].info.actor = cloth;
}

void PhysXSoftManager::move(std::string scene, float * transform) {
	PxCloth * cloth = getCloth(scene);
	if (cloth) {
		softBodies[scene].info.extInfo.transform = transform;
		cloth->setGlobalPose(PxTransform(PxMat44(const_cast<float*> (transform))));
	}
}

void PhysXSoftManager::setVerticalStretch(const std::string & scene, float * value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setStretchConfig(PxClothFabricPhaseType::eVERTICAL, *getClothStretchConfig(value));
}

void PhysXSoftManager::setHorizontalStretch(const std::string & scene, float * value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setStretchConfig(PxClothFabricPhaseType::eHORIZONTAL, *getClothStretchConfig(value));
}

void PhysXSoftManager::setShearing(const std::string & scene, float * value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setStretchConfig(PxClothFabricPhaseType::eSHEARING, *getClothStretchConfig(value));
}

void PhysXSoftManager::setBending(const std::string & scene, float * value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setStretchConfig(PxClothFabricPhaseType::eBENDING, *getClothStretchConfig(value));
}

void PhysXSoftManager::setInertiaScale(const std::string & scene, float value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setInertiaScale(value);
}

void PhysXSoftManager::setSolverFrequency(const std::string & scene, float value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setSolverFrequency(value);
}

void PhysXSoftManager::setFrictionCoefficient(const std::string & scene, float value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setFrictionCoefficient(value);
}

void PhysXSoftManager::setCollisionMassScale(const std::string & scene, float value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setCollisionMassScale(value);
}

void PhysXSoftManager::setSelfCollisionDistance(const std::string & scene, float value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setSelfCollisionDistance(value);
}

void PhysXSoftManager::setSelfCollisionStiffness(const std::string & scene, float value) {
	PxCloth * cloth = getCloth(scene);
	if (cloth)
		cloth->setSelfCollisionStiffness(value);
}

physx::PxCloth * PhysXSoftManager::getCloth(const std::string & scene) {
	if (softBodies.find(scene) != softBodies.end())
		return softBodies[scene].info.actor->is<PxCloth>();
	return NULL;
}

physx::PxClothStretchConfig * PhysXSoftManager::getClothStretchConfig(float * value) {
	PxClothStretchConfig * stretchConfig = new PxClothStretchConfig();
	stretchConfig->stiffness			= value[0];
	stretchConfig->stiffnessMultiplier	= value[1];
	stretchConfig->compressionLimit		= value[2];
	stretchConfig->stretchLimit			= value[3];
	return stretchConfig;
}
