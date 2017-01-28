#include "physXParticleManager.h"

using namespace physx;

PhysXParticleManager::PhysXParticleManager() {
	std::srand(static_cast<unsigned int>(time(NULL)));
	luaManager = new PhysXLuaManager();
}


PhysXParticleManager::~PhysXParticleManager() {
}

void PhysXParticleManager::update() {
	for (auto scene : particleSystems) {

		if ((scene.second.currentNbParticles < scene.second.extInfo.nbVertices) && ((particleSystems[scene.first].iterStep++ % scene.second.maxIterStep) == 0)) {
				createParticles(scene.first, 100, 0.3f);
		}

		PxParticleFluidReadData* rd = scene.second.particleSystem->lockParticleFluidReadData();

		if (rd) {
			PxU32 nPart = rd->nbValidParticles;
			//PxVec4* newPositions = new PxVec4[nPart];

			PxStrideIterator<const PxParticleFlags> flagsIt(rd->flagsBuffer);
			PxStrideIterator<const PxVec3> positionIt(rd->positionBuffer);

			int index = 0;
			for (unsigned i = 0; i < rd->validParticleRange; ++i, ++flagsIt, ++positionIt) {
				if (*flagsIt & PxParticleFlag::eVALID) {
					const PxVec3& position = *positionIt;
					scene.second.extInfo.vertices[index*4] = position.x;
					scene.second.extInfo.vertices[(index*4) + 1] = position.y;
					scene.second.extInfo.vertices[(index*4) + 2] = position.z;
					scene.second.extInfo.vertices[(index*4) + 3] = 1.0f;
					index++;
					//newPositions[index++] = PxVec4(position, 1.0f);
					//scene.second.positions->push_back(position.x);
					//scene.second.positions->push_back(position.y);
					//scene.second.positions->push_back(position.z);
					//scene.second.positions->push_back(1.0f);
				}
			}
			


			rd->unlock();
		}
	}
}

void PhysXParticleManager::addParticleSystem(physx::PxScene * world, const std::string &scene, const std::string &material, float maxParticles, float * positions, float *transform) {
	int max = static_cast<int>(maxParticles);
	particleSystems[scene].extInfo = externalInfo(max, positions, 0, NULL, transform);
	particleSystems[scene].materialName = material;
	particleSystems[scene].currentNbParticles = 0;
	particleSystems[scene].maxIterStep = 1;
	particleSystems[scene].iterStep = 1;

	PxPhysics *gPhysics = &(world->getPhysics());


	particleSystems[scene].particleSystem = gPhysics->createParticleFluid(max);

	// TODO: Get Properties from PropertiesManager;
	float particleDistance = 0.01f;
	//particleSystems[scene].particleSystem->setGridSize(5.0f);
	particleSystems[scene].particleSystem->setMaxMotionDistance(0.3f);
	//particleSystem->setRestOffset(particleDistance*0.3f);
	particleSystems[scene].particleSystem->setRestOffset(0.036f);
	//particleSystem->setContactOffset(particleDistance*0.3f * 2);
	particleSystems[scene].particleSystem->setContactOffset(0.04f);
	particleSystems[scene].particleSystem->setDamping(0.0f);
	particleSystems[scene].particleSystem->setRestitution(0.3f);
	particleSystems[scene].particleSystem->setDynamicFriction(0.001f);
	particleSystems[scene].particleSystem->setRestParticleDistance(particleDistance);
	particleSystems[scene].particleSystem->setViscosity(60.0f);
	particleSystems[scene].particleSystem->setStiffness(45.0f);
	//particleSystems[scene].particleSystem->setParticleReadDataFlag(PxParticleReadDataFlag::eVELOCITY_BUFFER, true);

	particleSystems[scene].particleSystem->userData = static_cast<void*> (new std::string(scene));

	if (particleSystems[scene].particleSystem)
		world->addActor(*particleSystems[scene].particleSystem);

	particleSystems[scene].particleIndexPool = PxParticleExt::createIndexPool(max);
	//createParticles();
	luaManager->initLua("C:\\Users\\david\\Development\\physics\\nau\\head\\projects\\physics\\createParticles.lua");
	//luaManager->initLua("createParticles.lua");
}

float getRandomNumber(float factor) {
	return ((std::rand() % int((factor*2) *100.f)) / 100.0f) - factor;
}

void PhysXParticleManager::createParticles(std::string scene, int n, float randomFactor) {
	PxU32 existingParticles = particleSystems[scene].currentNbParticles;
	int nb = particleSystems[scene].currentNbParticles;
	std::vector<PxU32> mTmpIndexArray;
	int newNb;
	PxTransform trans = PxTransform(PxMat44(const_cast<float*> (particleSystems[scene].extInfo.transform)));
	std::vector<PxVec3> * partPositions = new std::vector<PxVec3>();

	if (luaManager->isFileLoaded()) {
		luaManager->createNewParticles("create");
		PhysXLuaManager::NewParticles newParts = luaManager->getCurrentNewParticles();

		nb += newParts.nbParticle;
		mTmpIndexArray.resize(nb);
		PxStrideIterator<PxU32> indexData(&mTmpIndexArray[0]);

		newNb = particleSystems[scene].particleIndexPool->allocateIndices((nb - existingParticles), indexData);
		for (int i = 0; i < newNb; i++) {
			partPositions->push_back(
				PxVec3(
					newParts.positions[i * 3],
					newParts.positions[(i * 3) + 1],
					newParts.positions[(i * 3) + 2]
				)
			);
		}
	}
	else {
		nb += n;
		mTmpIndexArray.resize(nb);
		PxStrideIterator<PxU32> indexData(&mTmpIndexArray[0]);

		newNb = particleSystems[scene].particleIndexPool->allocateIndices((nb - existingParticles), indexData);
		for (int i = 0; i < newNb; i++) {
			partPositions->push_back(PxVec3(trans.p.x + getRandomNumber(randomFactor), trans.p.y, trans.p.z + getRandomNumber(randomFactor)));
		}
	}

	PxParticleCreationData particleCreationData;
	particleCreationData.numParticles = newNb;
	particleCreationData.indexBuffer = PxStrideIterator<const PxU32>(&mTmpIndexArray[0]);
	particleCreationData.positionBuffer = PxStrideIterator<const PxVec3>(&partPositions->at(0));
	bool ok = particleSystems[scene].particleSystem->createParticles(particleCreationData);

	particleSystems[scene].currentNbParticles += newNb;
}

std::map<std::string, int>* PhysXParticleManager::getParticleSystemsParticleNb() {
	std::map<std::string, int> * result = new std::map<std::string, int>();
	for (auto scene : particleSystems) {
		result->emplace(scene.second.materialName, scene.second.currentNbParticles);
	}
	return result;
}
