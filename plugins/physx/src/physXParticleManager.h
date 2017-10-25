#ifndef _PHYSXPARTICLEMANAGER_H
#define _PHYSXPARTICLEMANAGER_H

#include <map>
#include <vector>
#include <stdlib.h>
#include <time.h> 
#include "PxPhysicsAPI.h"
#include "physXScene.h"
#include "physXLuaManager.h"

class PhysXParticleManager
{
public:
	PhysXParticleManager();
	~PhysXParticleManager();

	void update();
	void addParticleSystem(physx::PxScene * world, const std::string &scene, const std::string &material, float maxParticles, float * positions, float *transform);
	void createParticles(std::string scene, int n, float randomFactor);
	std::map<std::string, int> * getParticleSystemsParticleNb();


protected:
	typedef struct {
		externalInfo extInfo;
		physx::PxParticleFluid * particleSystem;
		physx::PxParticleExt::IndexPool* particleIndexPool;
		int currentNbParticles;
		int maxIterStep;
		int iterStep;
		std::string materialName;
	} PhysXParticleSystem;

	std::map<std::string, PhysXParticleSystem> particleSystems;
	PhysXLuaManager * luaManager;

};

#endif