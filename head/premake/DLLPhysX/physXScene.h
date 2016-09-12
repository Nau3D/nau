#ifndef _PHYSXSCENE_H
#define _PHYSXSCENE_H

#include "PxPhysicsAPI.h"
#include <string>
#include <vector>


typedef struct ExternalInfo {
	int nbVertices;
	float * vertices;
	int nbIndices;
	unsigned int * indices;
	float * transform;

	ExternalInfo() {
		nbVertices		= 0;
		vertices		= 0;
		nbIndices		= 0;
		indices			= 0;
		transform		= 0;
	};

	ExternalInfo(int nbVert, float * vert, int nbInd, unsigned int * ind, float * transf) {
		nbVertices		= nbVert;
		vertices		= vert;
		nbIndices		= nbInd;
		indices			= ind;
		transform		= transf;
	};
} externalInfo;

/*typedef struct ExternalParticles {
	float maxParticles;
	float * nbParticles;
	float * positions;
	float * transform;

	/*ExternalParticles() {
		maxParticles	= 0;
		nbParticles		= 0;
		positions		= 0;
		transform		= 0;
	};

	ExternalParticles(float * maxParts, float * nbParts, float * pos, float * transf) {
		maxParticles	= maxParts;
		nbParticles		= nbParts;
		positions		= pos;
		transform		= transf;
	};

} externalParticles;*/


typedef struct {

	//union {
		externalInfo extInfo;
	//	externalParticles extParticles;
	//};

	physx::PxActor* actor;

} PhysXScene;


void getMatFromPhysXTransform(physx::PxTransform transform, float* matrix);
void getMatFromPhysXTransform(physx::PxMat44 transform, float* matrix);

#endif