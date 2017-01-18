#ifndef _BULLETSCENE_H
#define _BULLETSCENE_H

#include "btBulletDynamicsCommon.h"
#include <string>

typedef struct ExternalInfo {
	int nbVertices;
	float *vertices;
	int nbIndices;
	unsigned int *indices;
	float *transform;

	ExternalInfo() {
		nbVertices = 0;
		vertices = 0;
		nbIndices = 0;
		indices = 0;
		transform = 0;
	};

	ExternalInfo(int nbVert, float *vert, int nbInd, unsigned int *ind, float *transf) {
		nbVertices = nbVert;
		vertices = vert;
		nbIndices = nbInd;
		indices = ind;
		transform = transf;
	};
} externalInfo;

typedef struct {
	externalInfo extInfo;
	btCollisionObject * object;
} BulletScene;

#endif