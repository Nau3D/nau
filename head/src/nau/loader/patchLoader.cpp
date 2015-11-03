#include "nau/loader/patchLoader.h"

#include "nau.h"
#include "nau/errors.h"
#include "nau/slogger.h"
#include "nau/scene/sceneObjectFactory.h"
#include "nau/geometry/boundingvolumefactory.h"
#include "nau/material/materialGroup.h"
#include "nau/render/IAPISupport.h"

#include <stdlib.h>

using namespace nau::loader;
using namespace nau::geometry;

void PatchLoader::loadScene(nau::scene::IScene *aScene, std::string &aFilename) {

	IAPISupport *sup = IAPISupport::GetInstance();
	if (!sup->apiSupport(IAPISupport::TESSELATION_SHADERS))
		NAU_THROW("Patches are not supported");

	FILE *fp = fopen(aFilename.c_str(),"rt");

	if (fp == NULL) {
		NAU_THROW("Empty file: %s", aFilename.c_str());
	}

	int numPatches, numVert;
	unsigned int verticesPerPatch;
	std::vector<unsigned int> *indices = new std::vector<unsigned int>;
	std::vector<nau::math::vec4>* vertices = new std::vector<nau::math::vec4>;


	fscanf(fp, "%d\n", &verticesPerPatch);
	fscanf(fp, "%d\n", &numPatches);

	char hasIndices[128];
	fscanf(fp,"%s\n", hasIndices);
	if (hasIndices[0] == 'y') {
		unsigned int index;
		for (int i = 0; i < numPatches; ++i) {
			for (unsigned int j = 0; j < verticesPerPatch; ++j) {
		
				fscanf(fp, "%u,", &index);
				indices->push_back(index);
			}
		}
	}

	fscanf(fp," %d\n", &numVert);


	float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;
	float minX =  FLT_MAX, minY =  FLT_MAX, minZ =  FLT_MAX; 

	vec4 v;
	for (int i = 0; i < numVert; ++i) {
		fscanf(fp, "%f, %f, %f\n", &v.x, &v.y, &v.z);
		v.w = 1.0;

		if (v.x > maxX)
			maxX = v.x;
		if (v.x < minX)
			minX = v.x;

		if (v.y > maxY)
			maxY = v.y;
		if (v.y < minY)
			minY = v.y;

		if (v.z > maxZ)
			maxZ = v.z;
		if (v.z < minZ)
			minZ = v.z;
		vertices->push_back(v);
	}
	fclose(fp);

	float center[3];
	center[0] = (maxX + minX) * 0.5f;
	center[1] = (maxY + minY) * 0.5f;
	center[2] = (maxZ + minZ) * 0.5f;

	float tmp;
	tmp = maxX - minX;
	tmp = maxY - minY > tmp? maxY - minY:tmp;
	tmp = maxZ - minZ > tmp? maxZ - minZ:tmp;

	SceneObject *anObject = SceneObjectFactory::create("SimpleObject"); 
	anObject->setName(aFilename);

	IBoundingVolume *aBoundingVolume = BoundingVolumeFactory::create("BoundingBox");
	aBoundingVolume->set(vec3(minX,minY,minZ), vec3(maxX, maxY, maxZ));
	anObject->setBoundingVolume(aBoundingVolume);

	IRenderable *aRenderable;
	aRenderable = RESOURCEMANAGER->createRenderable("Mesh", "patch", aFilename);

	VertexData *vData = &(aRenderable->getVertexData());
	vData->setDataFor(VertexData::GetAttribIndex(std::string("position")), vertices);

	MaterialGroup *aMatGroup = MaterialGroup::Create(aRenderable, "dirLightDifAmbPix");
	//aMatGroup->setParent(aRenderable);
	//aMatGroup->setMaterialName("dirLightDifAmbPix");
	if (hasIndices[0] == 'y')
		aMatGroup->setIndexList(indices);

	aRenderable->addMaterialGroup(aMatGroup);
	aRenderable->setDrawingPrimitive(IRenderable::PATCHES);
	aRenderable->setNumberOfVerticesPerPatch(verticesPerPatch);

	anObject->setRenderable(aRenderable);

	aScene->add(anObject);
}