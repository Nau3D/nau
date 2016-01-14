#include "nau/loader/cboLoader.h"

#include "nau.h"
#include "nau/clogger.h"
#include "nau/slogger.h"
#include "nau/geometry/boundingvolumefactory.h"
#include "nau/geometry/iBoundingVolume.h"
#include "nau/material/materialGroup.h"
#include "nau/material/iState.h"
#include "nau/math/matrix.h"
#include "nau/math/vec3.h"
#include "nau/render/iRenderable.h"
#include "nau/scene/sceneObject.h"
#include "nau/scene/sceneObjectFactory.h"
#include "nau/system/file.h"

#include <assert.h>
#include <fstream>
#include <map>


using namespace nau::loader;
using namespace nau::scene;
using namespace nau::math;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;
using namespace nau::system;

std::string CBOLoader::m_FileName;

void
CBOLoader::_writeVertexData (std::shared_ptr<VertexData>& aVertexData, std::fstream &f) {

	unsigned int siz;
	unsigned int sizeVec;
	unsigned int countFilledArrays = 0;

	for (int i = 0; i < VertexData::MaxAttribs; i++) {
	
		std::shared_ptr<std::vector<VertexData::Attr>> &aVec = aVertexData->getDataOf (i);
		if (aVec->size())
			countFilledArrays++;
	}

	// write number of filled arrays
	f.write (reinterpret_cast<char *> (&countFilledArrays), sizeof (countFilledArrays));


	for (int i = 0; i < VertexData::MaxAttribs; i++) {

		std::shared_ptr<std::vector<VertexData::Attr>> &aVec = aVertexData->getDataOf (i);
		sizeVec = (unsigned int)aVec->size();
		if (sizeVec > 0) {

			_writeString(VertexData::Syntax[i],f);
			// write size of array
			f.write (reinterpret_cast<char *> (&sizeVec), sizeof (sizeVec));
			// write attribute data
			f.write (reinterpret_cast<char *> (&(aVec.get()[0])), 
					 sizeVec * sizeof(VertexData::Attr));
		}
	}

	//std::vector<unsigned int> &aVec = aVertexData.getIndexData();
	//size = aVec.size();
	siz = 0;
	f.write (reinterpret_cast<char *> (&siz), sizeof (siz));

	//if (aVec.size() > 0) {
	//	f.write (reinterpret_cast<char *> (&(aVec[0])), 
	//			 static_cast<std::streamsize> (size) * sizeof(unsigned int));
	//}
}


void
CBOLoader::_writeIndexData (std::shared_ptr<nau::geometry::IndexData>& aVertexData, std::fstream &f) {

	unsigned int siz;

	unsigned int countFilledArrays = 0;

	// write number of filled arrays
	f.write (reinterpret_cast<char *> (&countFilledArrays), sizeof (countFilledArrays));


	std::shared_ptr<std::vector<unsigned int>> &aVec = aVertexData->getIndexData();
	siz = (unsigned int)aVec->size();
	f.write (reinterpret_cast<char *> (&siz), sizeof (siz));

	if (aVec->size() > 0) {
		f.write (reinterpret_cast<char *> (&(aVec->at(0))), siz * sizeof(unsigned int));
	}
}


void
CBOLoader::_readVertexData (std::shared_ptr<VertexData>& aVertexData, std::fstream &f) {

	unsigned int siz;
	unsigned int countFilledArrays;
	char buffer[1024];

	f.read(reinterpret_cast<char *> (&countFilledArrays), sizeof (countFilledArrays));

	for (unsigned int i = 0; i < countFilledArrays; i++) {

		// read attribs name
		_readString(buffer, f);

		// read size of array
		f.read (reinterpret_cast<char *> (&siz), sizeof (siz));

		std::shared_ptr<std::vector<VertexData::Attr>> aVector = 
			std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(siz));

		// read attrib data
		f.read (reinterpret_cast<char *> (&(*aVector)[0]), siz * sizeof (VertexData::Attr));

		unsigned int index = VertexData::GetAttribIndex(std::string(buffer));
		aVertexData->setDataFor (index, aVector);
	}
	
	f.read (reinterpret_cast<char *> (&siz), sizeof (siz));
}


void
CBOLoader::_readIndexData (std::shared_ptr<nau::geometry::IndexData>& anIndexData, std::fstream &f) {

	unsigned int siz;
	unsigned int countFilledArrays;

	f.read(reinterpret_cast<char *> (&countFilledArrays), sizeof (countFilledArrays));

	f.read (reinterpret_cast<char *> (&siz), sizeof (siz));
	if (siz > 0) {
		std::shared_ptr<std::vector<unsigned int>> &aNewVector = 
			std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>(siz));

		f.read (reinterpret_cast<char *> (&(*aNewVector)[0]), siz * sizeof (unsigned int));

		anIndexData->setIndexData (aNewVector);
	}
}

//void
//CBOLoader::_ignoreVertexData (std::fstream &f)
//{
//	unsigned int siz;
//
//	for (int i = 0; i <= VertexData::MaxAttribs; i++) {
//		f.read (reinterpret_cast<char *> (&siz), sizeof (siz));
//
//		if (siz > 0) {
//			f.ignore (siz * sizeof (vec3));
//		}
//	}
//	
//	f.read (reinterpret_cast<char *> (&siz), sizeof (siz));
//	if (siz > 0) {
//		f.ignore (siz * sizeof (unsigned int));
//	}
//
//}

void
CBOLoader::_writeString (const std::string& aString, std::fstream &f) {

	unsigned int siz = (unsigned int)aString.size();

	f.write (reinterpret_cast<char *> (&siz), sizeof (siz));
	f.write (aString.c_str(), siz + 1);
}


void
CBOLoader::_readString ( char *buffer, std::fstream &f) {

	unsigned int siz;

	memset (buffer, 0, 1024);
	f.read (reinterpret_cast<char *> (&siz), sizeof (siz));
	f.read (buffer, siz + 1);
}


void
CBOLoader::_ignoreString (std::fstream &f) {

	unsigned int siz;
	f.read (reinterpret_cast<char *> (&siz), sizeof (siz));
	f.ignore (siz + 1);
}


void 
CBOLoader::loadScene (nau::scene::IScene *aScene, std::string &aFilename, std::string &params) {

	//CLogger::getInstance().addLog(LEVEL_INFO, "debug.txt");

	m_FileName = aFilename;
	std::string path = File::GetPath(aFilename);

	std::fstream f (aFilename.c_str(), std::fstream::in | std::fstream::binary);

	std::map<std::string, std::shared_ptr<IRenderable>> renderables; 
	//std::map<std::pair<std::string, std::string>, int> materialTrack;

	if (!f.is_open()) {
		NAU_THROW ("Cannot open file: %s", aFilename.c_str());
		return;
	}

	char buffer[1024];
	unsigned int nObjects;
	unsigned int nMatGroups, nMats;

	// MATERIALS
	//LOG_INFO ("[Reading] Materials start");
	f.read (reinterpret_cast<char *> (&nMats), sizeof(nMats));
	for (unsigned int i = 0 ; i < nMats; i++) {
		
		_readMaterial(path,f);
	}
	//LOG_INFO ("[Reading] Materials done");
	//GEOMETRY
	_readString (buffer, f);

	if (!strcmp(buffer,"OctreeByMatScene")) {
	
		_readOctreeByMat((OctreeByMatScene *)aScene, f);
		f.close();
		return;
	}
	//LOG_INFO ("[Reading] Scene type: [%s]", buffer);
	f.read (reinterpret_cast<char *> (&nObjects), sizeof(nObjects));
	//LOG_INFO ("[Reading] Number of Objects: [%d]", nObjects);

	for (unsigned int i = 0; i < nObjects; i++) {

		_readString (buffer, f);
		//LOG_INFO ("[Reading] Type of object: [%s]", buffer);
		std::shared_ptr<SceneObject> &aObject = SceneObjectFactory::Create(buffer);

		_readString (buffer, f);
		aObject->setName (buffer);

		aObject->readSpecificData (f);

		_readString (buffer, f);
		//LOG_INFO ("[Reading] Type of BoundingVolume: [%s]", buffer);
		IBoundingVolume *aBoundingVolume = BoundingVolumeFactory::create (buffer);
		//aObject->setBoundingVolume (aBoundingVolume);

		std::vector<vec3>& points = aBoundingVolume->getPoints();
		unsigned int nPoints;
		f.read (reinterpret_cast<char *> (&nPoints), sizeof (nPoints));
		//LOG_INFO ("[Reading] Number of points [%d]", nPoints);
		float *flo = new float[nPoints*3];
		f.read (reinterpret_cast<char *> (flo), nPoints * 3 * sizeof(float));
		for (unsigned int k = 0; k < nPoints; ++k)
			points[k] = vec3(flo[k * 3], flo[k * 3 + 1], flo[k * 3 + 2]);
		delete flo;

		mat4 mat; 
		f.read (reinterpret_cast<char *> (const_cast<float*>(mat.getMatrix())), sizeof(float)*16);

		//for (int i = 0; i < 16; i++) {
		//	LOG_INFO ("Matrix(%d): [%f]", i, mat.getMatrix()[i]);
		//}

		//aTransform->setMat44 (mat);
		aObject->setTransform (mat);

		_readString (buffer, f);
		//LOG_INFO ("[Reading] Renderable's name: [%s]", buffer);
		std::string renderableName(buffer);

		if (0 == renderableName.compare ("NULLOBJECT")) {
			continue;
		}

		std::shared_ptr<IRenderable> aRenderable;

		if (0 == renderables.count (renderableName)) {
			/*Create the new renderable */

			_readString (buffer, f);
			aRenderable = RESOURCEMANAGER->createRenderable(buffer,renderableName,aFilename);
			//aRenderable->setName (renderableName);
			//RESOURCEMANAGER->addRenderable(aRenderable,aFilename);
			//assert (0 == aRenderable);
			unsigned int primitive;
			if (params.find("USE_ADJACENCY") != std::string::npos) 
				primitive = IRenderable::TRIANGLES_ADJACENCY;
			else 
				primitive = IRenderable::TRIANGLES;
			aRenderable->setDrawingPrimitive(primitive);
			renderables[renderableName] = aRenderable;
			
			std::shared_ptr<VertexData> &vertexData = aRenderable->getVertexData();
			_readVertexData (vertexData, f);

			//LOG_INFO ("[Reading] Renderable type: [%s]", buffer);

			f.read (reinterpret_cast<char *> (&nMatGroups), sizeof (nMatGroups));
			for (unsigned int i = 0; i < nMatGroups; i++) {

				_readString (buffer, f);
				//LOG_INFO ("[Reading] Material Groups name: [%s]", buffer);

				std::shared_ptr<MaterialGroup> aMatGroup = MaterialGroup::Create(aRenderable.get(), buffer);
				if (primitive == IRenderable::TRIANGLES_ADJACENCY)
					aMatGroup->getIndexData()->useAdjacency(true);
				//aMatGroup->setMaterialName (buffer);				
				//aMatGroup->setParent (aRenderable);


				std::shared_ptr<IndexData> &indexData = aMatGroup->getIndexData();
				_readIndexData (indexData, f);
				
				//_readString (buffer, f);
				//LOG_INFO ("[Reading] Material name: [%s]", buffer);
				aRenderable->getMaterialGroups().push_back (aMatGroup);
			}

		} else {
			/*Shared geometry*/
			aRenderable = renderables[renderableName];
		}

		//assert (0 == aRenderable);
		aObject->setRenderable (aRenderable);

		aScene->add (aObject);
	}
}


void
CBOLoader::_readOctreeByMatSceneObject(std::shared_ptr<SceneObject> &so, std::fstream &f) {

	char buffer[1024];

	_readString(buffer,f);
	so->setName(buffer);

	// read the bounding volume

	//LOG_INFO ("[Reading] Type of BoundingVolume: [%s]", buffer);
	BoundingBox *aBoundingVolume = (BoundingBox *)BoundingVolumeFactory::create ("BoundingBox");

	float *flo = new float[6];
	f.read(reinterpret_cast<char *> (flo), 6 * sizeof(float));

	//LOG_INFO ("[Reading] Number of points [%d]", nPoints);
	aBoundingVolume->set(vec3(flo[0], flo[1], flo[2]), vec3(flo[3], flo[4], flo[5]));
	so->setBoundingVolume (aBoundingVolume);

	mat4 mat; 
	f.read (reinterpret_cast<char *> (const_cast<float*>(mat.getMatrix())), sizeof(float)*16);
	//aTransform->setMat44 (mat);
	so->setTransform (mat);

	_readString(buffer,f);
	std::shared_ptr<IRenderable> &aRenderable = RESOURCEMANAGER->createRenderable("Mesh", buffer, m_FileName);
			
	std::shared_ptr<VertexData> &vertexData = aRenderable->getVertexData();
	_readVertexData (vertexData, f);

	//LOG_INFO ("[Reading] Renderable type: [%s]", buffer);
	_readString (buffer, f);
		//LOG_INFO ("[Reading] Material Groups name: [%s]", buffer);


	std::shared_ptr<MaterialGroup> aMatGroup = MaterialGroup::Create(aRenderable.get(), buffer);
	//aMatGroup->setParent (aRenderable);
	//aMatGroup->setMaterialName (buffer);				


	std::shared_ptr<nau::geometry::IndexData> &indexData = aMatGroup->getIndexData();
	_readIndexData (indexData, f);
				
		//_readString (buffer, f);
		//LOG_INFO ("[Reading] Material name: [%s]", buffer);
	aRenderable->getMaterialGroups().push_back (aMatGroup);

	so->setRenderable(aRenderable);

}


void 
CBOLoader::_writeOctreeByMatSceneObject(std::shared_ptr<SceneObject> &so, std::fstream &f) {

	_writeString (so->getName(), f); 
	
	/* Write the bounding volume */
	const IBoundingVolume *aBoundingVolume = so->getBoundingVolume();
	BoundingBox *b = (BoundingBox *)aBoundingVolume;

	std::vector<vec3> &points = b->getNonTransformedPoints();
	float flo[6];
	flo[0] = points[0].x;
	flo[1] = points[0].y;
	flo[2] = points[0].z;
	flo[3] = points[1].x;
	flo[4] = points[1].y;
	flo[5] = points[1].z;
	f.write(reinterpret_cast<char*> (flo), 6 * sizeof (float));

	/* Write the transform */
	 mat4 aTransform = so->getTransform();
	 f.write(reinterpret_cast<char*> ((float *)aTransform.getMatrix()), sizeof(float) * 16);


	 std::shared_ptr<IRenderable> &aRenderablePtr = so->getRenderable();
	_writeString(aRenderablePtr->getName(),f);
	/* Vertices data */
	std::shared_ptr<VertexData> &aVertexData = aRenderablePtr->getVertexData();

	_writeVertexData (aVertexData, f);

	/* Material groups */
	std::vector<std::shared_ptr<MaterialGroup>>& materialGroups = aRenderablePtr->getMaterialGroups();

	_writeString (materialGroups[0]->getMaterialName(), f);

	/* Indices Data */
	std::shared_ptr<nau::geometry::IndexData> &mgIndexData = materialGroups[0]->getIndexData();

	_writeIndexData (mgIndexData, f);
}


void
CBOLoader::_readOctreeByMatNode(std::shared_ptr<OctreeByMatNode> &n, std::fstream &f) {

	char buffer[1024];

	float flo[12];
	f.read (reinterpret_cast<char *> (flo), 12 * sizeof(float));
	n->m_BoundingVolume.set(vec3(flo[0], flo[1], flo[2]), vec3(flo[3], flo[4], flo[5]));
	
	n->m_TightBoundingVolume.set(vec3(flo[6], flo[7], flo[8]), vec3(flo[9], flo[10], flo[11]));

	unsigned int siz;
	f.read (reinterpret_cast<char *> (&siz), sizeof(siz));

	for (unsigned int i = 0; i < siz; ++i) {
		_readString(buffer, f);
		std::shared_ptr<SceneObject> &so = SceneObjectFactory::Create ("SimpleObject");
		so->setName(buffer);
		_readOctreeByMatSceneObject(so, f);
		n->m_pLocalMeshes[buffer] = so;
	}

	f.read (reinterpret_cast<char *> (&n->m_ChildCount), sizeof(n->m_ChildCount));
	f.read (reinterpret_cast<char *> (&n->m_NodeId), sizeof(n->m_NodeId));
	f.read (reinterpret_cast<char *> (&n->m_NodeDepth), sizeof(n->m_NodeDepth));

	if (n->m_ChildCount)
		n->m_Divided = true;

	for (int i = 0; i < n->m_ChildCount; ++i) {
		std::shared_ptr<OctreeByMatNode> &o = std::shared_ptr<OctreeByMatNode>(new OctreeByMatNode());
		_readOctreeByMatNode(o,f);
		o->m_pParent = n;
		n->m_pChilds[o->m_NodeId] = o;
	}
}


void
CBOLoader::_writeOctreeByMatNode(std::shared_ptr<OctreeByMatNode> &n, std::fstream &f) {

	unsigned int siz;

	BoundingBox& aBoundingVolume = (BoundingBox &)n->m_BoundingVolume;
	std::vector<vec3>& points = aBoundingVolume.getNonTransformedPoints();
	float flo[6];
	flo[0] = points[0].x;
	flo[1] = points[0].y;
	flo[2] = points[0].z;
	flo[3] = points[1].x;
	flo[4] = points[1].y;
	flo[5] = points[1].z;
	f.write(reinterpret_cast<char*> (flo), 6 * sizeof(float));

	aBoundingVolume = (BoundingBox &)n->m_TightBoundingVolume;
	points = aBoundingVolume.getNonTransformedPoints();
	flo[0] = points[0].x;
	flo[1] = points[0].y;
	flo[2] = points[0].z;
	flo[3] = points[1].x;
	flo[4] = points[1].y;
	flo[5] = points[1].z;
	f.write(reinterpret_cast<char*> (flo), 6 * sizeof(float));


	siz = (unsigned int)n->m_pLocalMeshes.size();
	f.write (reinterpret_cast<char *> (&siz), sizeof(siz));

	for (auto so: n->m_pLocalMeshes) {
		_writeString(so.first, f);
		_writeOctreeByMatSceneObject(so.second, f);
	}

	f.write (reinterpret_cast<char *> (&n->m_ChildCount), sizeof(n->m_ChildCount));
	f.write (reinterpret_cast<char *> (&n->m_NodeId), sizeof(n->m_NodeId));
	f.write (reinterpret_cast<char *> (&n->m_NodeDepth), sizeof(n->m_NodeDepth));

	for (int i = 0; i < 8; ++i)
		if (n->m_pChilds[i] != NULL)
			_writeOctreeByMatNode(n->m_pChilds[i],f);
}


void
CBOLoader::_readOctreeByMat(OctreeByMatScene *aScene, std::fstream &f) {

	char buffer[1024];
	/* Read bounding box */
	float flo[6];
	f.read (reinterpret_cast<char *> (flo), 6 * sizeof (float));

	aScene->m_BoundingBox.set(vec3(flo[0], flo[1], flo[2]), vec3(flo[3], flo[4], flo[5]));

	OctreeByMat *o = new OctreeByMat();
	aScene->m_pGeometry = o;
	_readString(buffer,f);
	//aScene->setName(buffer);
	o->setName(buffer);

	std::shared_ptr<OctreeByMatNode> n = std::shared_ptr<OctreeByMatNode>(new OctreeByMatNode());
	_readOctreeByMatNode(n,f);
	o->m_pOctreeRootNode = n;
}


void
CBOLoader::_writeOctreeByMat(OctreeByMatScene *aScene, std::fstream &f) {

	/* Write the bounding box */
	BoundingBox& aBoundingVolume = (BoundingBox &)aScene->getBoundingVolume();
	std::vector<vec3> &points = aBoundingVolume.getNonTransformedPoints();
	float flo[6];
	flo[0] = points[0].x;
	flo[1] = points[0].y;
	flo[2] = points[0].z;
	flo[3] = points[1].x;
	flo[4] = points[1].y;
	flo[5] = points[1].z;
	f.write(reinterpret_cast<char*> (flo), 6 * sizeof(float));

	OctreeByMat *o = aScene->m_pGeometry;
	_writeString(o->getName(),f);

	std::shared_ptr<OctreeByMatNode> &n = o->m_pOctreeRootNode;

	_writeOctreeByMatNode(n,f);
}


void 
CBOLoader::writeScene (nau::scene::IScene *aScene, std::string &aFilename) {

//	CLogger::getInstance().addLog(LEVEL_INFO, "debug.txt");

	std::string path = File::GetPath(aFilename);

	std::map<std::string, std::shared_ptr<IRenderable>> renderables;
	std::set<std::string> materials;

	//std::fstream f (aFilename.c_str(), std::fstream::out);
	std::fstream f (aFilename.c_str(), std::fstream::out | std::fstream::binary);


	if (!f.is_open()) {
		NAU_THROW ("Cannot open file: %s", aFilename.c_str());
		return;
	}

	unsigned int siz; 

	std::vector<std::shared_ptr<SceneObject>> sceneObjects;
	aScene->getAllObjects(&sceneObjects);
	std::vector<std::shared_ptr<SceneObject>>::iterator objIter;


	// MATERIALS - collect materials
	objIter = sceneObjects.begin();

	// For each object in the scene 
	for ( ; objIter != sceneObjects.end(); objIter++) {

		std::shared_ptr<IRenderable> &aRenderablePtr = (*objIter)->getRenderable();

		if (NULL == aRenderablePtr) {
			continue;
		}

		std::shared_ptr<IRenderable> &aRenderable = (*objIter)->getRenderable();
		
		// Material groups 
		std::vector<std::shared_ptr<MaterialGroup>>& materialGroups = aRenderable->getMaterialGroups();

		// collect material names in a set
		for (auto &aMaterialGroup: materialGroups) {

			std::string matName = aMaterialGroup->getMaterialName();
			materials.insert(matName);
		}
	}
	// write number of materials
	siz = (unsigned int)materials.size();
	size_t k = sizeof(siz);
	f.write (reinterpret_cast<char *> (&siz), k);

	// write materials
	std::set<std::string>::iterator matIter;

	matIter = materials.begin();
	for(; matIter != materials.end(); matIter++) {
		_writeMaterial(*matIter,path,f);
	}

	// writing geometry
	_writeString (aScene->getType(), f);
	LOG_INFO ("[Writing] scene type: %s", aScene->getType().c_str()); 


	if (aScene->getType() == "OctreeByMatScene") {
		_writeOctreeByMat((OctreeByMatScene *)aScene,f);
		f.close();
		return;
	}
	// Else write "normal" scenes


	/* Number of objects */

	siz = (unsigned int)sceneObjects.size();
	f.write (reinterpret_cast<char *> (&siz), sizeof(siz));

	objIter = sceneObjects.begin();

	/* For each object in the scene */
	for (; objIter != sceneObjects.end(); objIter++) {
		/* Write the object type */

		_writeString((*objIter)->getType(), f);

		LOG_INFO("[Writing] object type: [%s]", (*objIter)->getType().c_str());


		/* Write the object's name */
		_writeString((*objIter)->getName(), f); //Misses getId()

		LOG_INFO("[Writing] object name: [%s]", (*objIter)->getName().c_str());

		/* Write the specific data */
		(*objIter)->writeSpecificData(f);

		/* Write the bounding volume */
		IBoundingVolume *aBoundingVolume = (*objIter)->getBoundingVolume();

		/* Bounding volume type */

		_writeString(aBoundingVolume->getType(), f);

		/* Bounding volume points */
		std::vector<vec3> points = aBoundingVolume->getPoints();
		siz = (unsigned int)points.size();
		float *flo = new float[siz * 3];
		for (unsigned int k = 0; k < siz; ++k) {
			flo[k * 3] = points[k].x;
			flo[k * 3 + 1] = points[k].y;
			flo[k * 3 + 2] = points[k].z;
		}
		f.write (reinterpret_cast<char*> (&siz), sizeof(siz));
		f.write(reinterpret_cast<char*> (flo), siz * 3 * sizeof(float));
		delete flo;

		/* Write the transform */
		mat4 aTransform = (*objIter)->getTransform();
		
		/* The transform's matrix */
		f.write(reinterpret_cast<char*> (const_cast<float *>(aTransform.getMatrix())), sizeof(float)*16);

		std::shared_ptr<IRenderable> &aRenderablePtr = (*objIter)->getRenderable();

		if (0 == aRenderablePtr) {
			_writeString ("NULLOBJECT", f);
			continue;
		}

		/* Write the object's renderable */
		std::shared_ptr<IRenderable> &aRenderable = (*objIter)->getRenderable();


		/* The renderable name, for later lookup */
		std::string name = aRenderable->getName();
		unsigned int pos = (unsigned int)name.rfind("#");
		std::string name2;
		if (pos != string::npos)
			pos = (unsigned int)name.rfind("#", pos-1);
			if (pos != string::npos)
				name2 = name.substr(pos+1);
		else
			name2 = name;
		_writeString (name2, f);

		LOG_INFO ("[Writing] Renderable's name: [%s]", aRenderable->getName().c_str()); 

		if (0 == renderables.count (aRenderable->getName())) {

			renderables[aRenderable->getName()] = aRenderable;

			_writeString (aRenderable->getType(), f);

			LOG_INFO ("[Writing] Renderable's type: [%s]", aRenderable->getType().c_str()); 

			/* Vertices data */
			std::shared_ptr<VertexData> &aVertexData = aRenderable->getVertexData();

			_writeVertexData (aVertexData, f);

			/* Material groups */
			std::vector<std::shared_ptr<MaterialGroup>>& materialGroups = aRenderable->getMaterialGroups();
			siz = (unsigned int)materialGroups.size();

			f.write (reinterpret_cast<char*> (&siz), sizeof (siz));

			for (auto &aMaterialGroup : materialGroups) {
				
				/*Write material's name */
				_writeString (aMaterialGroup->getMaterialName(), f);

				LOG_INFO ("[Writing] MaterialGroup's name: %s", aMaterialGroup->getMaterialName().c_str()); 
							
				/* Indices Data */
				std::shared_ptr<nau::geometry::IndexData> &mgIndexData = aMaterialGroup->getIndexData();

				_writeIndexData (mgIndexData, f);
			}
				
		}
	}
	f.close();
}


void 
CBOLoader::_writeMaterial(std::string matName, std::string path, std::fstream &f) {

	std::shared_ptr<Material> &aMaterial = MATERIALLIBMANAGER->getMaterialFromDefaultLib (matName);

	_writeString (matName, f);

	LOG_INFO ("[Writing] Material's name: %s", aMaterial->getName().c_str()); 

	// write color
	vec4 v = aMaterial->getColor().getPropf4(ColorMaterial::AMBIENT);
	f.write (reinterpret_cast<char*>(&v.x), sizeof (float) * 4);
	v = aMaterial->getColor().getPropf4(ColorMaterial::SPECULAR);
	f.write (reinterpret_cast<char*>(&v.x), sizeof (float) * 4);
	v = aMaterial->getColor().getPropf4(ColorMaterial::DIFFUSE);
	f.write (reinterpret_cast<char*>(&v.x), sizeof (float) * 4);
	v = aMaterial->getColor().getPropf4(ColorMaterial::EMISSION);
	f.write (reinterpret_cast<char*>(&v.x), sizeof (float) * 4);

	float value = aMaterial->getColor().getPropf(ColorMaterial::SHININESS);
	f.write (reinterpret_cast<char*> (&value), sizeof (float));

	// write textures
	for (int i = 0; i < 8; i++) { /***MARK***/ //8!? Is it a magic number!?
		if (NULL != aMaterial->getTexture(i)) {
			std::string label = aMaterial->getTexture(i)->getLabel();
			std::string k = File::GetRelativePathTo(path, label);
			_writeString (k, f);
		} else {
			_writeString ("<no texture>", f);
		}
	}

}


void
CBOLoader::_readMaterial(std::string path, std::fstream &f) {


	// read materials name
	char buffer[1024];
	_readString(buffer, f);

	std::shared_ptr<Material> &aMaterial = MATERIALLIBMANAGER->createMaterial(buffer);

	float values[4];
	float value;

	f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
	aMaterial->getColor().setPropf4(ColorMaterial::AMBIENT, values[0], values[1], values[2], values[3]);

	f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
	aMaterial->getColor().setPropf4(ColorMaterial::SPECULAR, values[0], values[1], values[2], values[3]);

	f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
	aMaterial->getColor().setPropf4(ColorMaterial::DIFFUSE, values[0], values[1], values[2], values[3]);

	f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
	aMaterial->getColor().setPropf4(ColorMaterial::EMISSION, values[0], values[1], values[2], values[3]);

	f.read (reinterpret_cast<char*> (&value), sizeof (float));
	aMaterial->getColor().setPropf(ColorMaterial::SHININESS, value);


	// Textures
	for (int i = 0; i < 8; i++) { /***MARK***/ //8!? Is it a magic number!?
		_readString (buffer, f);
		if (0 != std::string(buffer).compare("<no texture>")) {
			aMaterial->createTexture (i, File::GetFullPath(path,buffer));
		}
	}
}
