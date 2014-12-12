#include <nau/loader/cboloader.h>

#include <nau.h>

#include <nau/scene/sceneobject.h>
#include <nau/scene/sceneobjectfactory.h>
#include <nau/geometry/iboundingvolume.h>
#include <nau/geometry/boundingvolumefactory.h>
#include <nau/math/vec3.h>
#include <nau/math/bvec4.h>
#include <nau/math/mat4.h>
#include <nau/math/transformfactory.h>
#include <nau/render/irenderable.h>
#include <nau/material/materialgroup.h>
#include <nau/material/materialgroup.h>
#include <nau/slogger.h>
#include <nau/system/fileutil.h>

#include <assert.h>
#include <fstream>
#include <map>

//#ifdef NAU_OPENGL
//#include <nau/render/opengl/glstate.h>
//#elif NAU_DIRECTX
//#include <nau/render/dx/dxstate.h>
//#endif
#include <nau/render/istate.h>

using namespace nau::loader;
using namespace nau::scene;
using namespace nau::math;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;

std::string CBOLoader::m_FileName;

void
CBOLoader::_writeVertexData (VertexData& aVertexData, std::fstream &f) 
{
	size_t size;
	size_t sizeVec;
	unsigned int countFilledArrays = 0;

	for (int i = 0; i < VertexData::MaxAttribs; i++) {
	
		std::vector<VertexData::Attr> &aVec = aVertexData.getDataOf (i);
		if (aVec.size())
			countFilledArrays++;
	}

	// write number of filled arrays
	f.write (reinterpret_cast<char *> (&countFilledArrays), sizeof (countFilledArrays));


	for (int i = 0; i < VertexData::MaxAttribs; i++) {

		std::vector<VertexData::Attr> &aVec = aVertexData.getDataOf (i);
		sizeVec = aVec.size();
		if (sizeVec > 0) {

			_writeString(VertexData::Syntax[i],f);
			// write size of array
			f.write (reinterpret_cast<char *> (&sizeVec), sizeof (sizeVec));
			// write attribute data
			f.write (reinterpret_cast<char *> (&(aVec[0])), 
					 static_cast<std::streamsize>(sizeVec) * sizeof(vec4));
		}
	}

	//std::vector<unsigned int> &aVec = aVertexData.getIndexData();
	//size = aVec.size();
	size = 0;
	f.write (reinterpret_cast<char *> (&size), sizeof (size));

	//if (aVec.size() > 0) {
	//	f.write (reinterpret_cast<char *> (&(aVec[0])), 
	//			 static_cast<std::streamsize> (size) * sizeof(unsigned int));
	//}
}


void
CBOLoader::_writeIndexData (IndexData& aVertexData, std::fstream &f) 
{
	size_t size;

	unsigned int countFilledArrays = 0;

	// write number of filled arrays
	f.write (reinterpret_cast<char *> (&countFilledArrays), sizeof (countFilledArrays));


	std::vector<unsigned int> &aVec = aVertexData.getIndexData();
	size = aVec.size();
	f.write (reinterpret_cast<char *> (&size), sizeof (size));

	if (aVec.size() > 0) {
		f.write (reinterpret_cast<char *> (&(aVec[0])), 
				 static_cast<std::streamsize> (size) * sizeof(unsigned int));
	}
}

void
CBOLoader::_readVertexData (VertexData& aVertexData, std::fstream &f)
{
	size_t size;
	unsigned int countFilledArrays;
	char buffer[1024];

	f.read(reinterpret_cast<char *> (&countFilledArrays), sizeof (countFilledArrays));

	for (unsigned int i = 0; i < countFilledArrays; i++) {

		// read attribs name
		_readString(buffer, f);

		// read size of array
		f.read (reinterpret_cast<char *> (&size), sizeof (size));

		std::vector<vec4> *aVector = new std::vector<vec4>(size);

		// read attrib data
		f.read (reinterpret_cast<char *> (&(*aVector)[0]), size * sizeof (vec4));
		//if (i == 0) {
		//	for (int j = 0; j < aVector->size(); ++j) {
		//	
		//		aVector->at(j).x /= 100.0;
		//		aVector->at(j).y /= 100.0;
		//		aVector->at(j).z /= 100.0;
		//	}
		//}

		unsigned int index = VertexData::getAttribIndex(buffer);
		aVertexData.setDataFor (index, aVector);
	}
	
	f.read (reinterpret_cast<char *> (&size), sizeof (size));
	//if (size > 0) {
	//	std::vector<unsigned int> *aNewVector = new std::vector<unsigned int>(size);

	//	f.read (reinterpret_cast<char *> (&(*aNewVector)[0]), size * sizeof (unsigned int));

	//	aVertexData.setIndexData (aNewVector);
	//}

	//unsigned int size;

	//for (int i = VertexData::VERTEX_ARRAY; i <= VertexData::CUSTOM_ATTRIBUTE_ARRAY7; i++) {
	//	f.read (reinterpret_cast<char *> (&size), sizeof (size));

	//	if (size > 0) {
	//		std::vector<vec3> *aNewVector = new std::vector<vec3>(size);
	//		std::vector<vec4> *aVector = new std::vector<vec4>(size);

	//		f.read (reinterpret_cast<char *> (&(*aNewVector)[0]), size * sizeof (vec3));
	//		for (unsigned int j = 0 ; j < size; j++) 
	//			aVector->at(j).set(aNewVector->at(j).x, aNewVector->at(j).y, aNewVector->at(j).z);
	//		aVertexData.setDataFor ((VertexData::VertexDataType)i, aVector);
	//	}
	//}
	//
	//f.read (reinterpret_cast<char *> (&size), sizeof (size));
	//if (size > 0) {
	//	std::vector<unsigned int> *aNewVector = new std::vector<unsigned int>(size);

	//	f.read (reinterpret_cast<char *> (&(*aNewVector)[0]), size * sizeof (unsigned int));

	//	aVertexData.setIndexData (aNewVector);
	//}
}


void
CBOLoader::_readIndexData (IndexData& aVertexData, std::fstream &f)
{
	size_t size;
	unsigned int countFilledArrays;

	f.read(reinterpret_cast<char *> (&countFilledArrays), sizeof (countFilledArrays));

	f.read (reinterpret_cast<char *> (&size), sizeof (size));
	if (size > 0) {
		std::vector<unsigned int> *aNewVector = new std::vector<unsigned int>(size);

		f.read (reinterpret_cast<char *> (&(*aNewVector)[0]), size * sizeof (unsigned int));

		aVertexData.setIndexData (aNewVector);
	}
}

void
CBOLoader::_ignoreVertexData (std::fstream &f)
{
	unsigned int size;

	for (int i = 0; i <= VertexData::MaxAttribs; i++) {
		f.read (reinterpret_cast<char *> (&size), sizeof (size));

		if (size > 0) {
			f.ignore (size * sizeof (vec3));
		}
	}
	
	f.read (reinterpret_cast<char *> (&size), sizeof (size));
	if (size > 0) {
		f.ignore (size * sizeof (unsigned int));
	}

}

void
CBOLoader::_writeString (const std::string& aString, std::fstream &f)
{
	unsigned int size = (unsigned int)aString.size();

	f.write (reinterpret_cast<char *> (&size), sizeof (size));
	f.write (aString.c_str(), size + 1);
}
/*
void
_writeString (const std::string& aString, std::fstream &f)
{
	size_t size = aString.size();

	f.write (reinterpret_cast<char *> (&size), sizeof (size));
	f.write (aString.c_str(), 
		static_cast<std::streamsize>(size) + 1);
}
*/
void
CBOLoader::_readString ( char *buffer, std::fstream &f)
{
	unsigned int size;

	memset (buffer, 0, 1024);
	f.read (reinterpret_cast<char *> (&size), sizeof (size));
	f.read (buffer, size + 1);
}

void
CBOLoader::_ignoreString (std::fstream &f)
{
	unsigned int size;
	f.read (reinterpret_cast<char *> (&size), sizeof (size));
	f.ignore (size + 1);
}

void 
CBOLoader::loadScene (nau::scene::IScene *aScene, std::string &aFilename)
{
	m_FileName = aFilename;
	std::string path = FileUtil::GetPath(aFilename);

	std::fstream f (aFilename.c_str(), std::fstream::in | std::fstream::binary);

	std::map<std::string, IRenderable*> renderables; /***MARK***/ //PROTO Renderables Manager
	//std::map<std::pair<std::string, std::string>, int> materialTrack;

	if (!f.is_open()) {
		NAU_THROW ("Cannot open file: %s", aFilename.c_str());
		return;
	}

	char buffer[1024];
	unsigned int nObjects;
	unsigned int nMatGroups, nMats;

	// MATERIALS
	f.read (reinterpret_cast<char *> (&nMats), sizeof(nMats));
	for (unsigned int i = 0 ; i < nMats; i++) {
		
		_readMaterial(path,f);
	}

	//GEOMETRY
	_readString (buffer, f);

	if (!strcmp(buffer,"OctreeByMatScene")) {
	
		_readOctreeByMat((OctreeByMatScene *)aScene, f);
		f.close();
		return;
	}
	//SLOG ("[Reading] Scene type: [%s]", buffer);

	f.read (reinterpret_cast<char *> (&nObjects), sizeof(nObjects));
	//SLOG ("[Reading] Number of Objects: [%d]", nObjects);

	for (unsigned int i = 0; i < nObjects; i++) {

		_readString (buffer, f);
		//SLOG ("[Reading] Type of object: [%s]", buffer);
		SceneObject *aObject = SceneObjectFactory::create (buffer);

		_readString (buffer, f);
		aObject->setName (buffer);

		aObject->readSpecificData (f);

		_readString (buffer, f);
		//SLOG ("[Reading] Type of BoundingVolume: [%s]", buffer);
		IBoundingVolume *aBoundingVolume = BoundingVolumeFactory::create (buffer);
		//aObject->setBoundingVolume (aBoundingVolume);

		std::vector<vec3>& points = aBoundingVolume->getPoints();
		unsigned int nPoints;
		f.read (reinterpret_cast<char *> (&nPoints), sizeof (nPoints));
		//SLOG ("[Reading] Number of points [%d]", nPoints);
		f.read (reinterpret_cast<char *> (&(points[0])), nPoints * sizeof (vec3));

		_readString (buffer, f);
		//SLOG ("[Reading] Type of Transform: [%s]", buffer);
		ITransform *aTransform = TransformFactory::create (buffer);
		mat4 mat; 
		f.read (reinterpret_cast<char *> (const_cast<float*>(mat.getMatrix())), sizeof(float)*16);

		//for (int i = 0; i < 16; i++) {
		//	SLOG ("Matrix(%d): [%f]", i, mat.getMatrix()[i]);
		//}

		aTransform->setMat44 (mat);
		aObject->setTransform (aTransform);

		_readString (buffer, f);
		//SLOG ("[Reading] Renderable's name: [%s]", buffer);
		std::string renderableName(buffer);

		if (0 == renderableName.compare ("NULLOBJECT")) {
			continue;
		}

		IRenderable *aRenderable = 0;

		if (0 == renderables.count (renderableName)) {
			/*Create the new renderable */

			_readString (buffer, f);
			aRenderable = RESOURCEMANAGER->createRenderable(buffer,renderableName,aFilename);
			//aRenderable->setName (renderableName);
			//RESOURCEMANAGER->addRenderable(aRenderable,aFilename);
			//assert (0 == aRenderable);
			renderables[renderableName] = aRenderable;
			
			VertexData &vertexData = aRenderable->getVertexData();
			_readVertexData (vertexData, f);

			//SLOG ("[Reading] Renderable type: [%s]", buffer);

			f.read (reinterpret_cast<char *> (&nMatGroups), sizeof (nMatGroups));
			for (unsigned int i = 0; i < nMatGroups; i++) {

				_readString (buffer, f);
				//SLOG ("[Reading] Material Groups name: [%s]", buffer);

				MaterialGroup *aMatGroup = new MaterialGroup(aRenderable, buffer);
				//aMatGroup->setMaterialName (buffer);				
				//aMatGroup->setParent (aRenderable);


				IndexData &indexData = aMatGroup->getIndexData();
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
CBOLoader::_readOctreeByMatSceneObject(SceneObject *so, std::fstream &f) {

	char buffer[1024];

	_readString(buffer,f);
	so->setName(buffer);

	// read the bounding volume

	//SLOG ("[Reading] Type of BoundingVolume: [%s]", buffer);
	BoundingBox *aBoundingVolume = (BoundingBox *)BoundingVolumeFactory::create ("BoundingBox");

	std::vector<vec3> points(2); 

	//SLOG ("[Reading] Number of points [%d]", nPoints);
	f.read (reinterpret_cast<char *> (&(points[0])), 2 * sizeof (vec3));
	//for (int j = 0; j < points.size(); ++j) {
	//		points[j].x /= 100.0;
	//		points[j].y /= 100.0;
	//		points[j].z /= 100.0;
	//	}
	aBoundingVolume->set(points[0], points[1]);
	so->setBoundingVolume (aBoundingVolume);

	_readString (buffer, f);
	//SLOG ("[Reading] Type of Transform: [%s]", buffer);
	ITransform *aTransform = TransformFactory::create (buffer);
	mat4 mat; 
	f.read (reinterpret_cast<char *> (const_cast<float*>(mat.getMatrix())), sizeof(float)*16);
	aTransform->setMat44 (mat);
	so->setTransform (aTransform);

	IRenderable *aRenderable = 0;

	_readString(buffer,f);
	aRenderable = RESOURCEMANAGER->createRenderable("Mesh", buffer, m_FileName);
			
	VertexData &vertexData = aRenderable->getVertexData();
	_readVertexData (vertexData, f);

	//SLOG ("[Reading] Renderable type: [%s]", buffer);
	_readString (buffer, f);
		//SLOG ("[Reading] Material Groups name: [%s]", buffer);


	MaterialGroup *aMatGroup = new MaterialGroup(aRenderable, buffer);
	//aMatGroup->setParent (aRenderable);
	//aMatGroup->setMaterialName (buffer);				


	IndexData &indexData = aMatGroup->getIndexData();
	_readIndexData (indexData, f);
				
		//_readString (buffer, f);
		//LOG_INFO ("[Reading] Material name: [%s]", buffer);
	aRenderable->getMaterialGroups().push_back (aMatGroup);

	so->setRenderable(aRenderable);

}


void 
CBOLoader::_writeOctreeByMatSceneObject(SceneObject *so, std::fstream &f) {

	_writeString (so->getName(), f); 
	
	/* Write the bounding volume */
	const IBoundingVolume *aBoundingVolume = so->getBoundingVolume();
	BoundingBox *b = (BoundingBox *)aBoundingVolume;

	std::vector<vec3> &points = b->getNonTransformedPoints();

	f.write(reinterpret_cast<char*> (&(points[0])), 
		static_cast<std::streamsize> (2) * sizeof (vec3));

	/* Write the transform */
	const ITransform &aTransform = so->getTransform();
	_writeString (aTransform.getType(), f);
	f.write(reinterpret_cast<char*> (const_cast<float *>(aTransform.getMat44().getMatrix())), sizeof(float)*16);


	IRenderable *aRenderablePtr = so->_getRenderablePtr();
	_writeString(aRenderablePtr->getName(),f);
	/* Vertices data */
	VertexData &aVertexData = aRenderablePtr->getVertexData();

	_writeVertexData (aVertexData, f);

	/* Material groups */
	std::vector<nau::material::MaterialGroup*>& materialGroups = aRenderablePtr->getMaterialGroups();

	MaterialGroup *aMaterialGroup = materialGroups[0];
				
	_writeString (aMaterialGroup->getMaterialName(), f);

	/* Indices Data */
	IndexData &mgIndexData = aMaterialGroup->getIndexData();

	_writeIndexData (mgIndexData, f);
}


void
CBOLoader::_readOctreeByMatNode(OctreeByMatNode *n, std::fstream &f) {

	int size;
	char buffer[1024];

	std::vector<vec3> points(2); 
	f.read (reinterpret_cast<char *> (&(points[0])), 2 * sizeof (vec3));
	//for (int j = 0; j < points.size(); ++j) {
	//		points[j].x /= 100.0;
	//		points[j].y /= 100.0;
	//		points[j].z /= 100.0;
	//	}

	n->m_BoundingVolume.set(points[0], points[1]);
	f.read (reinterpret_cast<char *> (&(points[0])), 2 * sizeof (vec3));
	//for (int j = 0; j < points.size(); ++j) {
	//		points[j].x /= 100.0;
	//		points[j].y /= 100.0;
	//		points[j].z /= 100.0;
	//	}
	n->m_TightBoundingVolume.set(points[0], points[1]);


	f.read (reinterpret_cast<char *> (&size), sizeof(size));

	for (int i = 0; i < size; ++i) {
		_readString(buffer, f);
		SceneObject *so = SceneObjectFactory::create ("SimpleObject");
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
		OctreeByMatNode *o = new OctreeByMatNode();
		_readOctreeByMatNode(o,f);
		o->m_pParent = n;
		n->m_pChilds[o->m_NodeId] = o;
	}
}

void
CBOLoader::_writeOctreeByMatNode(OctreeByMatNode *n, std::fstream &f) {

	int size;

	BoundingBox& aBoundingVolume = (BoundingBox &)n->m_BoundingVolume;
	std::vector<vec3>& points = aBoundingVolume.getNonTransformedPoints();
	f.write(reinterpret_cast<char*> (&(points[0])), 
		static_cast<std::streamsize> (2 * sizeof (vec3)));
	aBoundingVolume = (BoundingBox &)n->m_TightBoundingVolume;
	points = aBoundingVolume.getNonTransformedPoints();
	f.write(reinterpret_cast<char*> (&(points[0])), 
		static_cast<std::streamsize> (2 * sizeof (vec3)));


	size = n->m_pLocalMeshes.size();
	f.write (reinterpret_cast<char *> (&size), sizeof(size));

	std::map<std::string, nau::scene::SceneObject *>::iterator iter;

	iter = n->m_pLocalMeshes.begin();
	for (; iter != n->m_pLocalMeshes.end(); ++iter) {
		_writeString(iter->first, f);
		_writeOctreeByMatSceneObject(iter->second, f);
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
	std::vector<vec3> points(2); 
	f.read (reinterpret_cast<char *> (&(points[0])), 2 * sizeof (vec3));
	//for (int j = 0; j < points.size(); ++j) {
	//		points[j].x /= 100.0;
	//		points[j].y /= 100.0;
	//		points[j].z /= 100.0;
	//	}

	aScene->m_BoundingBox.set(points[0], points[1]);

	OctreeByMat *o = new OctreeByMat();
	aScene->m_pGeometry = o;
	_readString(buffer,f);
	//aScene->setName(buffer);
	o->setName(buffer);

	OctreeByMatNode *n = new OctreeByMatNode();
	_readOctreeByMatNode(n,f);
	o->m_pOctreeRootNode = n;
}



void
CBOLoader::_writeOctreeByMat(OctreeByMatScene *aScene, std::fstream &f) {

	/* Write the bounding box */
	BoundingBox& aBoundingVolume = (BoundingBox &)aScene->getBoundingVolume();
	std::vector<vec3> &points = aBoundingVolume.getNonTransformedPoints();
	f.write(reinterpret_cast<char*> (&(points[0])), 
		static_cast<std::streamsize> (2 * sizeof (vec3)));

	OctreeByMat *o = aScene->m_pGeometry;
	_writeString(o->getName(),f);

	OctreeByMatNode *n = o->m_pOctreeRootNode;

	_writeOctreeByMatNode(n,f);
}


void 
CBOLoader::writeScene (nau::scene::IScene *aScene, std::string &aFilename)
{

	std::string path = FileUtil::GetPath(aFilename);

	std::map<std::string, IRenderable*> renderables;
	std::set<std::string> materials;

	//std::fstream f (aFilename.c_str(), std::fstream::out);
	std::fstream f (aFilename.c_str(), std::fstream::out | std::fstream::binary);


	if (!f.is_open()) {
		NAU_THROW ("Cannot open file: %s", aFilename.c_str());
		return;
	}

	//unsigned int size;
	size_t size; 


	std::vector<SceneObject*> &sceneObjects = aScene->getAllObjects();
	std::vector<SceneObject*>::iterator objIter;


	// MATERIALS - collect materials
	objIter = sceneObjects.begin();

	// For each object in the scene 
	for ( ; objIter != sceneObjects.end(); objIter++) {

		IRenderable *aRenderablePtr = (*objIter)->_getRenderablePtr();

		if (0 == aRenderablePtr) {
			continue;
		}

		IRenderable &aRenderable = (*objIter)->getRenderable();
		
		// Material groups 
		std::vector<nau::material::MaterialGroup*>& materialGroups = aRenderable.getMaterialGroups();
		std::vector<nau::material::MaterialGroup*>::iterator mgIter;

		// collect material names in a set
		mgIter = materialGroups.begin();

		for ( ; mgIter != materialGroups.end(); mgIter++) {
			MaterialGroup *aMaterialGroup = (*mgIter);
			
			std::string matName = aMaterialGroup->getMaterialName();

			materials.insert(matName);
		}

	}
	// write number of materials
	size = materials.size();
	f.write (reinterpret_cast<char *> (&size), sizeof(size));

	// write materials
	std::set<std::string>::iterator matIter;

	matIter = materials.begin();
	for(; matIter != materials.end(); matIter++) {
		_writeMaterial(*matIter,path,f);
	}

	// writing geometry
	_writeString (aScene->getType(), f);
	SLOG ("[Writing] scene type: %s", aScene->getType().c_str()); 


	if (aScene->getType() == "OctreeByMatScene") {
		_writeOctreeByMat((OctreeByMatScene *)aScene,f);
		f.close();
		return;
	}
	// Else write "normal" scenes


	/* Number of objects */

	size = sceneObjects.size();
	f.write (reinterpret_cast<char *> (&size), sizeof(size));

	objIter = sceneObjects.begin();

	/* For each object in the scene */
	for ( ; objIter != sceneObjects.end(); objIter++) {
		/* Write the object type */

		_writeString ((*objIter)->getType(), f);

		LOG_INFO ("[Writing] object type: [%s]", (*objIter)->getType().c_str()); 


		/* Write the object's name */
		_writeString ((*objIter)->getName(), f); //Misses getId()

		LOG_INFO ("[Writing] object name: [%s]", (*objIter)->getName().c_str()); 

		/* Write the specific data */		
		(*objIter)->writeSpecificData (f);
		
		/* Write the bounding volume */
		const IBoundingVolume *aBoundingVolume = (*objIter)->getBoundingVolume();

		/* Bounding volume type */

		_writeString (aBoundingVolume->getType(), f);

		/* Bounding volume points */
		std::vector<vec3> &points = aBoundingVolume->getPoints();

		size = points.size();
		f.write (reinterpret_cast<char*> (&size), sizeof(size));

		f.write(reinterpret_cast<char*> (&(points[0])), 
			static_cast<std::streamsize> (size) * sizeof (vec3));

		/* Write the transform */
		const ITransform &aTransform = (*objIter)->getTransform();
		
		/* Transform type */
		_writeString (aTransform.getType(), f);

		/* The transform's matrix */
		f.write(reinterpret_cast<char*> (const_cast<float *>(aTransform.getMat44().getMatrix())), sizeof(float)*16);

		IRenderable *aRenderablePtr = (*objIter)->_getRenderablePtr();

		if (0 == aRenderablePtr) {
			_writeString ("NULLOBJECT", f);
			continue;
		}

		/* Write the object's renderable */
		IRenderable &aRenderable = (*objIter)->getRenderable();


		/* The renderable name, for later lookup */
		std::string name = aRenderable.getName();
		int pos = name.rfind("#");
		std::string name2;
		if (pos != string::npos)
			pos = name.rfind("#", pos-1);
			if (pos != string::npos)
				name2 = name.substr(pos+1);
		else
			name2 = name;
		_writeString (name2, f);

		LOG_INFO ("[Writing] Renderable's name: [%s]", aRenderable.getName().c_str()); 

		if (0 == renderables.count (aRenderable.getName())) {

			renderables[aRenderable.getName()] = &aRenderable;

			_writeString (aRenderable.getType(), f);

			LOG_INFO ("[Writing] Renderable's type: [%s]", aRenderable.getType().c_str()); 

			/* Vertices data */
			VertexData &aVertexData = aRenderable.getVertexData();

			_writeVertexData (aVertexData, f);

			/* Material groups */
			std::vector<nau::material::MaterialGroup*>& materialGroups = aRenderable.getMaterialGroups();
			std::vector<nau::material::MaterialGroup*>::iterator mgIter;

			size = materialGroups.size();

			f.write (reinterpret_cast<char*> (&size), sizeof (size));

			mgIter = materialGroups.begin();

			for ( ; mgIter != materialGroups.end(); mgIter++) {
				MaterialGroup *aMaterialGroup = (*mgIter);
				
				/*Write material's name */

				_writeString (aMaterialGroup->getMaterialName(), f);

				LOG_INFO ("[Writing] MaterialGroup's name: %s", aMaterialGroup->getMaterialName().c_str()); 
							
				/* Indices Data */
				IndexData &mgIndexData = aMaterialGroup->getIndexData();

				_writeIndexData (mgIndexData, f);
			}
				
		}
	}
	f.close();
}




void 
CBOLoader::_writeMaterial(std::string matName, std::string path, std::fstream &f) 
{
	Material *aMaterial = MATERIALLIBMANAGER->getDefaultMaterial (matName); 

	_writeString (matName, f);

	LOG_INFO ("[Writing] Material's name: %s", aMaterial->getName().c_str()); 

	// write color
	vec4 v = aMaterial->getColor().getPropf4(ColorMaterial::AMBIENT);
	f.write ((char *)&v.x, sizeof (float) * 4);
	v = aMaterial->getColor().getPropf4(ColorMaterial::SPECULAR);
	f.write ((char *)&v.x, sizeof (float) * 4);
	v = aMaterial->getColor().getPropf4(ColorMaterial::DIFFUSE);
	f.write ((char *)&v.x, sizeof (float) * 4);
	v = aMaterial->getColor().getPropf4(ColorMaterial::EMISSION);
	f.write ((char *)&v.x, sizeof (float) * 4);

	float value = aMaterial->getColor().getPropf(ColorMaterial::SHININESS);
	f.write (reinterpret_cast<char*> (&value), sizeof (float));

	// write textures
	for (int i = 0; i < 8; i++) { /***MARK***/ //8!? Is it a magic number!?
		if (0 != aMaterial->getTextures() && 0 != aMaterial->getTextures()->getTexture(i)) {
			std::string label = aMaterial->getTextures()->getTexture(i)->getLabel();
			_writeString (FileUtil::GetRelativePathTo(path,label), f);
		} else {
			_writeString ("<no texture>", f);
		}
	}

	// write shader

	// shader filenames
	IProgram *aProgram = aMaterial->getProgram();
	_writeString("",f);
	_writeString("",f);
	_writeString("",f);
	_writeString("",f);
	//_writeString(aProgram->getName(),f);
	//_writeString(aProgram->getShaderFile(IProgram::VERTEX_SHADER),f);
	//_writeString(aProgram->getShaderFile(IProgram::GEOMETRY_SHADER),f);
	//_writeString(aProgram->getShaderFile(IProgram::FRAGMENT_SHADER),f);

	// shader program values
	size_t numValues = 0; // HACK
	f.write (reinterpret_cast<char*> (&numValues), sizeof(numValues));


	// write state
//#ifdef NAU_OPENGL
//	GlState *s = (GlState *)aMaterial->getState();
//#elif NAU_DIRECTX
//	DXState *s = (DXState *)aMaterial->getState();
//#endif
	IState *s = aMaterial->getState();

	_writeString(s->getName(),f);


	IState::VarType vt;
	int ivalue;

	float fvalue, fvalues[4];
	bool bvalue, bvalues[4];
	bvec4 bvec;

	vt = IState::BOOL;
	int numProps = IState::COUNT_BOOLPROPERTY;
	f.write (reinterpret_cast<char *> (&numProps), sizeof(int));
	for (int i = 0; i < numProps; i++) {
		bvalue = s->getPropb((IState::BoolProperty)i);
		f.write (reinterpret_cast<char *> (&bvalue), sizeof(bool));
	}

	vt = IState::ENUM;
	numProps = IState::COUNT_ENUMPROPERTY;
	f.write (reinterpret_cast<char *> (&numProps), sizeof(int));
	for (int i = 0; i < numProps; i++) {
		ivalue = s->getPrope((IState::EnumProperty)i);
		f.write (reinterpret_cast<char *> (&ivalue), sizeof(int));
	}

	vt = IState::INT;
	numProps = IState::COUNT_INTPROPERTY;
	f.write (reinterpret_cast<char *> (&numProps), sizeof(int));
	for (int i = 0; i < numProps; i++) {
		ivalue = s->getPropi((IState::IntProperty)i);
		f.write (reinterpret_cast<char *> (&ivalue), sizeof(int));
	}

	vt = IState::FLOAT;
	numProps = IState::COUNT_FLOATPROPERTY;
	f.write (reinterpret_cast<char *> (&numProps), sizeof(int));
	for (int i = 0; i < numProps; i++) {
		fvalue = s->getPropf((IState::FloatProperty)i);
		f.write (reinterpret_cast<char *> (&fvalue), sizeof(float));
	}

	vt = IState::FLOAT4;
	numProps = IState::COUNT_FLOAT4PROPERTY;
	f.write (reinterpret_cast<char *> (&numProps), sizeof(int));
	for (int i = 0; i < numProps; i++) {
		v = s->getProp4f((IState::Float4Property)i);
		fvalues[0] = v.x; fvalues[1] = v.y; fvalues[2] = v.z; fvalues[3] = v.w;
		f.write (reinterpret_cast<char *> (fvalues), sizeof(float)*4);
	}

	vt = IState::BOOL4;
	numProps = IState::COUNT_BOOL4PROPERTY;
	f.write (reinterpret_cast<char *> (&numProps), sizeof(int));
	for (int i = 0; i < numProps; i++) {
		bvec = s->getProp4b((IState::Bool4Property)i);
		bvalues[0] = bvec.x; bvalues[1] = bvec.y; bvalues[2] = bvec.z; bvalues[3] = bvec.w;
		f.write (reinterpret_cast<char *> (bvalues), sizeof(bool)*4);
	}
}


void
CBOLoader::_readMaterial(std::string path, std::fstream &f)
{
	Material* aMaterial;// = new Material;

	// read materials name
	char buffer[1024];
	_readString(buffer, f);
	//aMaterial->setName (buffer);
	aMaterial = MATERIALLIBMANAGER->createMaterial(buffer);

	float values[4];
	float value;

	f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
	aMaterial->getColor().setProp(ColorMaterial::AMBIENT, values);

	f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
	aMaterial->getColor().setProp(ColorMaterial::SPECULAR, values);

	f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
	aMaterial->getColor().setProp(ColorMaterial::DIFFUSE, values);

	f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
	aMaterial->getColor().setProp(ColorMaterial::EMISSION, values);

	f.read (reinterpret_cast<char*> (&value), sizeof (float));
	aMaterial->getColor().setProp(ColorMaterial::SHININESS, value);


	// Textures
	for (int i = 0; i < 8; i++) { /***MARK***/ //8!? Is it a magic number!?
		_readString (buffer, f);
		if (0 != std::string(buffer).compare("<no texture>")) {
			aMaterial->createTexture (i, FileUtil::GetFullPath(path,buffer));
		}
	}

	// Shaders
	// shader filenames
	_readString(buffer, f);
	IProgram *aProgram = RESOURCEMANAGER->getProgram(buffer);
	_readString(buffer,f);
	aProgram->setShaderFile(IProgram::VERTEX_SHADER,buffer);
	_readString(buffer,f);
#if NAU_OPENGL_VERSION >= 320
	aProgram->setShaderFile(IProgram::GEOMETRY_SHADER,buffer);
#endif
	_readString(buffer,f);
	aProgram->setShaderFile(IProgram::FRAGMENT_SHADER,buffer);
	aProgram->reload();

	// shader program values
	size_t numValues; // HACK
	f.read (reinterpret_cast<char*> (&numValues), sizeof(numValues));
	for (unsigned int i = 0; i < numValues; i++) {
		// read Program Values
	}


	// State
//#ifdef NAU_OPENGL
//	GlState *s = (GlState *)aMaterial->getState();
//#elif NAU_DIRECTX
//	DXState *s = (DXState *)aMaterial->getState();
//#endif

	IState *s = aMaterial->getState();
	_readString(buffer,f);
	s->setName(buffer);

	int nProps,ivalue;
	float fvalue, fvalues[4];
	bool bvalue, bvalues[4];
	IState::VarType type;

	type = IState::BOOL;
	f.read (reinterpret_cast<char *> (&nProps), sizeof (nProps));
	for (int i = 0 ;  i < nProps; i++) {
		f.read(reinterpret_cast<char *> (&bvalue), sizeof (bool));
		s->setProp((IState::BoolProperty)i,bvalue);
	}

	type = IState::ENUM;
	f.read (reinterpret_cast<char *> (&nProps), sizeof (nProps));
	for (int i = 0 ;  i < nProps; i++) {
		f.read(reinterpret_cast<char *> (&ivalue), sizeof (int));
		s->setProp((IState::EnumProperty)i,ivalue);
	}

	type = IState::INT;
	f.read (reinterpret_cast<char *> (&nProps), sizeof (nProps));
	for (int i = 0 ;  i < nProps; i++) {
		f.read(reinterpret_cast<char *> (&ivalue), sizeof (int));
		s->setProp((IState::IntProperty)i,ivalue);
	}

	type = IState::FLOAT;
	f.read (reinterpret_cast<char *> (&nProps), sizeof (nProps));
	for (int i = 0 ;  i < nProps; i++) {
		f.read(reinterpret_cast<char *> (&fvalue), sizeof (float));
		s->setProp((IState::FloatProperty)i,fvalue);
	}

	type = IState::FLOAT4;
	f.read (reinterpret_cast<char *> (&nProps), sizeof (nProps));
	for (int i = 0 ;  i < nProps; i++) {
		f.read(reinterpret_cast<char *> (&fvalues), sizeof (float)*4);
		s->setProp((IState::Float4Property)i,fvalues[0],fvalues[1],fvalues[2],fvalues[3]);
	}

	type = IState::BOOL4;
	f.read (reinterpret_cast<char *> (&nProps), sizeof (nProps));
	for (int i = 0 ;  i < nProps; i++) {
		f.read(reinterpret_cast<char *> (&bvalues), sizeof (bool)*4);
		s->setProp((IState::Bool4Property)i,bvalues[0],bvalues[1],bvalues[2],bvalues[3]);
	}

	//MATERIALLIBMANAGER->addMaterial (DEFAULTMATERIALLIBNAME, aMaterial);
}
