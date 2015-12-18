#include "nau/scene/octreeByMat.h"

#include "nau.h"
#include "nau/clogger.h" 
#include "nau/scene/octreeByMatNode.h"


#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <ctime>
#include <list>

using namespace nau::scene;
using namespace nau::geometry;

//int nodesRendered;

OctreeByMat::OctreeByMat() :
	m_pOctreeRootNode (0),
	m_vReturnVector()
{
	
   //ctor
}


OctreeByMat::~OctreeByMat()
{
	//delete m_pOctreeRootNode;
   //dtor
	/***MARK***/ //The OctreeNode must be deleted
}


void
OctreeByMat::setName(std::string name) {

	m_Name = name;
}


std::string
OctreeByMat::getName() {

	return m_Name;
}


void 
OctreeByMat::build (std::vector<SceneObject*> &sceneObjects)
{
	//Get all objects currently in the object's array and build a static octree of them
	// For each SceneObject, burn the transform on vertices
	srand((unsigned)time(0));
	_transformSceneObjects(sceneObjects);

	// Calculate the maximum AABB for the present geometry
	BoundingBox sceneBoundingBox = _calculateBoundingBox(sceneObjects);

	// Create the Octree's root node
	m_pOctreeRootNode = new OctreeByMatNode ();
	m_pOctreeRootNode->setName(m_Name);

	Mesh *m = (Mesh *)RESOURCEMANAGER->createRenderable("Mesh");//new Mesh();

	// Send the Renderables down the octree
	std::vector<SceneObject*>::iterator objIter;
	objIter = sceneObjects.begin();

	std::map<Mesh*,int> meshMap;
	for ( ; objIter != sceneObjects.end(); objIter++) {
		if (meshMap.count((Mesh *)(*objIter)->_getRenderablePtr()))
			int x = 2;
		else
			meshMap[(Mesh *)(*objIter)->_getRenderablePtr()] = 1;
	}

	std::map<Mesh*,int>::iterator iter = meshMap.begin();
	for ( ; iter != meshMap.end(); ++iter) {
		m->merge((iter->first));
		
	//	m_pOctreeRootNode->addRenderable (&(*objIter)->getRenderable()); /***MARK***/
	}
	//objIter = sceneObjects.begin();
	//for ( ; objIter != sceneObjects.end(); objIter++) {
	//	m->merge(&(*objIter)->getRenderable());
	//	
	////	m_pOctreeRootNode->addRenderable (&(*objIter)->getRenderable()); /***MARK***/
	//}
	m_pOctreeRootNode->setRenderable(m);

	RESOURCEMANAGER->removeRenderable(m->getName());
	
}


void
OctreeByMat::getMaterialNames(std::set<std::string> *nameList) 
{
	if (m_pOctreeRootNode)

		m_pOctreeRootNode->getMaterialNames(nameList);
}


void
OctreeByMat::_transformSceneObjects (std::vector<SceneObject*> &sceneObjects)
{
	std::vector<SceneObject*>::iterator objIter;

	objIter = sceneObjects.begin();

	for ( ; objIter != sceneObjects.end(); objIter++) {
		(*objIter)->burnTransform();	
	}
}


BoundingBox
OctreeByMat::_calculateBoundingBox (std::vector<SceneObject*> &sceneObjects)
{
	BoundingBox sceneBoundingBox;

	std::vector<SceneObject*>::iterator objIter;

	objIter = sceneObjects.begin();

	for ( ; objIter != sceneObjects.end();
		objIter++) {
		const IBoundingVolume *aBoundingBox = (*objIter)->getBoundingVolume();
		sceneBoundingBox.compound (aBoundingBox);
	}

	return sceneBoundingBox;
}

void
OctreeByMat::_compile (void)
{
	if (0 != m_pOctreeRootNode) {
		m_pOctreeRootNode->_compile();
	}
}

void
OctreeByMat::_findVisibleSceneObjects (std::vector<SceneObject*> &m_vReturnVector,
											 nau::geometry::Frustum &aFrustum, Camera &aCamera,
											 bool conservative)
{
	if (0 != m_pOctreeRootNode) {
		m_pOctreeRootNode->_findVisibleSceneObjects (m_vReturnVector, aFrustum, aCamera, conservative);
	}
	//return m_vReturnVector;
}

void 
OctreeByMat::_getAllObjects (std::vector<nau::scene::SceneObject*> &returnVector)
{

	m_pOctreeRootNode->getAllSceneObjects(returnVector);
}


void 
OctreeByMat::_place (nau::scene::SceneObject *aSceneObject)
{
	static std::list<nau::scene::OctreeByMatNode*> tmpVector;
	static int octant = 0;

	OctreeByMatNode* aNode = reinterpret_cast<OctreeByMatNode*> (aSceneObject);

	if (0 == m_pOctreeRootNode) {
		m_pOctreeRootNode = aNode;
		tmpVector.push_back (m_pOctreeRootNode);
		return;
	}

	if (aNode->m_NodeId < octant) {
		tmpVector.pop_front();
	}

	OctreeByMatNode* currentNode = *(tmpVector.begin());
	octant = aNode->m_NodeId;

	currentNode->_setChild(octant, aNode);
	if (aNode->_getChildCount() > 0) {
		tmpVector.push_back (aNode);
	}
	
}


void OctreeByMat::unitize(vec3 &center, vec3 &min, vec3 &max) {

	m_pOctreeRootNode->unitize(center, min, max);
}


void 
OctreeByMat::updateOctreeTransform(nau::math::mat4 &t)
{
	m_pOctreeRootNode->updateNodeTransform(t);
}


