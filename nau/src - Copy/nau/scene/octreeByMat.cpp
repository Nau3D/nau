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
	m_pOctreeRootNode (0)
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
OctreeByMat::build (std::vector<std::shared_ptr<SceneObject>> &sceneObjects)
{
	//Get all objects currently in the object's array and build a static octree of them
	// For each SceneObject, burn the transform on vertices
	srand((unsigned)time(0));
	_transformSceneObjects(sceneObjects);

	// Calculate the maximum AABB for the present geometry
	BoundingBox sceneBoundingBox = _calculateBoundingBox(sceneObjects);

	// Create the Octree's root node
	m_pOctreeRootNode = std::shared_ptr<OctreeByMatNode>(new OctreeByMatNode ());
	m_pOctreeRootNode->setName(m_Name);

	std::shared_ptr<nau::render::IRenderable> &m = RESOURCEMANAGER->createRenderable("Mesh","");

	// Send the Renderables down the octree

	std::map<std::shared_ptr<nau::render::IRenderable>,int> meshMap;
	for (auto &so : sceneObjects) {
			meshMap[so->getRenderable()] = 1;
	}

	
	for(auto &r: meshMap) {
		std::shared_ptr<nau::render::IRenderable> rend = r.first;
		m->merge(rend);
		
	}
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
OctreeByMat::_transformSceneObjects (std::vector<std::shared_ptr<SceneObject>> &sceneObjects)
{
	for (auto &so : sceneObjects) {
		so->burnTransform();
	}
}


BoundingBox
OctreeByMat::_calculateBoundingBox (std::vector<std::shared_ptr<SceneObject>> &sceneObjects)
{
	BoundingBox sceneBoundingBox;

	for (auto &so : sceneObjects) {
		const IBoundingVolume *aBoundingBox = so->getBoundingVolume();
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
OctreeByMat::_findVisibleSceneObjects (std::vector<std::shared_ptr<SceneObject>> *returnVector,
											 nau::geometry::Frustum &aFrustum, Camera &aCamera,
											 bool conservative)
{
	if (0 != m_pOctreeRootNode) {
		m_pOctreeRootNode->_findVisibleSceneObjects (returnVector, aFrustum, aCamera, conservative);
	}
}

void 
OctreeByMat::_getAllObjects (std::vector<std::shared_ptr<SceneObject>> *returnVector)
{

	m_pOctreeRootNode->getAllSceneObjects(returnVector);
}


void 
OctreeByMat::_place (std::shared_ptr<SceneObject> &aSceneObject)
{
	static std::list<std::shared_ptr<OctreeByMatNode>> tmpVector;
	static int octant = 0;

	std::shared_ptr<OctreeByMatNode> &aNode = dynamic_pointer_cast<OctreeByMatNode> (aSceneObject);

	if (0 == m_pOctreeRootNode) {
		m_pOctreeRootNode = aNode;
		tmpVector.push_back (m_pOctreeRootNode);
		return;
	}

	if (aNode->m_NodeId < octant) {
		tmpVector.pop_front();
	}

	std::shared_ptr<OctreeByMatNode> &currentNode = *(tmpVector.begin());
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


