#include <algorithm>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <ctime>

#include "nau/scene/octree.h"
#include "nau/scene/octreeNode.h"
#include "nau/clogger.h" /***MARK***/
#include "nau.h"

#include <list>

using namespace nau::scene;
using namespace nau::geometry;

//int nodesRendered;

Octree::Octree() :
	m_pOctreeRootNode (0),
	m_vReturnVector()
{
	
   //ctor
}

Octree::~Octree()
{
	delete m_pOctreeRootNode;
   //dtor
	/***MARK***/ //The OctreeNode must be deleted
}

void 
Octree::build (std::vector<SceneObject*> &sceneObjects)
{
	//Get all objects currently in the object's array and build a static octree of them
	// For each SceneObject, burn the transform on vertices
	srand((unsigned)time(0));
	_transformSceneObjects(sceneObjects);

	// Calculate the maximum AABB for the present geometry
	BoundingBox sceneBoundingBox = _calculateBoundingBox(sceneObjects);

	// Create the Octree's root node
	m_pOctreeRootNode = new OctreeNode (0, new BoundingBox(sceneBoundingBox));

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
	m_pOctreeRootNode->resetCounter();
	m_pOctreeRootNode->setRenderable(m);
	
}


void
Octree::getMaterialNames(std::set<std::string> *nameList) 
{
	if (m_pOctreeRootNode)

		m_pOctreeRootNode->getMaterialNames(nameList);
}


void
Octree::_transformSceneObjects (std::vector<SceneObject*> &sceneObjects)
{
	std::vector<SceneObject*>::iterator objIter;

	objIter = sceneObjects.begin();

	for ( ; objIter != sceneObjects.end(); objIter++) {
		(*objIter)->burnTransform();	
	}
}


BoundingBox
Octree::_calculateBoundingBox (std::vector<SceneObject*> &sceneObjects)
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
Octree::_compile (void)
{
	if (0 != m_pOctreeRootNode) {
		m_pOctreeRootNode->_compile();
	}
}

void
Octree::_findVisibleSceneObjects (std::vector<SceneObject*> &m_vReturnVector,
											 nau::geometry::Frustum &aFrustum, 
											 Camera &aCamera,
											 bool conservative)
{
	if (0 != m_pOctreeRootNode) {
		m_pOctreeRootNode->_findVisibleSceneObjects (m_vReturnVector, aFrustum, aCamera,conservative);
	}
	//return m_vReturnVector;
}

void 
Octree::_getAllObjects (std::vector<nau::scene::SceneObject*> &m_vReturnVector)
{
	std::vector<nau::scene::OctreeNode*> tmpVector (10000);
	unsigned int count = 0;
	
	if (0 != m_pOctreeRootNode) {
		tmpVector.at(count) = (m_pOctreeRootNode);
		count++;
	} 

	for (unsigned int i = 0; i < count; i++) {
		for (int j = 0; j < 8; j++) {
			if (0 != tmpVector.at(i)->_getChild (j)) {
				tmpVector.at(count) = (tmpVector.at(i)->_getChild (j));
				count++;
			}
		}
	}

	//std::vector<nau::scene::OctreeNode*>::iterator objIter;

	//objIter = tmpVector.begin();

	for (unsigned int i = 0; i < count; i++) {
		m_vReturnVector.push_back (tmpVector.at (i));
	}
}

void 
Octree::_place (nau::scene::SceneObject *aSceneObject)
{
	static std::list<nau::scene::OctreeNode*> tmpVector;
	static int octant = 0;

	OctreeNode* aNode = reinterpret_cast<OctreeNode*> (aSceneObject);

	if (0 == m_pOctreeRootNode) {
		m_pOctreeRootNode = aNode;
		tmpVector.push_back (m_pOctreeRootNode);
		return;
	}

	if (aNode->m_NodeId < octant) {
		tmpVector.pop_front();
	}

	OctreeNode* currentNode = *(tmpVector.begin());
	octant = aNode->m_NodeId;

	currentNode->_setChild(octant, aNode);
	if (aNode->_getChildCount() > 0) {
		tmpVector.push_back (aNode);
	}
	
}


void Octree::unitize(vec3 &center, vec3 &min, vec3 &max) {

	m_pOctreeRootNode->unitize(center, min, max);
}


void 
Octree::updateOctreeTransform(nau::math::mat4 &t)
{
	m_pOctreeRootNode->updateGlobalTransform(t);
}


/*

void 
COctree::setWorld (IWorld &aWorld)
{
	
	std::vector<IRenderable*> &renderables = aWorld.getStaticRenderables (); 
	std::vector<IRenderable*>::iterator iter;

	for (iter = renderables.begin (); iter != renderables.end (); ++iter){
		IRenderable *pRenderable = (*iter);
		
		std::vector<IMaterialGroup*>& materialGroups = pRenderable->getMaterialGroups ();
		std::vector<IMaterialGroup*>::iterator materialGroupsIter;

		//Copying the vertices from this iRenderable to the internal vertice pool

		int offset = m_VerticesPool.size ();

		LOG_INFO ("The offset %d", offset);

		std::vector<vec3> &vertices = pRenderable->getVerticesList ();
		std::vector<vec3>::iterator verticesIter;

		LOG_INFO("Number of vertices: %d", vertices.size ());

		//While copying, erase the original vertices in order to save some memory
		for (verticesIter = vertices.begin ();
			verticesIter != vertices.end (); ){
				//LOG_INFO("V: [%.4f, %.4f, %.4f]", (*verticesIter).x, (*verticesIter).y, (*verticesIter).z);
				m_VerticesPool.push_back ((*verticesIter));
				verticesIter = vertices.erase (verticesIter);
		}
		
		//Copying the normals from this iRenderable to the internal vertice pool

		std::vector<vec3> &normals = pRenderable->getNormalsList ();
		std::vector<vec3>::iterator normalsIter;

		//While copying, erase the original normals in order to save some memory
		for (normalsIter = normals.begin ();
			normalsIter != normals.end (); ){
				m_NormalsPool.push_back ((*normalsIter));
				normalsIter = normals.erase (normalsIter);
		}		

		//Copying the texture coords from this iRenderable to the internal vertice pool

		std::vector<vec3> &textureCoords = pRenderable->getTexCoordsList ();
		std::vector<vec3>::iterator textureCoordsIter;

		//While copying, erase the original texture coords in order to save some memory
		for (textureCoordsIter = textureCoords.begin ();
			textureCoordsIter != textureCoords.end (); ){
				m_TextureCoordsPool.push_back ((*textureCoordsIter));
				textureCoordsIter = textureCoords.erase (textureCoordsIter);
		}		

		//Prepare the material list for sending down the octree:
		// 1. Add the offset to indicesList

		for (materialGroupsIter = materialGroups.begin ();
			materialGroupsIter != materialGroups.end ();
			materialGroupsIter++){
				IMaterialGroup *group = (*materialGroupsIter);
				
				std::vector<unsigned int> &indices = group->getIndexList ();
				std::vector<unsigned int>::iterator indicesIter;
				for (indicesIter = indices.begin ();
					indicesIter != indices.end ();
					indicesIter++){
						(*indicesIter) += offset;
				}
		}
	}

	//Now calculate the initial bounding box for the root octree node
	
	vec3 bbMin;
	vec3 bbMax;

	std::vector<vec3>::iterator vec3Iter;
	for (vec3Iter = m_VerticesPool.begin ();
			vec3Iter != m_VerticesPool.end ();
			vec3Iter++){
		vec3 aVec = *vec3Iter;
	
		//Check if it is minimum
		if (aVec.x < bbMin.x){
			bbMin.x = aVec.x;
		}
		if (aVec.y < bbMin.y){
			bbMin.y = aVec.y;
		}
		if (aVec.z < bbMin.z){
			bbMin.z = aVec.z;
		}

		//Check if it is maximum
		if (aVec.x > bbMax.x){
			bbMax.x = aVec.x;
		}
		if (aVec.y > bbMax.y){
			bbMax.y = aVec.y;
		}
		if (aVec.z > bbMax.z){
			bbMax.z = aVec.z;
		}
	}
	
	//Make the bounding box a cube
	float max = fabs(bbMin.x);
	if (max < fabs(bbMin.y)){
		max = fabs(bbMin.y);
	}
	if (max < fabs(bbMin.z)){
		max = fabs(bbMin.z);
	}

	if (max < fabs(bbMax.x)){
		max = fabs(bbMax.x);
	}
	if (max < fabs(bbMax.y)){
		max = fabs(bbMax.y);
	}
	if (max < fabs(bbMax.z)){
		max = fabs(bbMax.z);
	}

	bbMin.x = -max;
	bbMin.y = -max;
	bbMin.z = -max;

	bbMax.x = max;
	bbMax.y = max;
	bbMax.z = max;

	//Create octree root node
	if (0 == m_OctreeRootNode){
		m_OctreeRootNode = new COctreeNode (this, bbMin, bbMax);
	}

	//Feed the octree
	for (iter = renderables.begin (); iter != renderables.end (); ++iter){
		IRenderable *r = (*iter);

		std::vector<IMaterialGroup*>& materialGroups = r->getMaterialGroups ();
		std::vector<IMaterialGroup*>::iterator materialGroupsIter;

		for (materialGroupsIter = materialGroups.begin ();
			materialGroupsIter != materialGroups.end (); ){
			IMaterialGroup *group = (*materialGroupsIter);
		
			m_OctreeRootNode->addMaterialGroup (*group);

			materialGroupsIter = materialGroups.erase (materialGroupsIter);
			delete (group);
		}

	}

	m_OctreeRootNode->buildLists();

	m_VerticesPool.clear();
	m_NormalsPool.clear();
	m_TextureCoordsPool.clear();
}

vec3& 
Octree::getVertice (unsigned int i)
{
	return m_VerticesPool.at(i);
}

std::vector<vec3>& 
Octree::getVerticesList ()
{
	return m_VerticesPool;
}

std::vector<vec3>& 
Octree::getNormalsList ()
{
	return m_NormalsPool;
}

std::vector<vec3>& 
Octree::getTexCoordsList ()
{
	return m_TextureCoordsPool;
}

*/
