#include <nau/scene/octreescene.h>
#include <nau/render/rendermanager.h>
#include <nau/material/imaterialgroup.h>

#include <nau/debug/profile.h>

using namespace nau::scene;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;

OctreeScene::OctreeScene(void) :
	m_vReturnVector(),
	m_SceneObjects(),
	//m_vCameras(),
	//m_vLights(),
	m_pGeometry (0),
	//m_pObjects (0),
	m_Compiled (false),
	m_Built (false),
	m_Visible (true),
	m_BoundingBox()
{
}

OctreeScene::~OctreeScene(void)
{
	delete m_pGeometry;

	std::vector<ISceneObject*>::iterator iter; /***MARK***/

    for(iter = m_SceneObjects.begin(); iter != m_SceneObjects.end(); iter++)
    {
		m_SceneObjects.erase(iter);
    }



}




bool 
OctreeScene::build (void)
{
	if (true == m_Built) {
		LOG_CRITICAL ("The compile/build command has already been invoke");
		return false;
	}

	// First create a list with objects that are static
	std::vector<ISceneObject*>::iterator objIter;
	std::vector<ISceneObject*> staticObjects;

	objIter = m_SceneObjects.begin();

	for ( ; objIter != m_SceneObjects.end(); objIter++) {

		if ((*objIter)->isStatic())
			staticObjects.push_back(*objIter);
	}

	m_Built = true;
	// Create Octree's root node
	m_pGeometry = new Octree;

	// Feed the octree with each renderable
	m_pGeometry->build (staticObjects);


	// Erase the scene object afterwards

	objIter = m_SceneObjects.begin();

	for ( ; objIter != m_SceneObjects.end(); ) {
			
		if ((*objIter)->isStatic())
			delete (*objIter);
		objIter = m_SceneObjects.erase (objIter);
	}
	return true;
}

IBoundingVolume& 
OctreeScene::getBoundingVolume (void)
{
	return m_BoundingBox;
}

bool
OctreeScene::compile (void)
{
	if (false == m_Compiled) {
		m_Compiled = true;
	} else {
		LOG_CRITICAL ("The compile/build command has already been invoke");
		return false;
	}

	if (0 != m_pGeometry) {
		m_pGeometry->_compile();
	} 

	std::vector<ISceneObject*>::iterator objIter;

	objIter = m_SceneObjects.begin();

	for ( ; objIter != m_SceneObjects.end(); objIter++) {
		(*objIter)->getRenderable().getVertexData().compile();
		std::vector<IMaterialGroup*> &matGroups = (*objIter)->getRenderable().getMaterialGroups();

		std::vector<IMaterialGroup*>::iterator matGroupsIter = matGroups.begin();

		for ( ; matGroupsIter != matGroups.end(); matGroupsIter++){
			(*matGroupsIter)->getVertexData().compile();
		}

	}

	return true;
}

void 
OctreeScene::add (ISceneObject *aSceneObject)
{
	if (0 == aSceneObject->getType().compare ("OctreeNode")) {
		m_Built = true;
		if (0 == m_pGeometry) {
			m_pGeometry = new Octree;
		}
		m_pGeometry->_place (aSceneObject);
	} else {
		m_SceneObjects.push_back (aSceneObject);
	}

	m_BoundingBox.compound (aSceneObject->getBoundingVolume());
}

std::vector <ISceneObject*>& 
OctreeScene::findVisibleSceneObjects (Frustum &aFrustum, Camera &aCamera)
{
	m_vReturnVector.clear();

	if (0 != m_pGeometry) {
		m_pGeometry->_findVisibleSceneObjects (m_vReturnVector, aFrustum, aCamera);
	}

	/* The objects NOT on the octree */
	std::vector<ISceneObject*>::iterator objIter;
	
	objIter = m_SceneObjects.begin();
	
	for ( ; objIter != m_SceneObjects.end(); objIter++) {
		/***MARK***/ /* View Frustum Culling Test */
		
		int side = aFrustum.isVolumeInside ((*objIter)->getBoundingVolume(), aCamera);

		if (Frustum::OUTSIDE != side) {
			m_vReturnVector.push_back (*(objIter));
		}
	}

	return m_vReturnVector;
}

std::vector<ISceneObject*>& 
OctreeScene::getAllObjects ()
{
	m_vReturnVector.clear();

	std::vector<ISceneObject*>::iterator objIter;

	objIter = m_SceneObjects.begin();

	for ( ; objIter != m_SceneObjects.end(); objIter++) {
		m_vReturnVector.push_back(*(objIter));
	}

	if (0 != m_pGeometry) {
		m_pGeometry->_getAllObjects (m_vReturnVector);
	}
	return m_vReturnVector;
}

ISceneObject* 
OctreeScene::getSceneObject (std::string name)
{
	std::vector<ISceneObject*>::iterator objIter;

	objIter = m_SceneObjects.begin();

	for ( ; objIter != m_SceneObjects.end(); objIter++) {
		if (0 == (*objIter)->getName().compare (name)) {
			return (*objIter);
		}
	}
	return 0;
}

ISceneObject*
OctreeScene::getSceneObject( int index) 
{
	if (index < 0 || (unsigned int)index >= m_SceneObjects.size())
		return NULL;

	return m_SceneObjects.at(index);
}


void 
OctreeScene::show (void)
{
	m_Visible = true;
}
			
void 
OctreeScene::hide (void)
{
	m_Visible = false;
}
			
bool 
OctreeScene::isVisible (void)
{
	return m_Visible;
}

std::string 
OctreeScene::getType (void) {
	return "Octree";
}

void OctreeScene::unitize() {

	unsigned int i;
	float max,min;
	nau::math::vec3 vMax,vMin;
	
	vMax = m_BoundingBox.getMax();
	vMin = m_BoundingBox.getMin();

	if (vMax.x > vMax.y)
		if (vMax.x > vMax.z)
			max = vMax.x;
		else 
			max = vMax.z;
	else if (vMax.y > vMax.z)
		max = vMax.y;
	else
		max = vMax.z;

	if (vMin.x > vMin.y)
		if (vMin.x > vMin.z)
			min = vMin.x;
		else
			min = vMin.z;
	else if (vMin.y > vMin.z)
		min = vMin.y;
	else
		min = vMin.z;

	if (m_SceneObjects.size()) {

		for ( i = 0; i < m_SceneObjects.size(); i++) 
			m_SceneObjects[i]->unitize(min,max);

		m_BoundingBox.intersect(m_SceneObjects[0]->getBoundingVolume());
		for (i = 1; i < m_SceneObjects.size(); i++)
			m_BoundingBox.compound(m_SceneObjects[i]->getBoundingVolume());
	}

	// if the scene is in a octree
	if (0 != m_pGeometry) {
		m_pGeometry->unitize(min,max);

	}

}

