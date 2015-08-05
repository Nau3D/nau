#include "nau/scene/octreeByMatscene.h"
#include "nau/render/renderManager.h"
#include "nau/material/materialGroup.h"

#include "nau/debug/profile.h"
#include "nau.h"

#include "nau/slogger.h"

using namespace nau::scene;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;


OctreeByMatScene::OctreeByMatScene(void) : IScenePartitioned(),
	m_vReturnVector(),
	m_SceneObjects(),
	m_pGeometry (0),
	m_BoundingBox()
{
//	m_Transform = TransformFactory::create("SimpleTransform");
	EVENTMANAGER->addListener("SET_POSITION", this);
	EVENTMANAGER->addListener("SET_ROTATION", this);
}


OctreeByMatScene::~OctreeByMatScene(void)
{
	delete m_pGeometry;

	m_SceneObjects.clear();
	m_vReturnVector.clear();
}


void
OctreeByMatScene::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{	
	vec4 *p = (vec4 *)evt->getData();
//	SLOG("Scene %s %s %s %f %f %f", m_Name.c_str(), sender.c_str(), eventType.c_str(), p->x, p->y, p->z);

	if (eventType == "SET_POSITION") {

		mat4 t;
		t.setIdentity();
		t.translate(p->x, p->y, p->z);
		this->setTransform(t);
	}
	if (eventType == "SET_ROTATION") {

		//nau::math::mat4 m = m_Transform->getMat44();
		m_Transform.setIdentity();
		m_Transform.rotate(p->w, p->x, p->y, p->z);
		updateSceneObjectTransforms();	
	}
}


mat4 &
OctreeByMatScene::getTransform() 
{
	return m_Transform;
}


void
OctreeByMatScene::setTransform(nau::math::mat4 &t)
{
	m_Transform = t;
	updateSceneObjectTransforms();
}


void
OctreeByMatScene::transform(nau::math::mat4 &t)
{
	m_Transform*= t;
	updateSceneObjectTransforms();
}


void 
OctreeByMatScene::updateSceneObjectTransforms()
{
	std::vector<SceneObject*>::iterator iter; 
	iter = m_SceneObjects.begin();
    for( ; iter != m_SceneObjects.end(); ++iter)
    {
		(*iter)->updateGlobalTransform(m_Transform);
    }

	if (m_pGeometry)
		m_pGeometry->updateOctreeTransform(m_Transform);
}


void 
OctreeByMatScene::build (void)
{
	if (true == m_Built) {
		return ;
	}

	m_Built = true;
	// First create a list with objects that are static
	std::vector<SceneObject*>::iterator objIter;
	std::vector<SceneObject*> staticObjects;

	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ++objIter) {

		if ((*objIter)->isStatic()) {
			staticObjects.push_back(*objIter);
		}
	}

	// Create Octree's root node
	m_pGeometry = new OctreeByMat;
	m_pGeometry->setName(this->m_Name);

	// Feed the octree with each renderable
	m_pGeometry->build (staticObjects);


	// Erase the scene object afterwards

	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ) {
			
		if ((*objIter)->isStatic()) {
			std::string renderableName = (*objIter)->getRenderable().getName();
			objIter = m_SceneObjects.erase (objIter);
			RESOURCEMANAGER->removeRenderable(renderableName);
		}
	}
}


IBoundingVolume& 
OctreeByMatScene::getBoundingVolume (void)
{
	return m_BoundingBox;
}


void
OctreeByMatScene::compile (void)
{
	if (true == m_Compiled) {
		return;
	}

	m_Compiled = true;

	if (0 != m_pGeometry) {
		m_pGeometry->_compile();
	} 

	std::vector<SceneObject*>::iterator objIter;
	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ++objIter) {
		std::vector<MaterialGroup*> &matGroups = (*objIter)->getRenderable().getMaterialGroups();

		std::vector<MaterialGroup*>::iterator matGroupsIter = matGroups.begin();
		for ( ; matGroupsIter != matGroups.end(); ++matGroupsIter){
			(*matGroupsIter)->compile();
		}

	}
}


void 
OctreeByMatScene::add (SceneObject *aSceneObject)
{
	if (0 == aSceneObject->getType().compare ("OctreeNode")) {
		m_Built = true;
		if (0 == m_pGeometry) {
			m_pGeometry = new OctreeByMat;
		}
		m_pGeometry->_place (aSceneObject);
	} else {
		m_SceneObjects.push_back (aSceneObject);
	}
	aSceneObject->updateGlobalTransform(m_Transform);
	m_BoundingBox.compound (aSceneObject->getBoundingVolume());
}


std::vector <SceneObject*>& 
OctreeByMatScene::findVisibleSceneObjects (Frustum &aFrustum, Camera &aCamera, bool conservative)
{
	m_vReturnVector.clear();

	if (0 != m_pGeometry) {
		m_pGeometry->_findVisibleSceneObjects (m_vReturnVector, aFrustum, aCamera, conservative);
	}

	/* The objects NOT on the octree */
	std::vector<SceneObject*>::iterator objIter;
	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ++objIter) {
		/***MARK***/ /* View Frustum Culling Test */
		
		int side = aFrustum.isVolumeInside ((*objIter)->getBoundingVolume(), conservative);

		if (Frustum::OUTSIDE != side) {
			m_vReturnVector.push_back (*(objIter));
		}
	}

	return m_vReturnVector;
}


std::vector<SceneObject*>& 
OctreeByMatScene::getAllObjects ()
{
	m_vReturnVector.clear();

	std::vector<SceneObject*>::iterator objIter;
	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ++objIter) {
		m_vReturnVector.push_back(*(objIter));
	}

	if (0 != m_pGeometry) {
		m_pGeometry->_getAllObjects (m_vReturnVector);
	}
	return m_vReturnVector;
}


void
OctreeByMatScene::getMaterialNames(std::set<std::string> *nameList)
{
	std::vector<SceneObject*>::iterator objIter;
	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ++objIter) {
		(*objIter)->getRenderable().getMaterialNames(nameList);
	}

	if (0 != m_pGeometry) {
		m_pGeometry->getMaterialNames(nameList);
	}
}


SceneObject* 
OctreeByMatScene::getSceneObject (std::string name)
{
	std::vector<SceneObject*>::iterator objIter;
	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ++objIter) {
		if (0 == (*objIter)->getName().compare (name)) {
			return (*objIter);
		}
	}
	return 0;
}


SceneObject*
OctreeByMatScene::getSceneObject( int index) 
{
	if (index < 0 || (unsigned int)index >= m_SceneObjects.size())
		return NULL;

	return m_SceneObjects.at(index);
}





std::string 
OctreeByMatScene::getType (void) {
	return "OctreeByMatScene";
}


void OctreeByMatScene::unitize() {

	unsigned int i;
	nau::math::vec3 vMax, vMin, vCenter;;
	
	vMax = m_BoundingBox.getMax();
	vMin = m_BoundingBox.getMin();
	vCenter = m_BoundingBox.getCenter();

	m_BoundingBox.set(vec3(-1), vec3(1));


	if (m_SceneObjects.size()) {

		for ( i = 0; i < m_SceneObjects.size(); i++) 
			m_SceneObjects[i]->unitize(vCenter, vMin, vMax);

		m_BoundingBox.intersect(m_SceneObjects[0]->getBoundingVolume());
		for (i = 1; i < m_SceneObjects.size(); i++)
			m_BoundingBox.compound(m_SceneObjects[i]->getBoundingVolume());
	}

	// if the scene is in a octree
	if (0 != m_pGeometry) {
		m_pGeometry->unitize(vCenter, vMin, vMax);

	}

}

//void 
//OctreeByMatScene::show (void)
//{
//	m_Visible = true;
//}
//	
//
//void 
//OctreeByMatScene::hide (void)
//{
//	m_Visible = false;
//}
//	
//
//bool 
//OctreeByMatScene::isVisible (void)
//{
//	return m_Visible;
//}

//void 
//OctreeByMatScene::translate(float x, float y, float z) 
//{
//	m_Transform->translate(x,y,z);
//	updateSceneObjectTransforms();
//}