#include "nau/scene/scene.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
#include "nau/material/materialGroup.h"
#include "nau/render/renderManager.h"


using namespace nau::scene;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;


Scene::Scene(void) :
	m_vReturnVector(),
	m_SceneObjects(),
	m_BoundingBox()
{
	EVENTMANAGER->addListener("SET_POSITION", this);
	EVENTMANAGER->addListener("SET_ROTATION", this);

}


Scene::~Scene(void) {

	m_SceneObjects.clear();
	m_vReturnVector.clear();
}


void
Scene::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
	if (eventType == "SET_POSITION") {

		vec4 *p = (vec4 *)evt->getData();
		m_Transform.setIdentity();
		m_Transform.translate(p->x, p->y, p->z);
//		SLOG("Scene SET_POS %f %f %f", p->x, p->y, p->z);
	}
	if (eventType == "SET_ROTATION") {

		vec4 *p = (vec4 *)evt->getData();
		
		m_Transform.setIdentity();
		m_Transform.rotate(p->w, p->x, p->y, p->z);
	}
}


mat4 &
Scene::getTransform() 
{
	return m_Transform;
}


void
Scene::setTransform(nau::math::mat4 &t)
{
	m_Transform.copy(t);
	updateSceneObjectTransforms();
}


void
Scene::transform(nau::math::mat4 &t)
{
	m_Transform *= t;
	updateSceneObjectTransforms();
}


void 
Scene::updateSceneObjectTransforms()
{
	std::vector<SceneObject*>::iterator iter; 
	iter = m_SceneObjects.begin();
    for( ; iter != m_SceneObjects.end(); ++iter)
    {
		(*iter)->updateGlobalTransform(m_Transform);
    }

}


void 
Scene::build (void)
{
}

IBoundingVolume& 
Scene::getBoundingVolume (void)
{
	return m_BoundingBox;
}

void
Scene::compile (void)
{
	if (true == m_Compiled) {
		return;
	}
		
	m_Compiled = true;

	std::vector<SceneObject*>::iterator objIter;
	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ++objIter) {
		(*objIter)->getRenderable().getVertexData().compile();
		std::vector<MaterialGroup*> &matGroups = (*objIter)->getRenderable().getMaterialGroups();

		std::vector<MaterialGroup*>::iterator matGroupsIter = matGroups.begin();

		for ( ; matGroupsIter != matGroups.end(); matGroupsIter++){
			(*matGroupsIter)->compile();
		}
	}
}


void 
Scene::add (SceneObject *aSceneObject)
{
	m_SceneObjects.push_back (aSceneObject);
	aSceneObject->updateGlobalTransform(m_Transform);
	m_BoundingBox.compound (aSceneObject->getBoundingVolume());
}


std::vector <SceneObject*>& 
Scene::findVisibleSceneObjects (Frustum &aFrustum, Camera &aCamera, bool conservative)
{
	m_vReturnVector.clear();

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
Scene::getAllObjects ()
{
	m_vReturnVector.clear();

	std::vector<SceneObject*>::iterator objIter;
	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ++objIter) {
		m_vReturnVector.push_back(*(objIter));
	}

	return m_vReturnVector;
}


SceneObject* 
Scene::getSceneObject (std::string name)
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
Scene::getSceneObject( int index) 
{
	if (index < 0 || (unsigned int)index >= m_SceneObjects.size())
		return NULL;

	return m_SceneObjects.at(index);
}


void
Scene::getMaterialNames(std::set<std::string> *nameList) 
{
	std::vector<SceneObject*>::iterator objIter;
	objIter = m_SceneObjects.begin();
	for ( ; objIter != m_SceneObjects.end(); ++objIter) {

		(*objIter)->getRenderable().getMaterialNames(nameList);
	}
	
}




std::string 
Scene::getType (void) {
	return "Scene";
}


void Scene::unitize() {

	unsigned int i;
	nau::math::vec3 max,min, center;
	
	max = m_BoundingBox.getMax();
	min = m_BoundingBox.getMin();
	center = m_BoundingBox.getCenter();

	m_BoundingBox.set(vec3(-1), vec3(1));

	if (m_SceneObjects.size()) {

		for ( i = 0; i < m_SceneObjects.size(); ++i) 
			m_SceneObjects[i]->unitize(center, min, max);

		m_BoundingBox.intersect(m_SceneObjects[0]->getBoundingVolume());
		for (i = 1; i < m_SceneObjects.size(); ++i)
			m_BoundingBox.compound(m_SceneObjects[i]->getBoundingVolume());
	}
}

//void 
//Scene::translate(float x, float y, float z) 
//{
//	m_Transform->translate(x,y,z);
//	updateSceneObjectTransforms();
//}

//void 
//Scene::show (void)
//{
//	m_Visible = true;
//}
//		
//
//void 
//Scene::hide (void)
//{
//	m_Visible = false;
//}
//
//
//bool 
//Scene::isVisible (void)
//{
//	return m_Visible;
//}