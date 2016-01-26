#include "nau/scene/octreeByMatScene.h"
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
	m_SceneObjects(),
	m_pGeometry (0),
	m_BoundingBox()
{
	EVENTMANAGER->addListener("SET_POSITION", this);
	EVENTMANAGER->addListener("SET_ROTATION", this);

	m_Type = "OctreeByMatScene";

}


OctreeByMatScene::~OctreeByMatScene(void) {

}


void
OctreeByMatScene::eventReceived(const std::string &sender, 
	const std::string &eventType, const std::shared_ptr<IEventData> &evt) {
	vec4 *p = (vec4 *)evt->getData();

	if (eventType == "SET_POSITION") {
		mat4 t;
		t.setIdentity();
		t.translate(p->x, p->y, p->z);
		this->setTransform(t);
	}
	if (eventType == "SET_ROTATION") {
		m_Transform.setIdentity();
		m_Transform.rotate(p->w, p->x, p->y, p->z);
		updateSceneObjectTransforms();	
	}
}


mat4 &
OctreeByMatScene::getTransform() {

	return m_Transform;
}


void
OctreeByMatScene::setTransform(nau::math::mat4 &t) {

	m_Transform = t;
	updateSceneObjectTransforms();
}


void
OctreeByMatScene::transform(nau::math::mat4 &t) {

	m_Transform*= t;
	updateSceneObjectTransforms();
}


void 
OctreeByMatScene::updateSceneObjectTransforms() {

    for(auto &so: m_SceneObjects)
    {
		so->updateGlobalTransform(m_Transform);
    }

	if (m_pGeometry)
		m_pGeometry->updateOctreeTransform(m_Transform);
}


void 
OctreeByMatScene::build (void) {

	if (true == m_Built) {
		return ;
	}

	m_Built = true;
	// First create a list with objects that are static
	std::vector<std::shared_ptr<SceneObject>> staticObjects;

	for (auto &so : m_SceneObjects) {

		if (so->isStatic()) {
			staticObjects.push_back(so);
		}
	}

	// Create Octree's root node
	m_pGeometry = new OctreeByMat;
	m_pGeometry->setName(this->m_Name);

	// Feed the octree with each renderable
	m_pGeometry->build (staticObjects);


	// Erase the scene object afterwards
	std::vector<std::shared_ptr<SceneObject>>::iterator objIter;
	for (objIter = m_SceneObjects.begin(); objIter != m_SceneObjects.end(); ) {
			
		if ((*objIter)->isStatic()) {
			std::string renderableName = (*objIter)->getRenderable()->getName();
			objIter = m_SceneObjects.erase (objIter);
			RESOURCEMANAGER->removeRenderable(renderableName);
		}
	}
}


IBoundingVolume& 
OctreeByMatScene::getBoundingVolume (void) {

	return m_BoundingBox;
}


void
OctreeByMatScene::compile (void) {

	if (true == m_Compiled) {
		return;
	}

	m_Compiled = true;

	if (0 != m_pGeometry) {
		m_pGeometry->_compile();
	} 

	for (auto &objIter: m_SceneObjects) {
		std::vector<std::shared_ptr<MaterialGroup>> &matGroups = objIter->getRenderable()->getMaterialGroups();

		for (auto& matGroupsIter: matGroups){
			matGroupsIter->compile();
		}
	}
}


void 
OctreeByMatScene::add (std::shared_ptr<SceneObject> &aSceneObject) {

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


 void
OctreeByMatScene::findVisibleSceneObjects (std::vector<std::shared_ptr<SceneObject>> *v, Frustum &aFrustum,
				Camera &aCamera, bool conservative) {

	if (0 != m_pGeometry) {
		m_pGeometry->_findVisibleSceneObjects (v, aFrustum, aCamera, conservative);
	}

	/* The objects NOT on the octree */
	for (auto &so:m_SceneObjects) {
		/***MARK***/ /* View Frustum Culling Test */
		
		int side = aFrustum.isVolumeInside (so->getBoundingVolume(), conservative);

		if (Frustum::OUTSIDE != side) {
			v->push_back (so);
		}
	}
}


 void
OctreeByMatScene::getAllObjects (std::vector<std::shared_ptr<SceneObject>> *v) {

	for (auto &so: m_SceneObjects) {
		v->push_back(so);
	}

	if (0 != m_pGeometry) {
		m_pGeometry->_getAllObjects (v);
	}
}


const std::set<std::string> &
OctreeByMatScene::getMaterialNames() {

	m_MaterialNames.clear();

	for ( auto objIter : m_SceneObjects) {
		objIter->getRenderable()->getMaterialNames(&m_MaterialNames);
	}

	if (0 != m_pGeometry) {
		m_pGeometry->getMaterialNames(&m_MaterialNames);
	}
	return m_MaterialNames;
}


std::shared_ptr<SceneObject> &
OctreeByMatScene::getSceneObject (std::string name) {

	for (auto &so: m_SceneObjects) {
		if (0 == so->getName().compare (name)) {
			return so;
		}
	}
	return m_EmptySOptr;
}


std::shared_ptr<SceneObject> &
OctreeByMatScene::getSceneObject( int index) {

	if (index < 0 || (unsigned int)index >= m_SceneObjects.size())
		return m_EmptySOptr;

	return m_SceneObjects.at(index);
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

