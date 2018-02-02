#include "nau/scene/octreeUnified.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/material/materialGroup.h"
#include "nau/render/renderManager.h"
#include "nau/render/opengl/glProfile.h"


using namespace nau::scene;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;


OctreeUnified::OctreeUnified(void) : IScenePartitioned(),
	m_SceneObject(NULL),
	m_BoundingBox() {

	EVENTMANAGER->addListener("SET_POSITION", this);
	EVENTMANAGER->addListener("SET_ROTATION", this);
	m_Type = "OctreeUnified";
}


OctreeUnified::~OctreeUnified(void) {

}


void
OctreeUnified::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<IEventData> &evt) {

	vec4 *p = (vec4 *)evt->getData();
	//	SLOG("Scene %s %s %s %f %f %f", m_Name.c_str(), sender.c_str(), eventType.c_str(), p->x, p->y, p->z);

	if (eventType == "SET_POSITION") {

		mat4 t;
		t.translate(p->x, p->y, p->z);
		this->setTransform(t);
	}
	if (eventType == "SET_ROTATION") {

		m_Mat4Props[TRANSFORM].setIdentity();
		m_Mat4Props[TRANSFORM].rotate(p->w, p->x, p->y, p->z);
		updateSceneObjectTransforms();
	}
}


mat4 &
OctreeUnified::getTransform() {
	return m_Mat4Props[TRANSFORM];
}


void
OctreeUnified::setTransform(nau::math::mat4 &t) {

	m_Mat4Props[TRANSFORM] = t;
	updateSceneObjectTransforms();
}


void
OctreeUnified::transform(nau::math::mat4 &t) {

	m_Mat4Props[TRANSFORM] *= t;
	updateSceneObjectTransforms();
}


void
OctreeUnified::updateSceneObjectTransforms() {

	if (m_SceneObject)
		m_SceneObject->updateGlobalTransform(m_Mat4Props[TRANSFORM]);
}


void
OctreeUnified::build(void) {

}


IBoundingVolume&
OctreeUnified::getBoundingVolume(void) {

	return m_BoundingBox;
}


void
OctreeUnified::compile(void) {

	if (true == m_Compiled) {
		return;
	}

	m_Compiled = true;
	
	if (m_SceneObject) {
		std::vector<std::shared_ptr<MaterialGroup>> &matGroups = m_SceneObject->getRenderable()->getMaterialGroups();

		for (auto mg : matGroups) {
			mg->compile();
		}
	}
}


void
OctreeUnified::add(std::shared_ptr<SceneObject> &aSceneObject) {

	aSceneObject->updateGlobalTransform(m_Mat4Props[TRANSFORM]);
	aSceneObject->burnTransform();

	if (!m_SceneObject) {
		m_SceneObject = SceneObjectFactory::Create("SimpleObject");
		m_SceneObject->setRenderable(aSceneObject->getRenderable());
	}
	else {
		//aSceneObject->burnTransform();
		m_SceneObject->getRenderable()->merge(aSceneObject->getRenderable());
	}
	m_BoundingBox.compound(aSceneObject->getBoundingVolume());
}


void
OctreeUnified::findVisibleSceneObjects(std::vector<std::shared_ptr<SceneObject>> *v, Frustum &aFrustum, Camera &aCamera, bool conservative) {

	/* The objects NOT on the octree */
	int side = Frustum::OUTSIDE;
	if (m_SceneObject)
		side = aFrustum.isVolumeInside(m_SceneObject->getBoundingVolume(), conservative);

	if (Frustum::OUTSIDE != side) {
		v->push_back(m_SceneObject);
	}
}


void
OctreeUnified::getAllObjects(std::vector<std::shared_ptr<SceneObject>> *v) {

	if (m_SceneObject)
		v->push_back(m_SceneObject);
}


const std::set<std::string> &
OctreeUnified::getMaterialNames() {

	m_MaterialNames.clear();
	if (m_SceneObject)
		m_SceneObject->getRenderable()->getMaterialNames(&m_MaterialNames);

	return m_MaterialNames;
}


std::shared_ptr<SceneObject> &
OctreeUnified::getSceneObject(std::string name) {

	if (m_SceneObject && m_SceneObject->getName() == name)
		return m_SceneObject;
	else
		return m_EmptySOptr;
}


std::shared_ptr<SceneObject> &
OctreeUnified::getSceneObject(int index) {

	if (index != 0)
		return m_EmptySOptr;
	else
		return m_SceneObject;
}


void OctreeUnified::unitize() {

//	float max, min;
	nau::math::vec3 max, min, center;

	max = m_BoundingBox.getMax();
	min = m_BoundingBox.getMin();
	center = m_BoundingBox.getCenter();

	if (m_SceneObject) {

		m_SceneObject->unitize(center, min, max);
	}
	m_BoundingBox = *(BoundingBox *)(m_SceneObject->getBoundingVolume());
}

