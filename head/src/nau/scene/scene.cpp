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
	m_SceneObjects(),
	m_BoundingBox()
{
	EVENTMANAGER->addListener("SET_POSITION", this);
	EVENTMANAGER->addListener("SET_ROTATION", this);
	m_Type = "Scene";
}


Scene::~Scene(void) {

	//while (!m_SceneObjects.empty()) {
	//	delete(*m_SceneObjects.begin());
	//	m_SceneObjects.erase(m_SceneObjects.begin());
	//}
}


void
Scene::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<IEventData> &evt)
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
Scene::getTransform() {

	return m_Transform;
}


void
Scene::setTransform(nau::math::mat4 &t) {

	m_Transform.copy(t);
	updateSceneObjectTransforms();
}


void
Scene::transform(nau::math::mat4 &t) {

	m_Transform *= t;
	updateSceneObjectTransforms();
}


void 
Scene::updateSceneObjectTransforms() {

    for(auto &so: m_SceneObjects) {
		so->updateGlobalTransform(m_Transform);
    }
}


void 
Scene::build (void) {

}


IBoundingVolume& 
Scene::getBoundingVolume (void) {

	return m_BoundingBox;
}


void
Scene::compile (void) {

	if (true == m_Compiled) {
		return;
	}
		
	m_Compiled = true;

	for (auto &so : m_SceneObjects) {
		so->getRenderable()->getVertexData()->compile();
		std::vector<std::shared_ptr<MaterialGroup>> &matGroups = so->getRenderable()->getMaterialGroups();

		for (auto &matGroup: matGroups) {
			matGroup->compile();
		}
	}
}


void 
Scene::add (std::shared_ptr<SceneObject> &aSceneObject) {

	m_SceneObjects.push_back (aSceneObject);
	aSceneObject->updateGlobalTransform(m_Transform);
	m_BoundingBox.compound (aSceneObject->getBoundingVolume());
}


void 
Scene::findVisibleSceneObjects (std::vector<std::shared_ptr<SceneObject>> *v, Frustum &aFrustum, Camera &aCamera, bool conservative) {
	
	for (auto &so:m_SceneObjects) {

		int side = aFrustum.isVolumeInside (so->getBoundingVolume(), conservative);

		if (Frustum::OUTSIDE != side) {
			v->push_back (so);
		}
	}
}


void 
Scene::getAllObjects (std::vector<std::shared_ptr<SceneObject>> *v) {

	for (auto &so : m_SceneObjects) 
		v->push_back(so);
}


std::shared_ptr<SceneObject> &
Scene::getSceneObject (std::string name)
{
	for (auto &so: m_SceneObjects) {
		if (0 == so->getName().compare (name)) {
			return so;
		}
	}
	return m_EmptySOptr;
}


std::shared_ptr<SceneObject> &
Scene::getSceneObject( int index) 
{
	if (index < 0 || (unsigned int)index >= m_SceneObjects.size())
		return m_EmptySOptr;

	return m_SceneObjects.at(index);
}


const std::set<std::string> &
Scene::getMaterialNames() 
{
	m_MaterialNames.clear();
	
	for ( auto so: m_SceneObjects) {

		so->getRenderable()->getMaterialNames(&m_MaterialNames);
	}
	return m_MaterialNames;
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