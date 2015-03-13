#include "nau/scene/octreeunified.h"
#include "nau/render/rendermanager.h"
#include "nau/material/materialgroup.h"

#include "nau/debug/profile.h"
#include "nau.h"

#include "nau/slogger.h"

using namespace nau::scene;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;


OctreeUnified::OctreeUnified(void) : IScenePartitioned(),
	m_vReturnVector(),
	m_SceneObject(NULL),
	m_BoundingBox()
{
	EVENTMANAGER->addListener("SET_POSITION", this);
	EVENTMANAGER->addListener("SET_ROTATION", this);
}


OctreeUnified::~OctreeUnified(void)
{
	if (m_SceneObject)
		delete m_SceneObject;

	m_vReturnVector.clear();
}


void
OctreeUnified::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
	vec4 *p = (vec4 *)evt->getData();
	//	SLOG("Scene %s %s %s %f %f %f", m_Name.c_str(), sender.c_str(), eventType.c_str(), p->x, p->y, p->z);

	if (eventType == "SET_POSITION") {

		mat4 t;
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
OctreeUnified::getTransform()
{
	return m_Transform;
}



void
OctreeUnified::setTransform(nau::math::mat4 &t)
{
	m_Transform = t;
	updateSceneObjectTransforms();
}


void
OctreeUnified::transform(nau::math::mat4 &t)
{
	m_Transform *= t;
	updateSceneObjectTransforms();
}


void
OctreeUnified::updateSceneObjectTransforms()
{
	if (m_SceneObject)
		m_SceneObject->updateGlobalTransform(m_Transform);
}


void
OctreeUnified::build(void)
{
}


IBoundingVolume&
OctreeUnified::getBoundingVolume(void)
{
	return m_BoundingBox;
}


void
OctreeUnified::compile(void)
{
	if (true == m_Compiled) {
		return;
	}

	m_Compiled = true;
	
	if (m_SceneObject) {
		std::vector<MaterialGroup*> &matGroups = m_SceneObject->getRenderable().getMaterialGroups();

		for (auto mg : matGroups) {
			mg->compile();
		}
	}
}


void
OctreeUnified::add(SceneObject *aSceneObject)
{
	aSceneObject->updateGlobalTransform(m_Transform);
	aSceneObject->burnTransform();

	if (!m_SceneObject) {
		m_SceneObject = SceneObjectFactory::create("SimpleObject");
		m_SceneObject->setRenderable(&(aSceneObject->getRenderable()));
	}
	else {
		aSceneObject->burnTransform();
		m_SceneObject->getRenderable().merge(&(aSceneObject->getRenderable()));
	}
	//int sizeV = m_SceneObject->getRenderable().getVertexData().getNumberOfVertices();
	//m_SceneObject->getRenderable().getVertexData().add(aSceneObject->getRenderable().getVertexData());
	//	
	//std::vector<IMaterialGroup*> &matGroups = aSceneObject->getRenderable().getMaterialGroups();
	//std::vector<IMaterialGroup*>::iterator matGroupsIter = matGroups.begin();
	//for (; matGroupsIter != matGroups.end(); ++matGroupsIter){
	//	m_SceneObject->getRenderable().addMaterialGroup(*matGroupsIter);
	//}

	m_BoundingBox.compound(aSceneObject->getBoundingVolume());
}


std::vector <SceneObject*>&
OctreeUnified::findVisibleSceneObjects(Frustum &aFrustum, Camera &aCamera, bool conservative)
{
	m_vReturnVector.clear();


	/* The objects NOT on the octree */
	int side = Frustum::OUTSIDE;
	if (m_SceneObject)
		side = aFrustum.isVolumeInside(m_SceneObject->getBoundingVolume(), conservative);

	if (Frustum::OUTSIDE != side) {
		m_vReturnVector.push_back(m_SceneObject);
	}

	return m_vReturnVector;
}


std::vector<SceneObject*>&
OctreeUnified::getAllObjects()
{
	m_vReturnVector.clear();

	if (m_SceneObject)
		m_vReturnVector.push_back(m_SceneObject);

	return m_vReturnVector;
}


void
OctreeUnified::getMaterialNames(std::set<std::string> *nameList)
{
	if (m_SceneObject)
		m_SceneObject->getRenderable().getMaterialNames(nameList);
}


SceneObject*
OctreeUnified::getSceneObject(std::string name)
{
	if (m_SceneObject && m_SceneObject->getName() == name)
		return m_SceneObject;
	else
		return NULL;
}


SceneObject*
OctreeUnified::getSceneObject(int index)
{
	if (index != 0)
		return NULL;
	else
		return m_SceneObject;
}




std::string
OctreeUnified::getType(void) {
	return "OctreeUnified";
}


void OctreeUnified::unitize() {

	float max, min;
	nau::math::vec3 vMax, vMin;

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

	if (m_SceneObject) {

		m_SceneObject->unitize(min, max);
	}
}

