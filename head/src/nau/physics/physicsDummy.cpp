#include "nau/physics/physicsDummy.h"

using namespace nau::physics;


static float translate = 0;
static float moveVertex = 0;

void
PhysicsDummy::update() {

	for (auto s : m_Scenes) {

		switch (s.second.sceneType) {
		case IPhysics::RIGID:
			translate += 0.001f;
			s.second.transform[12] = translate;
			break;

		case IPhysics::CLOTH:
			moveVertex += 0.001f;
			s.second.vertices[0] = moveVertex;
			break;
		}
	}
}

void 
PhysicsDummy::build() {

}


void 
PhysicsDummy::setSceneType(std::string & scene, SceneType type) {

	m_Scenes[scene].sceneType = type;
}


void nau::physics::PhysicsDummy::applyProperty(std::string & property, nau::math::Data * value) {

}


void nau::physics::PhysicsDummy::setSceneVertices(std::string & scene, float * vertices) {

	m_Scenes[scene].vertices = vertices;
}


void nau::physics::PhysicsDummy::setSceneIndices(std::string & scene, unsigned int * indices) {

	m_Scenes[scene].indices = indices;
}


float * nau::physics::PhysicsDummy::getSceneTransform(std::string & scene) {

	return m_Scenes[scene].transform;
}


void nau::physics::PhysicsDummy::setSceneTransform(std::string & scene, float * transform) {

	m_Scenes[scene].transform = transform;
}
