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
PhysicsDummy::setSceneType(const std::string & scene, SceneType type) {

	m_Scenes[scene].sceneType = type;
}


void 
PhysicsDummy::applyFloatProperty(const std::string &scene, const std::string &property, float value) {

}


void 
PhysicsDummy::applyVec4Property(const std::string &scene, const std::string &property, float *value) {

}


void
PhysicsDummy::applyGlobalFloatProperty(const std::string &property, float value) {

}


void
PhysicsDummy::applyGlobalVec4Property(const std::string &property, float *value) {

}


void 
PhysicsDummy::setScene(const std::string & scene, float * vertices, unsigned int *indices, float *transform) {

	m_Scenes[scene].vertices = vertices;
	m_Scenes[scene].indices = indices;
	m_Scenes[scene].transform = transform;
}


float * 
PhysicsDummy::getSceneTransform(const std::string & scene) {

	return m_Scenes[scene].transform;
}


void 
PhysicsDummy::setSceneTransform(const std::string & scene, float * transform) {

	m_Scenes[scene].transform = transform;
}


std::map < std::string, IPhysics::Prop> &
PhysicsDummy::getGlobalProperties() {

	return m_GlobalProps;
}


std::map < std::string, IPhysics::Prop> &
PhysicsDummy::getMaterialProperties() {

	return m_MaterialProps;
}


