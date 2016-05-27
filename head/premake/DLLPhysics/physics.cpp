#include "physics.h"

#include <memory>

static char className[] = "Dummy";

__declspec(dllexport)
void *
createPhysics() {

	Physics *p = new Physics();
	return p;
}


__declspec(dllexport)
void
init() {

}


__declspec(dllexport)
char *
getClassName() {

	return className;
}




Physics *
Physics::Create() {

	return new Physics();
}


Physics::Physics() {

	m_GlobalProps["GRAVITY"] = Prop(IPhysics::VEC4, 0.0f, 9.8f, 0.0f, 0.0f);
	m_GlobalProps["K"] = Prop(IPhysics::FLOAT, 0.1f);

	m_MaterialProps["ACCELERATION"] = Prop(IPhysics::VEC4, 1.0f, 0.0f, 0.0f, 0.0f);
	m_MaterialProps["MASS"] = Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["MASS1"] = Prop(IPhysics::FLOAT, 1.0f);

}


Physics::~Physics(void) {

}


std::map<std::string, nau::physics::IPhysics::Prop> &
Physics::getGlobalProperties() {

	return m_GlobalProps;
}


std::map<std::string, nau::physics::IPhysics::Prop> &
Physics::getMaterialProperties() {

	return m_MaterialProps;
}


static float translate1 = 0;
static float delta1 = 0.001f;
static float translate2 = 0;
static float delta2 = 0.000f;
static float moveVertex = 0;

void
Physics::update() {

	for (auto s : m_Scenes) {

		switch (s.second.sceneType) {
		case IPhysics::RIGID:
			translate1 += delta1;
			translate2 += delta2;
			s.second.transform[12] = translate1;
			s.second.transform[13] = translate2;

			break;

		case IPhysics::CLOTH:
			moveVertex += 0.001f;
			s.second.vertices[0] = moveVertex;
			break;
		}
	}
}


void
Physics::build() {

}


void
Physics::setSceneType(const std::string & scene, SceneType type) {

	m_Scenes[scene].sceneType = type;
}


void
Physics::applyFloatProperty(const std::string &scene, const std::string &property, float value) {

	delta2 = value / 100.0f;
}


void
Physics::applyVec4Property(const std::string &scene, const std::string &property, float *value) {

}


void
Physics::applyGlobalFloatProperty(const std::string &property, float value) {

}


void
Physics::applyGlobalVec4Property(const std::string &property, float *value) {

}


void
Physics::setScene(const std::string & scene, float * vertices, unsigned int *indices, float *transform) {

	m_Scenes[scene].vertices = vertices;
	m_Scenes[scene].indices = indices;
	m_Scenes[scene].transform = transform;
}


float *
Physics::getSceneTransform(const std::string & scene) {

	return m_Scenes[scene].transform;
}


void
Physics::setSceneTransform(const std::string & scene, float * transform) {

	m_Scenes[scene].transform = transform;
}


