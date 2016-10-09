#include "nauBulletInterface.h"

#include <memory>

static char className[] = "Bullet plugin for Nau3D";
static NauBulletInterface * Instance = NULL;

__declspec(dllexport)
void *
createPhysics() {
	NauBulletInterface *p = new NauBulletInterface();
	return p;
}

__declspec(dllexport)
void
deletePhysics() {
	delete Instance;
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

NauBulletInterface * NauBulletInterface::Create() {
	return new NauBulletInterface();
}

NauBulletInterface::NauBulletInterface() {
	//INFO: Declare Physics Properties reflected in XML file
	m_GlobalProps["GRAVITY"]	= Prop(IPhysics::VEC4, 0.0f, -9.8f, 0.0f, 0.0f);

	m_GlobalProps["CAMERA_TIME"] = Prop(IPhysics::FLOAT, 0.016666666667f);
	m_GlobalProps["CAMERA_PACE"] = Prop(IPhysics::FLOAT, 0.2f);
	m_GlobalProps["CAMERA_MIN_PACE"] = Prop(IPhysics::FLOAT, 0.05f);
	m_GlobalProps["CAMERA_HIT_MAGNITUDE"] = Prop(IPhysics::FLOAT, 100.0f);
	m_GlobalProps["CAMERA_STEP_OFFSET"] = Prop(IPhysics::FLOAT, 0.2f);
	m_GlobalProps["CAMERA_MASS"] = Prop(IPhysics::FLOAT, 10.0f);
	m_GlobalProps["CAMERA_STATIC_FRICTION"] = Prop(IPhysics::FLOAT, 0.5f);
	m_GlobalProps["CAMERA_DYNAMIC_FRICTION"] = Prop(IPhysics::FLOAT, 0.5f);
	m_GlobalProps["CAMERA_RESTITUTION"] = Prop(IPhysics::FLOAT, 0.5f);

	m_MaterialProps["MASS"]		= Prop(IPhysics::FLOAT, 0.0f);
	m_MaterialProps["INERTIA"]	= Prop(IPhysics::VEC4, 0.0f, 0.0f, 0.0f, 1.0f);

	m_MaterialProps["STATIC_FRICTION"]	= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["DYNAMIC_FRICTION"] = Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["ROLLING_FRICTION"] = Prop(IPhysics::FLOAT, 0.0f);
	m_MaterialProps["RESTITUTION"]		= Prop(IPhysics::FLOAT, 1.0f);
	
	m_MaterialProps["VITERATIONS"] = Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["PITERATIONS"] = Prop(IPhysics::FLOAT, 1.0f);
	
	m_MaterialProps["PACE"]				= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["HIT_MAGNITUDE"]	= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["HEIGHT"]			= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["RADIUS"]			= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["STEP_OFFSET"]		= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["DIRECTION"]		= Prop(IPhysics::VEC4, 0.0f, 0.0f, -1.0f, 1.0f);

	m_MaterialProps["UP"] = Prop(IPhysics::VEC4, 0.0f, 1.0f, 0.0f, 1.0f);

	m_MaterialProps["IMPULSE"] = Prop(IPhysics::VEC4, 0.0f, 0.0f, 0.0f, 1.0f);

	worldManager = new BulletWorldManager();
	Prop p = m_GlobalProps["GRAVITY"];
	worldManager->setGravity(p.x, p.y, p.z);
}


NauBulletInterface::~NauBulletInterface() {
}

void
NauBulletInterface::setPropertyManager(nau::physics::IPhysicsPropertyManager * pm) {
	m_PropertyManager = pm;
}

void NauBulletInterface::update() {
	worldManager->update();
}

void NauBulletInterface::build() {
}

void NauBulletInterface::applyFloatProperty(const std::string & scene, const std::string & property, float value) {
	if (m_Scenes[scene].sceneType == SceneType::RIGID || m_Scenes[scene].sceneType == SceneType::STATIC)
		worldManager->setRigidProperty(scene, property, value);
	else if (m_Scenes[scene].sceneType == SceneType::CLOTH)
		worldManager->setSoftProperty(scene, property, value);
	else if (m_Scenes[scene].sceneType == SceneType::CHARACTER)
		worldManager->setCharacterProperty(scene, property, &value);
}

void NauBulletInterface::applyVec4Property(const std::string & scene, const std::string & property, float * value) {
	if (m_Scenes[scene].sceneType == SceneType::RIGID)
		worldManager->setRigidProperty(scene, property, value);
	else if (m_Scenes[scene].sceneType == SceneType::CHARACTER)
		worldManager->setCharacterProperty(scene, property, value);
}

void NauBulletInterface::applyGlobalFloatProperty(const std::string & property, float value) {
	if (property.compare("TIME_STEP") == 0)
		worldManager->setTimeStep(value);
}

void NauBulletInterface::applyGlobalVec4Property(const std::string & property, float * value) {
	if (property.compare("GRAVITY") == 0)
		worldManager->setGravity(value[0], value[1], value[2]);
}

void NauBulletInterface::setScene(const std::string &scene, const std::string & material, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform) {
	m_Scenes[scene].vertices = vertices;
	m_Scenes[scene].indices = indices;
	m_Scenes[scene].transform = transform;
	m_Scenes[scene].material = material;
	switch (m_Scenes[scene].sceneType)
	{
	case IPhysics::STATIC:
		worldManager->addRigid(
			scene,
			nbVertices,
			vertices,
			nbIndices,
			indices,
			transform,
			m_Scenes[scene].boundingVolume,
			m_PropertyManager->getMaterialFloatProperty(material, "MASS"),
			true
		);
		break;
	case IPhysics::RIGID:
		worldManager->addRigid(
			scene,
			nbVertices,
			vertices,
			nbIndices,
			indices,
			transform,
			m_Scenes[scene].boundingVolume,
			m_PropertyManager->getMaterialFloatProperty(material, "MASS")
		);
		break;
	case IPhysics::CLOTH:
		worldManager->addCloth(
			scene,
			nbVertices,
			vertices,
			nbIndices,
			indices,
			transform,
			m_Scenes[scene].sceneCondition,
			m_PropertyManager->getMaterialVec4Property(material, "SCENE_CONDITION_VALUE")
		);
		break;
	case IPhysics::CHARACTER:
		worldManager->addCharacter(
			scene,
			nbVertices,
			vertices,
			nbIndices,
			indices,
			transform,
			m_PropertyManager->getMaterialFloatProperty(material, "HEIGHT"),
			m_PropertyManager->getMaterialFloatProperty(material, "RADIUS"),
			m_PropertyManager->getMaterialFloatProperty(material, "STEP_OFFSET")
		);
		break;
	case IPhysics::DEBUG:
		worldManager->setDebug();
		break;
	default:
		break;
	}
}

float * NauBulletInterface::getSceneTransform(const std::string & scene) {
	return m_Scenes[scene].transform;
}

void NauBulletInterface::setSceneTransform(const std::string & scene, float * transform) {
	m_Scenes[scene].transform = transform;
	if (m_Scenes[scene].sceneType == SceneType::RIGID || m_Scenes[scene].sceneType == SceneType::STATIC)
		worldManager->moveRigid(scene, transform);
	if (m_Scenes[scene].sceneType == SceneType::CLOTH)
		worldManager->moveSoft(scene, transform);
}

void NauBulletInterface::setCameraAction(const std::string & scene, const std::string & action, float * value) {
	if (!worldManager->hasCamera(scene)) {
		if (action.compare("POSITION") == 0) {
			worldManager->addCamera(
				scene,
				value,
				m_PropertyManager->getGlobalFloatProperty("CAMERA_PACE"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_MIN_PACE"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_HIT_MAGNITUDE"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_STEP_OFFSET"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_RADIUS"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_HEIGHT")
			);
		}
	}
	else {
		//TODO: Move camera
		//if (action.compare("POSITION") == 0) {}
		worldManager->setCharacterProperty(scene, action, value);
	}
}

std::vector<float> * NauBulletInterface::getDebug() {
	return worldManager->getDebug();
}

std::map<std::string, nau::physics::IPhysics::Prop>& NauBulletInterface::getGlobalProperties() {
	return m_GlobalProps;
}

std::map<std::string, nau::physics::IPhysics::Prop>& NauBulletInterface::getMaterialProperties() {
	return m_MaterialProps;
}
