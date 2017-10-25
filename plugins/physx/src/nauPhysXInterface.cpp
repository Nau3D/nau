#include "nauPhysXInterface.h"

#include <memory>

static char className[] = "PhysX plugin for Nau3D";
static NauPhysXInterface * Instance = NULL;

__declspec(dllexport)
void *
createPhysics() {
	Instance = new NauPhysXInterface();
	return Instance;
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

NauPhysXInterface * NauPhysXInterface::Create() {
	return new NauPhysXInterface();
}

NauPhysXInterface::NauPhysXInterface() {
	//INFO: Declare Physics Properties reflected in XML file
	m_GlobalProps["GRAVITY"]	= Prop(IPhysics::VEC4, 0.0f, -9.8f, 0.0f, 0.0f);
	
	m_GlobalProps["CAMERA_TIME"]				= Prop(IPhysics::FLOAT, 0.016666666667f);
	m_GlobalProps["CAMERA_PACE"]				= Prop(IPhysics::FLOAT, 0.2f);
	m_GlobalProps["CAMERA_MIN_PACE"]			= Prop(IPhysics::FLOAT, 0.05f);
	m_GlobalProps["CAMERA_HIT_MAGNITUDE"]		= Prop(IPhysics::FLOAT, 100.0f);
	m_GlobalProps["CAMERA_STEP_OFFSET"]			= Prop(IPhysics::FLOAT, 0.2f);
	m_GlobalProps["CAMERA_MASS"]				= Prop(IPhysics::FLOAT, 10.0f);
	m_GlobalProps["CAMERA_STATIC_FRICTION"]		= Prop(IPhysics::FLOAT, 0.5f);
	m_GlobalProps["CAMERA_DYNAMIC_FRICTION"]	= Prop(IPhysics::FLOAT, 0.5f);
	m_GlobalProps["CAMERA_RESTITUTION"]			= Prop(IPhysics::FLOAT, 0.5f);

	m_MaterialProps["MASS"]		= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["INERTIA"]	= Prop(IPhysics::VEC4, 1.0f, 1.0f, 1.0f, 1.0f);

	m_MaterialProps["STATIC_FRICTION"]	= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["DYNAMIC_FRICTION"]	= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["ROLLING_FRICTION"] = Prop(IPhysics::FLOAT, -1.0f);
	m_MaterialProps["RESTITUTION"]		= Prop(IPhysics::FLOAT, 1.0f);
	
	m_MaterialProps["SOLVER_FREQUENCY"]		= Prop(IPhysics::FLOAT, 240.0f);
	m_MaterialProps["INERTIA_SCALE"]		= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["VERTICAL_STRETCH"]		= Prop(IPhysics::VEC4, 1.0f, 1.0f, 1.0f, 1.0f);
	m_MaterialProps["HORIZONTAL_STRETCH"]	= Prop(IPhysics::VEC4, 1.0f, 1.0f, 1.0f, 1.0f);
	m_MaterialProps["SHEARING"]				= Prop(IPhysics::VEC4, 1.0f, 1.0f, 1.0f, 1.0f);
	m_MaterialProps["BENDING"]				= Prop(IPhysics::VEC4, 1.0f, 1.0f, 1.0f, 1.0f);
	m_MaterialProps["FRICTION_COEFFICIENT"] = Prop(IPhysics::FLOAT, 0.0f);
	m_MaterialProps["COLLISION_MASS_SCALE"] = Prop(IPhysics::FLOAT, 0.0f);
	m_MaterialProps["SELF_COLLISION_DISTANCE"]	=Prop(IPhysics::FLOAT, 0.0f);
	m_MaterialProps["SELF_COLLISION_STIFFNESS"] = Prop(IPhysics::FLOAT, 1.0f);

	m_MaterialProps["PACE"]				= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["HIT_MAGNITUDE"]	= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["HEIGHT"]			= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["RADIUS"]			= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["STEP_OFFSET"]		= Prop(IPhysics::FLOAT, 1.0f);
	m_MaterialProps["DIRECTION"]		= Prop(IPhysics::VEC4, 0.0f, 0.0f, -1.0f, 1.0f);
	
	m_MaterialProps["UP"] = Prop(IPhysics::VEC4, 0.0f, 1.0f, 0.0f, 1.0f);

	m_MaterialProps["IMPULSE"] = Prop(IPhysics::VEC4, 0.0f, 0.0f, 0.0f, 1.0f);
	
	worldManager = new PhysXWorldManager();
}


NauPhysXInterface::~NauPhysXInterface() {
}

void NauPhysXInterface::setPropertyManager(nau::physics::IPhysicsPropertyManager * pm) {
	m_PropertyManager = pm;
}

void NauPhysXInterface::update() {
	worldManager->update();
	for (auto particleMaterial : *worldManager->getMaterialParticleNb())
		m_PropertyManager->setMaterialFloatProperty(particleMaterial.first, "NBPARTICLES", static_cast<float>(particleMaterial.second));
}

void NauPhysXInterface::build() {	
}

void NauPhysXInterface::applyFloatProperty(const std::string & scene, const std::string & property, float value) {
	if (m_Scenes[scene].sceneType == SceneType::RIGID || m_Scenes[scene].sceneType == SceneType::STATIC)
		worldManager->setRigidProperty(scene, property, value);
	else if (m_Scenes[scene].sceneType == SceneType::CLOTH) 
		worldManager->setSoftProperty(scene, property, value);
	else if (m_Scenes[scene].sceneType == SceneType::CHARACTER) 
		worldManager->setCharacterProperty(scene, property, &value);
}

void NauPhysXInterface::applyVec4Property(const std::string & scene, const std::string & property, float * value) {
	if (m_Scenes[scene].sceneType == SceneType::RIGID) 
		worldManager->setRigidProperty(scene, property, value);
	else if (m_Scenes[scene].sceneType == SceneType::CLOTH) 
		worldManager->setSoftProperty(scene, property, value);
	else if (m_Scenes[scene].sceneType == SceneType::CHARACTER)
		worldManager->setCharacterProperty(scene, property, value);
}

void NauPhysXInterface::applyGlobalFloatProperty(const std::string & property, float value) {
	if (property.compare("TIME_STEP") == 0)
		worldManager->setTimeStep(value);
}

void NauPhysXInterface::applyGlobalVec4Property(const std::string & property, float * value) {
	if (property.compare("GRAVITY") == 0)
		worldManager->setGravity(value[0], value[1], value[2]);
}

void NauPhysXInterface::setScene(const std::string &scene, const std::string & material, int nbVertices, float * vertices, int nbIndices, unsigned int * indices, float * transform) {
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
			worldManager->createMaterial(
				m_PropertyManager->getMaterialFloatProperty(material, "DYNAMIC_FRICTION"),
				m_PropertyManager->getMaterialFloatProperty(material, "STATIC_FRICTION"),
				m_PropertyManager->getMaterialFloatProperty(material, "RESTITUTION")
			),
			m_Scenes[scene].boundingVolume,
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
			worldManager->createMaterial(
				m_PropertyManager->getMaterialFloatProperty(material, "DYNAMIC_FRICTION"),
				m_PropertyManager->getMaterialFloatProperty(material, "STATIC_FRICTION"),
				m_PropertyManager->getMaterialFloatProperty(material, "RESTITUTION")
			),
			m_Scenes[scene].boundingVolume
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
			worldManager->createMaterial(
				m_PropertyManager->getMaterialFloatProperty(material, "DYNAMIC_FRICTION"),
				m_PropertyManager->getMaterialFloatProperty(material, "STATIC_FRICTION"),
				m_PropertyManager->getMaterialFloatProperty(material, "RESTITUTION")
			),
			m_PropertyManager->getMaterialVec4Property(material, "UP")
		);
	case IPhysics::PARTICLES:
		worldManager->addParticles(
			scene,
			material,
			m_PropertyManager->getMaterialFloatProperty(material, "MAX_PARTICLES"),
			vertices,
			transform
		);
	default:
		break;
	}
}

float * NauPhysXInterface::getSceneTransform(const std::string & scene) {
	return m_Scenes[scene].transform;
}

void NauPhysXInterface::setSceneTransform(const std::string & scene, float * transform) {
	m_Scenes[scene].transform = transform;
	if (m_Scenes[scene].sceneType == SceneType::RIGID || m_Scenes[scene].sceneType == SceneType::STATIC)
		worldManager->moveRigid(scene, transform);
	else if (m_Scenes[scene].sceneType == SceneType::CLOTH)
		worldManager->moveSoft(scene, transform);
	else if (m_Scenes[scene].sceneType == SceneType::CHARACTER)
		worldManager->moveCharacter(scene, transform);
}

void NauPhysXInterface::setCameraAction(const std::string & scene, const std::string & action, float * value) {
	if (!worldManager->hasCamera(scene)) {
		if (action.compare("POSITION") == 0) {
			worldManager->addCamera(
				scene,
				value,
				m_PropertyManager->getGlobalVec4Property("CAMERA_UP"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_PACE"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_MIN_PACE"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_HIT_MAGNITUDE"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_TIME"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_STEP_OFFSET"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_MASS"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_RADIUS"),
				m_PropertyManager->getGlobalFloatProperty("CAMERA_HEIGHT"),
				worldManager->createMaterial(
					m_PropertyManager->getGlobalFloatProperty("CAMERA_DYNAMIC_FRICTION"),
					m_PropertyManager->getGlobalFloatProperty("CAMERA_STATIC_FRICTION"),
					m_PropertyManager->getGlobalFloatProperty("CAMERA_RESTITUTION")
				)
			);
		}
	}
	else {
		//TODO: Move camera
		//if (action.compare("POSITION") == 0) {}
		worldManager->setCharacterProperty(scene, action, value);
	}
}

std::vector<float> * NauPhysXInterface::getDebug() {
	return nullptr;
}

std::map<std::string, nau::physics::IPhysics::Prop>& NauPhysXInterface::getGlobalProperties() {
	return m_GlobalProps;
}

std::map<std::string, nau::physics::IPhysics::Prop>& NauPhysXInterface::getMaterialProperties() {
	return m_MaterialProps;
}
