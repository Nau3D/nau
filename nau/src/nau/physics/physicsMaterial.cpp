#include "nau/physics/physicsMaterial.h"

#include "nau.h"
#include "nau/enums.h"
#include "nau/math/data.h"

using namespace nau::physics;

bool
PhysicsMaterial::Init() {

	Attribs.add(Attribute(SCENE_TYPE, "SCENE_TYPE", Enums::DataType::ENUM, false, new NauInt(IPhysics::STATIC)));
	Attribs.listAdd("SCENE_TYPE", "STATIC", IPhysics::STATIC);
	Attribs.listAdd("SCENE_TYPE", "RIGID", IPhysics::RIGID);
	Attribs.listAdd("SCENE_TYPE", "CLOTH", IPhysics::CLOTH);
	Attribs.listAdd("SCENE_TYPE", "PARTICLES", IPhysics::PARTICLES);
	Attribs.listAdd("SCENE_TYPE", "CHARACTER", IPhysics::CHARACTER);
	Attribs.listAdd("SCENE_TYPE", "DEBUG", IPhysics::DEBUG);

	Attribs.add(Attribute(MAX_PARTICLE, "MAX_PARTICLES",  Enums::DataType::FLOAT, false, new NauFloat(0.0f)));
	Attribs.add(Attribute(NBPARTICLES, "NBPARTICLES", Enums::DataType::FLOAT, false, new NauFloat(0.0f)));

	Attribs.add(Attribute(BUFFER, "BUFFER", "BUFFER"));

	Attribs.add(Attribute(SCENE_SHAPE, "SCENE_SHAPE", Enums::DataType::ENUM, false, new NauInt(IPhysics::CUSTOM)));
	Attribs.listAdd("SCENE_SHAPE", "CUSTOM", IPhysics::CUSTOM);
	Attribs.listAdd("SCENE_SHAPE", "BOX", IPhysics::BOX);
	Attribs.listAdd("SCENE_SHAPE", "SPHERE", IPhysics::SPHERE);
	Attribs.listAdd("SCENE_SHAPE", "CAPSULE", IPhysics::CAPSULE);

	Attribs.add(Attribute(SCENE_CONDITION, "SCENE_CONDITION", Enums::DataType::ENUM, false, new NauInt(IPhysics::NONE)));
	Attribs.listAdd("SCENE_CONDITION", "GT", IPhysics::GT);
	Attribs.listAdd("SCENE_CONDITION", "LT", IPhysics::LT);
	Attribs.listAdd("SCENE_CONDITION", "EGT", IPhysics::EGT);
	Attribs.listAdd("SCENE_CONDITION", "ELT", IPhysics::ELT);
	Attribs.listAdd("SCENE_CONDITION", "EQ", IPhysics::EQ);
	Attribs.listAdd("SCENE_CONDITION", "NONE", IPhysics::NONE);
	Attribs.add(Attribute(SCENE_CONDITION_VALUE, "SCENE_CONDITION_VALUE", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 0.0f, 0.0f)));

	Attribs.add(Attribute(DIRECTION, "DIRECTION", Enums::DataType::VEC4, true, new vec4(0.0f, 0.0f, 0.0f, 1.0f)));

	//#ifndef _WINDLL
	NAU->registerAttributes("PHYSICS_MATERIAL", &Attribs);
	//#endif

	return true;
}


AttribSet PhysicsMaterial::Attribs;
bool PhysicsMaterial::Inited = Init();


AttribSet &
PhysicsMaterial::GetAttribs() {
	return Attribs;
}


PhysicsMaterial::PhysicsMaterial(const std::string &name): m_Name(name) {

	registerAndInitArrays(Attribs);
}


PhysicsMaterial::PhysicsMaterial() : m_Name("__nauDefault") {

	registerAndInitArrays(Attribs);
}


void
PhysicsMaterial::setPropf(FloatProperty p, float value) {

	m_FloatProps[p] = value;
	PhysicsManager::GetInstance()->applyMaterialFloatProperty(m_Name, Attribs.getName(p, Enums::FLOAT), value);
}


void
PhysicsMaterial::setPropf4(Float4Property p, vec4 &value) {

	m_Float4Props[p] = value;
	PhysicsManager::GetInstance()->applyMaterialVec4Property(m_Name, Attribs.getName(p, Enums::VEC4), &value.x);
}

void 
PhysicsMaterial::setProps(StringProperty prop, std::string & value) {
	m_StringProps[prop] = value;
	RESOURCEMANAGER->createBuffer(value);
}
