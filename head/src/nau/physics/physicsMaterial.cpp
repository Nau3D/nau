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

	NAU->registerAttributes("PHYSICS_MATERIAL", &Attribs);

	return true;
}


AttribSet PhysicsMaterial::Attribs;
bool PhysicsMaterial::Inited = Init();


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
