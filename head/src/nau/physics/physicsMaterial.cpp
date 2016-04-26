#include "nau/physics/physicsMaterial.h"

#include "nau.h"
#include "nau/enums.h"
#include "nau/math/data.h"

using namespace nau::physics;

bool
PhysicsMaterial::Init() {

	Attribs.add(Attribute(MASS, "MASS", Enums::DataType::FLOAT, false, new NauFloat(1.0f)));

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


PhysicsMaterial::PhysicsMaterial() {

	registerAndInitArrays(Attribs);
}