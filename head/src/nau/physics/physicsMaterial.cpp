#include "nau/physics/physicsMaterial.h"

#include "nau.h"
#include "nau/enums.h"
#include "nau/math/data.h"

using namespace nau::physics;

bool
PhysicsMaterial::Init() {

	Attribs.add(Attribute(MASS, "MASS", Enums::DataType::FLOAT, false, new NauFloat(1.0f)));

	NAU->registerAttributes("PHYSICS_MATERIAL", &Attribs);

	return true;
}

AttribSet PhysicsMaterial::Attribs;
bool PhysicsMaterial::Inited = Init();