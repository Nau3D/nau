#include "nau/world/physicalObject.h"

using namespace nau::world;

PhysicalObject::PhysicalObject():
	m_SceneObject (0)
{
}

PhysicalObject::~PhysicalObject()
{
	delete m_SceneObject;
}
