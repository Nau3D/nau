#include "nau/scene/geometricObject.h"
#include "nau/geometry/box.h"
#include "nau.h"
#include "nau/material/material.h"
#include <sstream>

using namespace nau::scene;
using namespace nau::geometry;


unsigned int GeometricObject::PrimitiveCounter = 0;

GeometricObject::GeometricObject() : SceneObject() {

	m_PrimitiveID = GeometricObject::PrimitiveCounter;
	++GeometricObject::PrimitiveCounter;

	std::stringstream z;
	z << "GeomObj";
	z << m_PrimitiveID;
	
	this->setName(z.str());
}


void
GeometricObject::eventReceived(const std::string &sender,
	const std::string &eventType,
	const std::shared_ptr<IEventData> &evt) {

}


GeometricObject::~GeometricObject() {
	
}


void 
GeometricObject::setRenderable (nau::render::IRenderable *renderable) {

	// It must be a primitive!
	m_Renderable = renderable;
}


void 
GeometricObject::setMaterial(const std::string &name) {

	std::shared_ptr<MaterialGroup> &mg = m_Renderable->getMaterialGroups().at(0);

	mg ->setMaterialName(name);
}


std::string 
GeometricObject::getType (void) {

	return "GeometricPrimitive";
}