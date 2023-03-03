#include "nau/material/colorMaterial.h"

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau.h"


using namespace nau::material;
using namespace nau::render;


bool
ColorMaterial::Init() {

	// VEC4
	Attribs.add(Attribute(AMBIENT, "AMBIENT", Enums::DataType::VEC4, false, new vec4 (0.2f, 0.2f, 0.2f, 1.0f), NULL, NULL, IAPISupport::APIFeatureSupport::OK, Attribute::Semantics::COLOR));
	Attribs.add(Attribute(DIFFUSE, "DIFFUSE", Enums::DataType::VEC4, false, new vec4 (0.8f, 0.8f, 0.8f, 1.0f), NULL, NULL, IAPISupport::APIFeatureSupport::OK, Attribute::Semantics::COLOR));
	Attribs.add(Attribute(SPECULAR, "SPECULAR", Enums::DataType::VEC4, false, new vec4 (0.0f, 0.0f, 0.0f, 1.0f), NULL, NULL, IAPISupport::APIFeatureSupport::OK, Attribute::Semantics::COLOR));
	Attribs.add(Attribute(EMISSION, "EMISSION", Enums::DataType::VEC4, false, new vec4 (0.0f, 0.0f, 0.0f, 1.0f), NULL, NULL, IAPISupport::APIFeatureSupport::OK, Attribute::Semantics::COLOR));
	// FLOAT
	Attribs.add(Attribute(SHININESS, "SHININESS", Enums::DataType::FLOAT, false, new NauFloat(0)));

//#ifndef _WINDLL
	NAU->registerAttributes("COLOR", &Attribs);
	//#endif

	return true;
}


AttribSet ColorMaterial::Attribs;
bool ColorMaterial::Inited = Init();

AttribSet &
ColorMaterial::GetAttribs() {
	return Attribs;
}


ColorMaterial::ColorMaterial() {

	registerAndInitArrays(Attribs);
}


ColorMaterial::~ColorMaterial() {

   //dtor
}


void 
ColorMaterial::prepare () {
		
	RENDERER->setColorMaterial(*this);
}


void 
ColorMaterial::restore() {


	vec4 diffuse = *(std::dynamic_pointer_cast<vec4>
		(ColorMaterial::Attribs.get((AttributeValues::Float4Property)DIFFUSE, Enums::VEC4)->getDefault()));
	vec4 ambient = *(std::dynamic_pointer_cast<vec4>
		(ColorMaterial::Attribs.get((AttributeValues::Float4Property)AMBIENT, Enums::VEC4)->getDefault()));
	vec4 emission = *(std::dynamic_pointer_cast<vec4>
		(ColorMaterial::Attribs.get((AttributeValues::Float4Property)EMISSION, Enums::VEC4)->getDefault()));
	vec4 specular = *(std::dynamic_pointer_cast<vec4>
		(ColorMaterial::Attribs.get((AttributeValues::Float4Property)SPECULAR, Enums::VEC4)->getDefault()));
	float shininess = (std::dynamic_pointer_cast<NauFloat>
		(ColorMaterial::Attribs.get((AttributeValues::FloatProperty)SHININESS, Enums::FLOAT)->getDefault()))->getNumber();
	
	RENDERER->setColorMaterial(diffuse, ambient, emission, specular, shininess);
}


void 
ColorMaterial::clear() {

	initArrays(Attribs);	
}


void 
ColorMaterial::clone( ColorMaterial &mat)
{
	copy(&mat);
}


