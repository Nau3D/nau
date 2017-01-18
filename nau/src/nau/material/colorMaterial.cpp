#include "nau/material/colorMaterial.h"

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau.h"


using namespace nau::material;
using namespace nau::render;


bool
ColorMaterial::Init() {

	// VEC4
	Attribs.add(Attribute(AMBIENT, "AMBIENT", Enums::DataType::VEC4, false, new vec4 (0.2f, 0.2f, 0.2f, 1.0f), NULL, NULL, IAPISupport::OK, Attribute::Semantics::COLOR));
	Attribs.add(Attribute(DIFFUSE, "DIFFUSE", Enums::DataType::VEC4, false, new vec4 (0.8f, 0.8f, 0.8f, 1.0f), NULL, NULL, IAPISupport::OK, Attribute::Semantics::COLOR));
	Attribs.add(Attribute(SPECULAR, "SPECULAR", Enums::DataType::VEC4, false, new vec4 (0.0f, 0.0f, 0.0f, 1.0f), NULL, NULL, IAPISupport::OK, Attribute::Semantics::COLOR));
	Attribs.add(Attribute(EMISSION, "EMISSION", Enums::DataType::VEC4, false, new vec4 (0.0f, 0.0f, 0.0f, 1.0f), NULL, NULL, IAPISupport::OK, Attribute::Semantics::COLOR));
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


//void 
//ColorMaterial::setProp(int prop, Enums::DataType type, void *value) {
//
//	switch (type) {
//
//		case Enums::FLOAT:
//			m_FloatProps[prop] = *(float *)value;
//			break;
//		case Enums::VEC4:
//			m_Float4Props[prop].set((vec4 *)value);
//			break;
//	}
//}		
//
//
//void
//ColorMaterial::setProp(FloatProperty prop, float value) {
//
//	assert(m_FloatProps.count(prop) > 0 );
//	m_FloatProps[prop] = value;
//}
//
//
//void
//ColorMaterial::setProp(Float4Property prop, float r, float g, float b, float a) {
//
//	assert(m_Float4Props.count(prop) > 0 && m_Float4Props.size()==4);
//	m_Float4Props[prop].set(r,g,b,a);
//}
//
//
//void
//ColorMaterial::setProp(Float4Property prop, float *v) {
//
//	assert(m_Float4Props.count(prop) > 0 && m_Float4Props.size()==4);
//	m_Float4Props[prop].set(v[0], v[1], v[2], v[3]);
//}
//
//
//void
//ColorMaterial::setProp(Float4Property prop, const vec4& values){
//
//	assert(m_Float4Props.count(prop) > 0 && m_Float4Props.size()==4);
//	m_Float4Props[prop].set(values);
//}


//float 
//ColorMaterial::getPropf(FloatProperty aProp)   {
//	
//	float f = m_FloatProps[aProp];
//	return f;
//}


//const vec4&
//ColorMaterial::getProp4f(Float4Property aProp)   {
//
//	assert(m_Float4Props.size()==4);
//	return this->m_Float4Props[aProp];
//}


