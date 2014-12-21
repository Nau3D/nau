#include <nau/material/colormaterial.h>

#include <nau/attribute.h>
#include <nau/attributeValues.h>
#include <nau.h>


using namespace nau::material;
using namespace nau::render;


bool
ColorMaterial::Init() {

	// VEC4
	Attribs.add(Attribute(AMBIENT, "AMBIENT", Enums::DataType::VEC4, false, new vec4 (0.2f, 0.2f, 0.2f, 1.0f)));
	Attribs.add(Attribute(DIFFUSE, "DIFFUSE", Enums::DataType::VEC4, false, new vec4 (0.8f, 0.8f, 0.8f, 1.0f)));
	Attribs.add(Attribute(SPECULAR, "SPECULAR", Enums::DataType::VEC4, false, new vec4 (0.0f, 0.0f, 0.0f, 1.0f)));
	Attribs.add(Attribute(EMISSION, "EMISSION", Enums::DataType::VEC4, false, new vec4 (0.0f, 0.0f, 0.0f, 1.0f)));
	// FLOAT
	Attribs.add(Attribute(SHININESS, "SHININESS", Enums::DataType::FLOAT, false, new float(0)));

	return true;
}


AttribSet ColorMaterial::Attribs;
bool ColorMaterial::Inited = Init();


//void
//ColorMaterial::initArrays() {
//
//	Attribs.initAttribInstanceVec4Array(m_Float4Props);
//	Attribs.initAttribInstanceFloatArray(m_FloatProps);
//
//}


ColorMaterial::ColorMaterial() {

	initArrays(Attribs);
}


ColorMaterial::~ColorMaterial() {

   //dtor
}


void 
ColorMaterial::prepare () {
		
	RENDERER->setMaterial(*this);
}


void 
ColorMaterial::restore() {

	assert(m_Float4Props.size()==4);
	float *ambient,*specular,*diffuse,*emission,*shininess;
	
	diffuse = (float *)ColorMaterial::Attribs.getDefault(DIFFUSE, Enums::DataType::VEC4);
	ambient = (float *)ColorMaterial::Attribs.getDefault(AMBIENT, Enums::DataType::VEC4);
	emission = (float *)ColorMaterial::Attribs.getDefault(EMISSION, Enums::DataType::VEC4);
	specular = (float *)ColorMaterial::Attribs.getDefault(SPECULAR, Enums::DataType::VEC4);
	shininess = (float *)ColorMaterial::Attribs.getDefault(SHININESS, Enums::DataType::FLOAT);
	
	RENDERER->setMaterial(diffuse, ambient, emission, specular, *shininess);
	assert(m_Float4Props.size()==4);
}


void 
ColorMaterial::clear() {

	assert(m_Float4Props.size()==4);
	setProp(ColorMaterial::DIFFUSE, *(vec4 *)ColorMaterial::Attribs.getDefault(DIFFUSE, Enums::DataType::VEC4));
	setProp(ColorMaterial::AMBIENT, *(vec4 *)ColorMaterial::Attribs.getDefault(AMBIENT, Enums::DataType::VEC4));
	setProp(ColorMaterial::EMISSION, *(vec4 *)ColorMaterial::Attribs.getDefault(EMISSION, Enums::DataType::VEC4));
	setProp(ColorMaterial::SPECULAR, *(vec4 *)ColorMaterial::Attribs.getDefault(SPECULAR, Enums::DataType::VEC4));
	setProp(ColorMaterial::SHININESS, *(float *)ColorMaterial::Attribs.getDefault(SHININESS, Enums::DataType::FLOAT));
	assert(m_Float4Props.size()==4);
	
}


void 
ColorMaterial::clone( ColorMaterial &mat)
{
	assert(m_Float4Props.size()==4);
	setProp(SHININESS, mat.getPropf(SHININESS));
	setProp(DIFFUSE, mat.getPropf4(DIFFUSE));
	setProp(AMBIENT, mat.getPropf4(AMBIENT));
	setProp(SPECULAR, mat.getPropf4(SPECULAR));
	setProp(EMISSION, mat.getPropf4(EMISSION));
	assert(m_Float4Props.size()==4);
}


void 
ColorMaterial::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {

		case Enums::FLOAT:
			m_FloatProps[prop] = *(float *)value;
			break;
		case Enums::VEC4:
			m_Float4Props[prop].set((vec4 *)value);
			break;
	}
}		


void
ColorMaterial::setProp(FloatProperty prop, float value) {

	assert(m_FloatProps.count(prop) > 0 );
	m_FloatProps[prop] = value;
}


void
ColorMaterial::setProp(Float4Property prop, float r, float g, float b, float a) {

	assert(m_Float4Props.count(prop) > 0 && m_Float4Props.size()==4);
	m_Float4Props[prop].set(r,g,b,a);
}


void
ColorMaterial::setProp(Float4Property prop, float *v) {

	assert(m_Float4Props.count(prop) > 0 && m_Float4Props.size()==4);
	m_Float4Props[prop].set(v[0], v[1], v[2], v[3]);
}


void
ColorMaterial::setProp(Float4Property prop, const vec4& values){

	assert(m_Float4Props.count(prop) > 0 && m_Float4Props.size()==4);
	m_Float4Props[prop].set(values);
}


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


