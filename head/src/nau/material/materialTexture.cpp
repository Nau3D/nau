#include "nau/material/materialTexture.h"

#include "nau.h"

using namespace nau::material;

bool
MaterialTexture::Init() {

	// INT
	Attribs.add(Attribute(UNIT, "UNIT", Enums::DataType::INT, false, new int(0)));

#ifndef _WINDLL
	NAU->registerAttributes("MATERIAL_TEXTURE", &Attribs);
#endif

	return true;
}


AttribSet MaterialTexture::Attribs;
bool MaterialTexture::Inited = Init();


MaterialTexture::MaterialTexture(int unit) : m_Texture(NULL), m_Sampler(NULL) {

	registerAndInitArrays(Attribs);
	m_IntProps[UNIT] = unit;
}


MaterialTexture::MaterialTexture() : m_Texture(NULL), m_Sampler(NULL) {

	registerAndInitArrays(Attribs);
}


MaterialTexture::MaterialTexture(const MaterialTexture &mt) {

	m_Texture = mt.m_Texture;
	m_Sampler = mt.m_Sampler;
	copy(&(AttributeValues)mt);
}


const MaterialTexture&
MaterialTexture::operator =(const MaterialTexture &mt) {

	if (this != &mt) {
		m_Texture = mt.m_Texture;
		m_Sampler = mt.m_Sampler;
		copy(&(AttributeValues)mt);
	}
	return *this;
}


void
MaterialTexture::bind() {

	assert(m_IntProps[UNIT] >= 0);

	RENDERER->addTexture(this);
	m_Texture->prepare(m_IntProps[UNIT], m_Sampler);
}


void
MaterialTexture::unbind() {

	assert(m_IntProps[UNIT] >= 0);

	RENDERER->removeTexture(m_IntProps[UNIT]);
	m_Texture->restore(m_IntProps[UNIT], m_Sampler);
}


void 
MaterialTexture::setTexture(ITexture *t) {

	m_Texture = t;
}


void MaterialTexture::setSampler(ITextureSampler *s) {

	m_Sampler = s;
}


ITexture *
MaterialTexture::getTexture() {

	return m_Texture;
}


ITextureSampler *
MaterialTexture::getSampler() {

	return m_Sampler;
}


