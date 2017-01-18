#include "nau/material/materialArrayOfTextures.h"

#include "nau.h"

using namespace nau::material;

bool
MaterialArrayOfTextures::Init() {

	// INT
	Attribs.add(Attribute(FIRST_UNIT, "FIRST_UNIT", Enums::DataType::INT, false, new NauInt(-1)));
	// INTARRAY
	Attribs.add(Attribute(SAMPLER_ARRAY, "SAMPLER_ARRAY", Enums::DataType::INTARRAY, true));

#ifndef _WINDLL
	NAU->registerAttributes("ARRAY_OF_TEXTURES_BINDING", &Attribs);
#endif

	return true;
}


AttribSet MaterialArrayOfTextures::Attribs;
bool MaterialArrayOfTextures::Inited = Init();


MaterialArrayOfTextures::MaterialArrayOfTextures(int unit): m_TextureArray(NULL), m_Sampler(NULL) {

	registerAndInitArrays(Attribs);
	m_IntProps[FIRST_UNIT] = unit;
}


MaterialArrayOfTextures::MaterialArrayOfTextures(): m_TextureArray(NULL), m_Sampler(NULL) {

	registerAndInitArrays(Attribs);
}


MaterialArrayOfTextures::MaterialArrayOfTextures(const MaterialArrayOfTextures &mt) {

	m_Sampler = mt.m_Sampler;
	m_TextureArray = mt.m_TextureArray;
	copy(&(AttributeValues)mt);
}


MaterialArrayOfTextures::~MaterialArrayOfTextures(void) {

}


//NauIntArray 
//MaterialArrayOfTextures::setPropiv(IntArrayProperty prop) {
//
//	switch (prop) {
//	case SAMPLER_ARRAY:
//		return m_TextureArray->getPropiv(IArrayOfTextures::SAMPLER_ARRAY);
//		break;
//	default:
//		return AttributeValues::getPropiv(prop);
//	}
//}


void
MaterialArrayOfTextures::bind() {

	if (m_TextureArray)
		m_TextureArray->prepare(m_IntProps[FIRST_UNIT], m_Sampler);
}


void
MaterialArrayOfTextures::unbind() {

	if (m_TextureArray)
		m_TextureArray->restore(m_IntProps[FIRST_UNIT], m_Sampler);
}


void
MaterialArrayOfTextures::setArrayOfTextures(IArrayOfTextures *t) {

	// t must have at least one texture
	assert(t->getPropui(IArrayOfTextures::TEXTURE_COUNT));

	m_TextureArray = t;
	m_Sampler = ITextureSampler::create(m_TextureArray->getTexture(0));
}


IArrayOfTextures *
MaterialArrayOfTextures::getArrayOfTextures() {

	return m_TextureArray;
}


ITextureSampler *
MaterialArrayOfTextures::getSampler() {

	return m_Sampler;
}


void 
MaterialArrayOfTextures::setPropi(IntProperty prop, int value) {

		switch (prop) {
		case FIRST_UNIT: {
			m_IntProps[FIRST_UNIT] = value;
			m_IntArrayProps[SAMPLER_ARRAY].clear();
			if (m_TextureArray) {
				int s = m_TextureArray->getPropiv(IArrayOfTextures::TEXTURE_ID_ARRAY).size();

				for (int i = 0; i < s; ++i)
					m_IntArrayProps[SAMPLER_ARRAY].append(i + value);
			}
			break;
		default:
			 AttributeValues::setPropi(prop, value);
		}
	}


}