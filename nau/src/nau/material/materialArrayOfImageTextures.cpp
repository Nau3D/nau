#include "nau/material/materialArrayOfImageTextures.h"

#include "nau.h"
#ifdef NAU_OPENGL
#include "nau/render/opengl/glMaterialArrayImageTextures.h"
#endif

using namespace nau::material;

bool
MaterialArrayOfImageTextures::Init() {

	// UINT
	Attribs.add(Attribute(LEVEL, "LEVEL", Enums::DataType::UINT, false, new NauUInt(0)));
	// ENUM
	Attribs.add(Attribute(ACCESS, "ACCESS", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(INTERNAL_FORMAT, "INTERNAL_FORMAT", Enums::DataType::ENUM, false));
	// BOOL
	Attribs.add(Attribute(CLEAR, "CLEAR", Enums::DataType::BOOL, false, new NauInt(false)));
	// INT
	Attribs.add(Attribute(FIRST_UNIT, "FIRST_UNIT", Enums::DataType::INT, true, new NauInt(-1)));
	// INTARRAY
	Attribs.add(Attribute(SAMPLER_ARRAY, "SAMPLER_ARRAY", Enums::DataType::INTARRAY, true));

	//#ifndef _WINDLL
	NAU->registerAttributes("ARRAY_OF_IMAGE_TEXTURES", &Attribs);
	//#endif

	return true;
}


AttribSet MaterialArrayOfImageTextures::Attribs;
bool MaterialArrayOfImageTextures::Inited = Init();


MaterialArrayOfImageTextures *
MaterialArrayOfImageTextures::Create() {
	
#ifdef NAU_OPENGL
	return new GLMaterialImageTextureArray();
#endif 
	return NULL;
}

MaterialArrayOfImageTextures::MaterialArrayOfImageTextures(): m_TextureArray(NULL) {

	registerAndInitArrays(Attribs);
}


MaterialArrayOfImageTextures::MaterialArrayOfImageTextures(const MaterialArrayOfImageTextures &mt) {

	const MaterialArrayOfImageTextures *mt2 = &mt;
	copy((AttributeValues *)mt2);
	m_TextureArray = mt.m_TextureArray; 
	//shallow copy
	m_ImageTextures = mt.m_ImageTextures;
}


MaterialArrayOfImageTextures::~MaterialArrayOfImageTextures(void) {

	for (int i = 0; i < m_ImageTextures.size(); ++i) {
		delete m_ImageTextures[i];
	}
}


void
MaterialArrayOfImageTextures::bind() {

	for (int i = 0; i < m_ImageTextures.size(); ++i) {
		m_ImageTextures[i]->prepare();
	}
}


void
MaterialArrayOfImageTextures::unbind() {

	for (int i = 0; i < m_ImageTextures.size(); ++i) {
		m_ImageTextures[i]->restore();
	}
}


void
MaterialArrayOfImageTextures::setArrayOfTextures(IArrayOfTextures *t) {

	// t must have at least one texture
	assert(t->getPropui(IArrayOfTextures::TEXTURE_COUNT));

	m_TextureArray = t;
	unsigned int  count = m_TextureArray->getPropui(IArrayOfTextures::TEXTURE_COUNT);
	m_ImageTextures.resize(count);

	m_IntArrayProps[SAMPLER_ARRAY].clear();
	for (unsigned int i = 0; i < count; ++i) {
		m_ImageTextures[i] = IImageTexture::Create("dummy", m_IntProps[FIRST_UNIT] + i,
			m_TextureArray->getTexture(i)->getPropi(ITexture::ID));
		m_IntArrayProps[SAMPLER_ARRAY].append(i + m_IntProps[FIRST_UNIT]);
	}
}


IArrayOfTextures *
MaterialArrayOfImageTextures::getArrayOfTextures() {

	return m_TextureArray;
}


void 
MaterialArrayOfImageTextures::setPropi(IntProperty prop, int value) {

	switch (prop) {
	case FIRST_UNIT: 
		m_IntProps[FIRST_UNIT] = value;
		m_IntArrayProps[SAMPLER_ARRAY].clear();
		for (int i = 0; i < m_ImageTextures.size(); ++i) {
			m_ImageTextures[i]->setPropi(IImageTexture::UNIT, value + i); 
			m_IntArrayProps[SAMPLER_ARRAY].append(i + value);
		}
		break;
	default:
		AttributeValues::setPropi(prop, value);
	}
}


void
MaterialArrayOfImageTextures::setPropui(UIntProperty prop, unsigned int value) {

	switch (prop) {
	case LEVEL:
		m_UIntProps[LEVEL] = value;
		for (int i = 0; i < m_ImageTextures.size(); ++i) {
			m_ImageTextures[i]->setPropui(IImageTexture::LEVEL, value);
		}
		break;
	default:
		AttributeValues::setPropui(prop, value);
	}
}


void
MaterialArrayOfImageTextures::setPrope(EnumProperty prop, int value) {

	switch (prop) {
	case ACCESS:
		m_EnumProps[ACCESS] = value;
		for (int i = 0; i < m_ImageTextures.size(); ++i) {
			m_ImageTextures[i]->setPrope(IImageTexture::ACCESS, value);
		}
		break;
	case INTERNAL_FORMAT:
		m_EnumProps[INTERNAL_FORMAT] = value;
		for (int i = 0; i < m_ImageTextures.size(); ++i) {
			m_ImageTextures[i]->setPrope(IImageTexture::INTERNAL_FORMAT, value);
		}
		break;
	default:
		AttributeValues::setPrope(prop, value);
	}
}


void
MaterialArrayOfImageTextures::setPropb(BoolProperty prop, bool value) {

	switch (prop) {
	case CLEAR:
		m_IntProps[CLEAR] = value;
		for (int i = 0; i < m_ImageTextures.size(); ++i) {
			m_ImageTextures[i]->setPropb(IImageTexture::CLEAR, value);
		}
		break;
	default:
		AttributeValues::setPropb(prop, value);
	}
}