#include "nau/render/opengl/glimagetexture.h"

#if NAU_OPENGL_VERSION >=  420

#include "nau.h"
#include "nau/render/irenderer.h"

using namespace nau::render;

bool GLImageTexture::Inited = GLImageTexture::InitGL();


bool
GLImageTexture::InitGL() {

	Attribs.listAdd("ACCESS", "READ_ONLY", GL_READ_ONLY);
	Attribs.listAdd("ACCESS", "WRITE_ONLY", GL_WRITE_ONLY);
	Attribs.listAdd("ACCESS", "READ_WRITE", GL_READ_WRITE);

	return(true);
};


GLImageTexture::GLImageTexture(std::string label, unsigned int unit, unsigned int texID, unsigned int level, unsigned int access) : ImageTexture() {

	m_EnumProps[ACCESS] = access;
	m_UIntProps[LEVEL] = level;
	m_UIntProps[TEX_ID] = texID;
	m_IntProps[UNIT] = unit;
	nau::render::Texture* t = RESOURCEMANAGER->getTextureByID(texID);
	assert(t != NULL);
	m_Format = t->getPrope(Texture::FORMAT);
	m_Type = t->getPrope(Texture::TYPE);
	m_Dimension = t->getPrope(Texture::DIMENSION);
	m_InternalFormat = t->getPrope(Texture::INTERNAL_FORMAT);
	memset(m_Data, 0, sizeof(m_Data));
	m_Label = label;
}


GLImageTexture::~GLImageTexture(void) {

}


void 
GLImageTexture::prepare() {

	nau::render::Texture* t = RESOURCEMANAGER->getTextureByID(m_UIntProps[TEX_ID]);
	RENDERER->addImageTexture(m_IntProps[UNIT], this);
#if NAU_OPENGL_VERSION >= 440
	if (m_BoolProps[CLEAR]) {
		t->clearLevel(m_UIntProps[LEVEL]);
		//glClearTexImage(m_UIntProps[TEX_ID], m_UIntProps[LEVEL], m_Format, m_Type, NULL);
	}
	glBindImageTexture(m_IntProps[UNIT], m_UIntProps[TEX_ID], m_UIntProps[LEVEL],GL_TRUE,0,m_EnumProps[ACCESS],m_InternalFormat);
#endif
}


void 
GLImageTexture::restore() {

	glBindImageTexture(m_IntProps[UNIT], 0, m_UIntProps[LEVEL], false, 0, m_EnumProps[ACCESS], m_InternalFormat);
	RENDERER->removeImageTexture(m_IntProps[UNIT]);
}



#endif