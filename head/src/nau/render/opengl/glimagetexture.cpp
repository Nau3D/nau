#include <nau/render/opengl/glimagetexture.h>

#if NAU_OPENGL_VERSION >=  420

#include <nau.h>
#include <nau/render/irenderer.h>

using namespace nau::render;

bool GLImageTexture::Inited = GLImageTexture::InitGL();


bool
GLImageTexture::InitGL() {

	Attribs.listAdd("ACCESS", "READ_ONLY", GL_READ_ONLY);
	Attribs.listAdd("ACCESS", "WRITE_ONLY", GL_WRITE_ONLY);
	Attribs.listAdd("ACCESS", "READ_WRITE", GL_READ_WRITE);

	return(true);
};


GLImageTexture::GLImageTexture(std::string label, unsigned int texID, unsigned int level, unsigned int access)
{
	m_EnumProps[ACCESS] = access;
	m_UIntProps[LEVEL] = level;
	m_UIntProps[TEX_ID] = texID;
	m_InternalFormat = RESOURCEMANAGER->getTexture(texID)->getPrope(Texture::INTERNAL_FORMAT);
	m_Label = label;
	m_Unit = 0;
}


GLImageTexture::~GLImageTexture(void)
{
}


void 
GLImageTexture::prepare(int aUnit) {

	m_Unit = aUnit;
	glBindImageTexture(aUnit, m_UIntProps[TEX_ID], m_UIntProps[LEVEL],false,0,m_EnumProps[ACCESS],m_InternalFormat);	
	RENDERER->addImageTexture(m_Unit, this);
}


void 
GLImageTexture::restore() {

	RENDERER->removeImageTexture(m_Unit);
	m_Unit = 0;
}



#endif