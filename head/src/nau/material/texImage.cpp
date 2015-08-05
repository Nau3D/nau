#include "nau/material/texImage.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glTexImage.h"
#endif

using namespace nau::material;


TexImage *
TexImage::create(Texture *t) 
{
	#ifdef NAU_OPENGL
		return new GLTexImage(t);
	#elif NAU_DIRECTX
		return new DXTexImage(t);
	#endif

}


TexImage::TexImage(Texture *t) 
{
	m_Texture = t;
	m_NumComponents = t->getPropi(Texture::COMPONENT_COUNT);
	m_Width = t->getPropi(Texture::WIDTH);
	m_Height = t->getPropi(Texture::HEIGHT);
	m_Depth = t->getPropi(Texture::DEPTH);
}


TexImage::~TexImage() 
{
	if (m_Data)
		delete(m_Data);
}


const std::string &
TexImage::getTextureName() 
{
	return m_Texture->getLabel();
}


std::string 
TexImage::getType() 
{
	return m_Texture->Attribs.getListStringOp(Texture::TYPE, m_Texture->getPrope(Texture::TYPE));//(getStringType();
}


int 
TexImage::getWidth()
{
	return m_Texture->getPropi(Texture::WIDTH);
}


int 
TexImage::getHeight()
{
	return m_Texture->getPropi(Texture::HEIGHT);
}


int 
TexImage::getDepth()
{
	return m_Texture->getPropi(Texture::DEPTH);
}


int 
TexImage::getNumComponents() 
{
	return m_NumComponents;
}
