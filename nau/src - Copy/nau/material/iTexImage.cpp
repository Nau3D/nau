#include "nau/material/iTexImage.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glTexImage.h"
#endif

using namespace nau::material;


ITexImage *
ITexImage::create(ITexture *t) 
{
	#ifdef NAU_OPENGL
		return new GLTexImage(t);
	#elif NAU_DIRECTX
		return new DXTexImage(t);
	#endif

}


ITexImage::ITexImage(ITexture *t) 
{
	m_Texture = t;
	m_NumComponents = t->getPropi(ITexture::COMPONENT_COUNT);
	m_Width = t->getPropi(ITexture::WIDTH);
	m_Height = t->getPropi(ITexture::HEIGHT);
	m_Depth = t->getPropi(ITexture::DEPTH);
}


ITexImage::~ITexImage() 
{
	if (m_Data)
		delete(m_Data);
}


const std::string &
ITexImage::getTextureName() 
{
	return m_Texture->getLabel();
}


std::string 
ITexImage::getType() 
{
	return m_Texture->Attribs.getListStringOp(ITexture::TYPE, m_Texture->getPrope(ITexture::TYPE));//(getStringType();
}


int 
ITexImage::getWidth()
{
	return m_Texture->getPropi(ITexture::WIDTH);
}


int 
ITexImage::getHeight()
{
	return m_Texture->getPropi(ITexture::HEIGHT);
}


int 
ITexImage::getDepth()
{
	return m_Texture->getPropi(ITexture::DEPTH);
}


int 
ITexImage::getNumComponents() 
{
	return m_NumComponents;
}
