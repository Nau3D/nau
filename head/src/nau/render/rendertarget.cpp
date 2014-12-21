#include <nau/render/rendertarget.h>

#include <nau/config.h>

#include <nau/render/opengl/glrendertarget.h>

using namespace nau::render;

RenderTarget* 
RenderTarget::Create (std::string name, unsigned int width, unsigned int height)
{
#ifdef NAU_OPENGL
	return new GLRenderTarget (name, width, height);
#elif NAU_DIRECTX
	return new DXRenderTarget (width, height);
#endif
}

RenderTarget* 
RenderTarget::Create (std::string name)
{
#ifdef NAU_OPENGL
	return new GLRenderTarget (name);
#elif NAU_DIRECTX
	return new DXRenderTarget ();
#endif
}

RenderTarget::RenderTarget (std::string name, unsigned int width, unsigned int height) :
	m_Id (0),
	m_Color(0),
	m_Depth(0),
	m_Stencil(0),
	m_Name (name),
	m_Width (width),
	m_Height (height),
	m_Samples(0),
	m_Layers(0),
	m_TexId(0),
	m_DepthTexture(NULL),
	m_StencilTexture(NULL)

{
}


void
RenderTarget::setClearValues(float r, float g, float b, float a) 
{
	m_ClearValues.x = r;
	m_ClearValues.y = g;
	m_ClearValues.z = b;
	m_ClearValues.w = 0;
}


void 
RenderTarget::setSampleCount(int samples) 
{
	m_Samples = samples;
}


void 
RenderTarget::setLayerCount(int layers) 
{
	m_Layers = layers;
}


nau::math::vec4 &
RenderTarget::getClearValues() 
{
	return m_ClearValues;
}


int
RenderTarget::getId (void)
{
	return m_Id;
}


std::string &
RenderTarget::getName (void)
{
	return m_Name;
}


unsigned int 
RenderTarget::getWidth (void)
{
	return m_Width;
}


unsigned int 
RenderTarget::getHeight (void) 
{
	return m_Height;
}


unsigned int 
RenderTarget::getNumberOfColorTargets() 
{
	return m_Color;
}


nau::render::Texture *
RenderTarget::getTexture(unsigned int i) 
{
	if (i <= m_TexId.size()) {
		return m_TexId[i];
	} 
	else return 0;
}