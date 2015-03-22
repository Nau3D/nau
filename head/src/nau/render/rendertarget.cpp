#include "nau/render/rendertarget.h"

#include "nau.h"
#include "nau/config.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glrendertarget.h"
#endif

using namespace nau::render;

//RenderTarget* 
//RenderTarget::Create (std::string name, unsigned int width, unsigned int height)
//{
//#ifdef NAU_OPENGL
//	return new GLRenderTarget (name, width, height);
//#elif NAU_DIRECTX
//	return new DXRenderTarget (width, height);
//#endif
//}

bool
RenderTarget::Init() {

	// UINT
	Attribs.add(Attribute(SAMPLES, "SAMPLES", Enums::DataType::UINT, false, new unsigned int(0)));
	Attribs.add(Attribute(LAYERS, "LAYERS", Enums::DataType::UINT, false, new unsigned int(0)));
	// UINT2
	Attribs.add(Attribute(SIZE, "SIZE", Enums::DataType::UIVEC2, false, new uivec2(0)));
	// VEC4
	Attribute a = Attribute(CLEAR_VALUES, "CLEAR_VALUES", Enums::DataType::VEC4, false, new vec4(0), new vec4(0), new vec4(1));
	Attribs.add(a);

	NAU->registerAttributes("RENDER_TARGET", &Attribs);

	return true;
}


AttribSet RenderTarget::Attribs;
bool RenderTarget::Inited = Init();



RenderTarget* 
RenderTarget::Create (std::string name) {

#ifdef NAU_OPENGL
	return new GLRenderTarget (name);
#elif NAU_DIRECTX
	return new DXRenderTarget ();
#endif
}


RenderTarget::RenderTarget () {
	
	registerAndInitArrays("RENDER_TARGET", Attribs);
	m_Color = 0;
	m_Depth = 0;
	m_Stencil = 0;
	m_Id = 0;
}


int
RenderTarget::getId (void) {

	return m_Id;
}


std::string &
RenderTarget::getName (void) {

	return m_Name;
}


unsigned int 
RenderTarget::getNumberOfColorTargets() {

	return m_Color;
}


nau::render::Texture *
RenderTarget::getTexture(unsigned int i) {

	if (i <= m_TexId.size()) {
		return m_TexId[i];
	} 
	else return 0;
}




//RenderTarget::RenderTarget (std::string name, unsigned int width, unsigned int height) :
//	m_Id (0),
//	m_Color(0),
//	m_Depth(0),
//	m_Stencil(0),
//	m_Name (name),
//	//m_Width (width),
//	//m_Height (height),
//	//m_Samples(0),
//	//m_Layers(0),
//	m_TexId(0),
//	m_DepthTexture(NULL),
//	m_StencilTexture(NULL)
//
//{
//}


//void
//RenderTarget::setClearValues(float r, float g, float b, float a) 
//{
//	m_ClearValues.x = r;
//	m_ClearValues.y = g;
//	m_ClearValues.z = b;
//	m_ClearValues.w = 0;
//}


//void 
//RenderTarget::setSampleCount(int samples) 
//{
//	m_Samples = samples;
//}
//
//
//void 
//RenderTarget::setLayerCount(int layers) 
//{
//	m_Layers = layers;
//}
//
//
//nau::math::vec4 &
//RenderTarget::getClearValues() 
//{
//	return m_ClearValues;
//}




//unsigned int 
//RenderTarget::getWidth (void)
//{
//	return m_Width;
//}
//
//
//unsigned int 
//RenderTarget::getHeight (void) 
//{
//	return m_Height;
//}


