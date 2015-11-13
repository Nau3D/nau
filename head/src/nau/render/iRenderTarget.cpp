#include "nau/render/iRenderTarget.h"

#include "nau.h"
#include "nau/config.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glRenderTarget.h"
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
IRenderTarget::Init() {

	// UINT
	Attribs.add(Attribute(SAMPLES, "SAMPLES", Enums::DataType::UINT, false, new NauUInt(0)));
	Attribs.add(Attribute(LAYERS, "LAYERS", Enums::DataType::UINT, false, new NauUInt(1)));
	Attribs.add(Attribute(LEVELS, "LEVELS", Enums::DataType::UINT, false, new NauUInt(0)));
	// UINT2
	Attribs.add(Attribute(SIZE, "SIZE", Enums::DataType::UIVEC2, false, new uivec2(0,0)));
	// VEC4
	//Attribute a = Attribute(CLEAR_VALUES, "CLEAR_VALUES", Enums::DataType::VEC4, false, new vec4(0), new vec4(0), new vec4(1));
	Attribs.add(Attribute(CLEAR_VALUES, "CLEAR_VALUES", Enums::DataType::VEC4, false, new vec4(0.0f), new vec4(0.0f), new vec4(1)));

#ifndef _WINDLL
	NAU->registerAttributes("RENDER_TARGET", &Attribs);
#endif

	return true;
}


AttribSet IRenderTarget::Attribs;
bool IRenderTarget::Inited = Init();



IRenderTarget*
IRenderTarget::Create (std::string name) {

#ifdef NAU_OPENGL
	return new GLRenderTarget (name);
#elif NAU_DIRECTX
	return new DXRenderTarget ();
#endif
}


IRenderTarget::IRenderTarget () {
	
	registerAndInitArrays(Attribs);
	m_Color = 0;
	m_Depth = 0;
	m_Stencil = 0;
	m_DepthTexture = NULL;
	m_StencilTexture = NULL;
	m_Id = 0;
}


int
IRenderTarget::getId (void) {

	return m_Id;
}


std::string &
IRenderTarget::getName (void) {

	return m_Name;
}


unsigned int 
IRenderTarget::getNumberOfColorTargets() {

	return m_Color;
}


ITexture *
IRenderTarget::getTexture(unsigned int i) {

	if (i <= m_TexId.size()) {
		return m_TexId[i];
	} 
	else return 0;
}


ITexture * 
IRenderTarget::getDepthTexture() {

	return m_DepthTexture;
}


ITexture *
IRenderTarget::getStencilTexture() {

	return m_StencilTexture;
}

