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
	Attribs.add(Attribute(LEVELS, "LEVELS", Enums::DataType::UINT, false, new unsigned int(1)));
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
	
	registerAndInitArrays(Attribs);
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

