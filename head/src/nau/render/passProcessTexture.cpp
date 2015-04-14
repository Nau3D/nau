#include "nau/render/passProcessTexture.h"

#include "nau.h"

using namespace nau::render;

bool
PassProcessTexture::Init() {

	// BOOL
	Attribs.add(Attribute(CLEAR, "CLEAR", Enums::DataType::BOOL, false, new bool(false)));
	Attribs.add(Attribute(MIPMAP, "MIPMAP", Enums::DataType::BOOL, false, new bool(false)));

	// INT
	Attribs.add(Attribute(CLEAR_LEVEL, "CLEAR_LEVEL", Enums::DataType::INT, false, new int(-1)));

	NAU->registerAttributes("PASS_PROCESS_TEXTURE", &Attribs);
	return true;
}


AttribSet PassProcessTexture::Attribs;
bool PassProcessTexture::Inited = Init();


PassProcessTexture::PassProcessTexture() : m_Tex(NULL), PassProcessItem() {

	registerAndInitArrays(Attribs);
}


void 
PassProcessTexture::setItem(Texture *tex) {

	m_Tex = tex;
}


void 
PassProcessTexture::process() {

	if (!m_Tex)
		return;

#if NAU_OPENGL_VERSION >= 440
	if (m_BoolProps[CLEAR])
		m_Tex->clear();
	if (m_IntProps[CLEAR_LEVEL] >= 0)
		m_Tex->clearLevel(m_IntProps[CLEAR_LEVEL]);
#endif
	if (m_BoolProps[MIPMAP])
		m_Tex->generateMipmaps();
}