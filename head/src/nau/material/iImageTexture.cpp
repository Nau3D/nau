#include "nau/material/iImageTexture.h"


#ifdef NAU_OPENGL
#include "nau/render/opengl/glImageTexture.h"
#endif

#include "nau.h"

using namespace nau::render;
using namespace nau;


bool
IImageTexture::Init() {

	// UINT
	Attribs.add(Attribute(TEX_ID, "TEX_ID", Enums::DataType::UINT, false, new int(0)));
	Attribs.add(Attribute(LEVEL, "LEVEL", Enums::DataType::UINT, false, new int(0)));
	// ENUM
	Attribs.add(Attribute(ACCESS, "ACCESS", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(INTERNAL_FORMAT, "INTERNAL_FORMAT", Enums::DataType::ENUM, false));
	// BOOL
	Attribs.add(Attribute(CLEAR, "CLEAR", Enums::DataType::BOOL, false, new bool(false)));
	// INT
	Attribs.add(Attribute(UNIT, "UNIT", Enums::DataType::INT, true, new int(-1)));

#ifndef _WINDLL
	NAU->registerAttributes("IMAGE_TEXTURE", &Attribs);
#endif

	return true;
}


AttribSet IImageTexture::Attribs;
bool IImageTexture::Inited = Init();




IImageTexture*
IImageTexture::Create(std::string label, unsigned int unit, unsigned int texID, unsigned int level, unsigned int access) {

#ifdef NAU_OPENGL
	return new GLImageTexture (label, unit, texID, level, access);
#elif NAU_DIRECTX
	//Meter função para DirectX
#endif
}


IImageTexture*
IImageTexture::Create(std::string label, unsigned int unit, unsigned int texID) {

#ifdef NAU_OPENGL
	return new GLImageTexture (label, unit, texID);
#elif NAU_DIRECTX
	//Meter função para DirectX
#endif
}


IImageTexture::IImageTexture() {

	registerAndInitArrays(Attribs);
}


std::string&
IImageTexture::getLabel (void) {

	return m_Label;
}


void
IImageTexture::setLabel (std::string label) {

	m_Label = label;
}
