#include "nau/material/iArrayOfTextures.h"

#include "nau.h"
#include "nau/material/iTexture.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glArrayOfTextures.h"
#endif


using namespace nau;
using namespace nau::material;


bool
IArrayOfTextures::Init() {

	// INT
	Attribs.add(Attribute(WIDTH, "WIDTH", Enums::DataType::INT, false, new NauInt(1)));
	Attribs.add(Attribute(HEIGHT, "HEIGHT", Enums::DataType::INT, false, new NauInt(1)));
	Attribs.add(Attribute(DEPTH, "DEPTH", Enums::DataType::INT, false, new NauInt(1)));
	Attribs.add(Attribute(SAMPLES, "SAMPLES", Enums::DataType::INT, false, new NauInt(0)));
	Attribs.add(Attribute(LEVELS, "LEVELS", Enums::DataType::INT, false, new NauInt(1)));
	Attribs.add(Attribute(LAYERS, "LAYERS", Enums::DataType::INT, false, new NauInt(1)));
	Attribs.add(Attribute(COMPONENT_COUNT, "COMPONENT_COUNT", Enums::DataType::INT, true, new NauInt(0)));
	Attribs.add(Attribute(ELEMENT_SIZE, "ELEMENT_SIZE", Enums::DataType::INT, true, new NauInt(0)));
	Attribs.add(Attribute(BUFFER_ID, "BUFFER_ID", Enums::DataType::INT, true, new NauInt(0)));
	// UINT
	Attribs.add(Attribute(TEXTURE_COUNT, "TEXTURE_COUNT", Enums::DataType::UINT, false, new NauUInt(1)));
	// BOOL
	Attribs.add(Attribute(MIPMAP, "MIPMAP", Enums::DataType::BOOL, false, new NauInt(false)));
	Attribs.add(Attribute(CREATE_BUFFER, "CREATE_BUFFER", Enums::DataType::BOOL, false, new NauInt(false), NULL, NULL, IAPISupport::BINDLESS_TEXTURES));
	// ENUM
	Attribs.add(Attribute(DIMENSION, "DIMENSION", Enums::DataType::ENUM, true));
	Attribs.add(Attribute(FORMAT, "FORMAT", Enums::DataType::ENUM, true));
	Attribs.add(Attribute(TYPE, "TYPE", Enums::DataType::ENUM, true));
	Attribs.add(Attribute(INTERNAL_FORMAT, "INTERNAL_FORMAT", Enums::DataType::ENUM, false));
	// INTARRAY
	Attribs.add(Attribute(TEXTURE_ID_ARRAY, "TEXTURE_ID_ARRAY", Enums::DataType::INTARRAY, true));
#ifndef _WINDLL
	// for direct access
	NAU->registerAttributes("ARRAY_OF_TEXTURES", &Attribs);
#endif

	return true;
}


AttribSet IArrayOfTextures::Attribs;
bool IArrayOfTextures::Inited = Init();


IArrayOfTextures*
IArrayOfTextures::Create(const std::string &label) {

#ifdef NAU_OPENGL
	return new GLArrayOfTextures(label);
#elif NAU_DIRECTX
	//Put here function for DirectX
#endif
}


IArrayOfTextures::IArrayOfTextures(const std::string &label) :
	m_Label(label), m_Buffer(NULL) {

	registerAndInitArrays(Attribs);

}


const std::string &
IArrayOfTextures::getLabel() {

	return m_Label;
}


void
IArrayOfTextures::setLabel(const std::string &label) {

	m_Label = label;
}


ITexture *
IArrayOfTextures::getTexture(unsigned int i) {

	if (i < m_Textures.size())
		return m_Textures[i];
	else
		return NULL;
}