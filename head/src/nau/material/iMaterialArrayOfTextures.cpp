#include "nau/material/iMaterialArrayOfTextures.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glMaterialArrayOfTextures.h"
#endif

#include "nau.h"

using namespace nau::material;

bool
IMaterialArrayOfTextures::Init() {

	// INT
	Attribs.add(Attribute(BINDING_POINT, "BINDING_POINT", Enums::DataType::INT, false, new NauInt(-1)));

#ifndef _WINDLL
	NAU->registerAttributes("MATERIAL_ARRAY_OF_TEXTURES", &Attribs);
#endif

	return true;
}


AttribSet IMaterialArrayOfTextures::Attribs;
bool IMaterialArrayOfTextures::Inited = Init();


IMaterialArrayOfTextures*
IMaterialArrayOfTextures::Create(nau::material::IArrayOfTextures *ta)
{

#ifdef NAU_OPENGL
	IMaterialArrayOfTextures *mta = (IMaterialArrayOfTextures *)(new nau::render::GLMaterialArrayOfTextures());
#endif

	mta->setTextureArray(ta);

	return mta;
}


void
IMaterialArrayOfTextures::setTextureArray(IArrayOfTextures *ta) {

	m_TextureArray = ta;
}


IArrayOfTextures *
IMaterialArrayOfTextures::getTextureArray() {

	return m_TextureArray;
}