#include "nau/material/iTextureSampler.h"
#include "nau.h"
#include "nau/config.h"

#include <map>

using namespace nau::material;
using namespace nau::render;

AttribSet ITextureSampler::Attribs;
bool ITextureSampler::Inited = ITextureSampler::Init();

bool
ITextureSampler::Init() {

	// INT
	Attribs.add(Attribute(ID, "ID", Enums::DataType::INT, true, new NauInt(0)));
	// BOOL
	Attribs.add(Attribute(MIPMAP, "MIPMAP", Enums::DataType::BOOL, true, new NauInt(false)));
	// ENUM
	Attribs.add(Attribute(WRAP_S, "WRAP_S", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(WRAP_T, "WRAP_T", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(WRAP_R, "WRAP_R", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(MAG_FILTER, "MAG_FILTER", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(MIN_FILTER, "MIN_FILTER", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(COMPARE_FUNC, "COMPARE_FUNC", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(COMPARE_MODE, "COMPARE_MODE", Enums::DataType::ENUM, false));
	//VEC4
	Attribs.add(Attribute(BORDER_COLOR, "BORDER_COLOR", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 0.0f, 0.0f)));

	return true;
}

#ifdef NAU_OPENGL
#include "nau/render/opengl/gltexturesampler.h"
#endif


ITextureSampler*
ITextureSampler::create(ITexture *t) {

#ifdef NAU_OPENGL
	return new GLTextureSampler(t) ;
#elif NAU_DIRECTX
	return new DXTextureSampler;
#endif
}


ITextureSampler::ITextureSampler() {

	registerAndInitArrays(Attribs);
}


