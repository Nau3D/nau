#include "nau/material/textureSampler.h"
#include "nau.h"
#include "nau/config.h"

#include <map>

using namespace nau::material;
using namespace nau::render;

AttribSet TextureSampler::Attribs;
bool TextureSampler::Inited = TextureSampler::Init();

bool
TextureSampler::Init() {

	// INT
	Attribs.add(Attribute(ID, "ID", Enums::DataType::INT, true, new int(0)));
	// BOOL
	Attribs.add(Attribute(MIPMAP, "MIPMAP", Enums::DataType::BOOL, true, new bool(false)));
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


TextureSampler*
TextureSampler::create(Texture *t) {

#ifdef NAU_OPENGL
	return new GLTextureSampler(t) ;
#elif NAU_DIRECTX
	return new DXTextureSampler;
#endif
}


TextureSampler::TextureSampler() {

	registerAndInitArrays(Attribs);
}


//int 
//TextureSampler::getPrope(EnumProperty prop) {
//
//	return m_EnumProps[prop];
//}
//
//
//int 
//TextureSampler::getPropi(IntProperty prop) {
//
//	return m_IntProps[prop];
//}
//
//
//const vec4& 
//TextureSampler::getPropf4(Float4Property prop) {
//
//	return m_Float4Props[prop];
//}
//
//
//void *
//TextureSampler::getProp(int prop, Enums::DataType type) {
//
//	switch (type) {
//
//		case Enums::FLOAT:
//			assert(m_FloatProps.count(prop) > 0);
//			return(&(m_FloatProps[prop]));
//			break;
//		case Enums::VEC4:
//			assert(m_Float4Props.count(prop) > 0);
//			return(&(m_Float4Props[prop]));
//			break;
//		case Enums::INT:
//			assert(m_IntProps.count(prop) > 0);
//			return(&(m_IntProps[prop]));
//			break;
//		case Enums::ENUM:
//			assert(m_EnumProps.count(prop) > 0);
//			return(&(m_EnumProps[prop]));
//			break;
//
//	}
//	return NULL;
//}
//
//
//void 
//TextureSampler::setProp(int prop, Enums::DataType type, void *value) {
//
//	switch (type) {
//
//		case Enums::FLOAT:
//			m_FloatProps[prop] = *(float *)value;
//			break;
//		case Enums::VEC4:
//			if (prop < COUNT_FLOAT4PROPERTY)
//				setProp((Float4Property)prop, *(vec4 *)value);
//			else
//				m_Float4Props[prop].set((vec4 *)value);
//			break;
//		case Enums::INT:
//			m_IntProps[prop] = *(int *)value;
//			break;
//		case Enums::ENUM:
//			assert(prop < COUNT_ENUMPROPERTY);
//			int m = *(int *)value;
//			setProp((EnumProperty)prop,m);
//			break;
//	}
//}		
