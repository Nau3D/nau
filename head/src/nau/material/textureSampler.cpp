#include <nau/material/textureSampler.h>
#include <nau/render/opengl/gltexturesampler.h>
#include <nau/config.h>
#include <map>

using namespace nau::material;
using namespace nau::render;

AttribSet TextureSampler::Attribs;

TextureSampler*
TextureSampler::create(Texture *t) {

#ifdef NAU_OPENGL
	return new GLTextureSampler(t) ;
#elif NAU_DIRECTX
	return new DXTextureSampler;
#endif
}


TextureSampler::TextureSampler() {}


int 
TextureSampler::getPrope(EnumProperty prop) {

	return m_EnumProps[prop];
}


unsigned int 
TextureSampler::getPropui(UIntProperty prop) {

	return m_UIntProps[prop];
}


const vec4& 
TextureSampler::getPropf4(Float4Property prop) {

	return m_Float4Props[prop];
}


void *
TextureSampler::getProp(int prop, Enums::DataType type) {

	switch (type) {

		case Enums::FLOAT:
			assert(m_FloatProps.count(prop) > 0);
			return(&(m_FloatProps[prop]));
			break;
		case Enums::VEC4:
			assert(m_Float4Props.count(prop) > 0);
			return(&(m_Float4Props[prop]));
			break;
		case Enums::INT:
			assert(m_IntProps.count(prop) > 0);
			return(&(m_IntProps[prop]));
			break;
		case Enums::ENUM:
			assert(m_EnumProps.count(prop) > 0);
			return(&(m_EnumProps[prop]));
			break;

	}
	return NULL;
}


void 
TextureSampler::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {

		case Enums::FLOAT:
			m_FloatProps[prop] = *(float *)value;
			break;
		case Enums::VEC4:
			if (prop < COUNT_FLOAT4PROPERTY)
				setProp((Float4Property)prop, *(vec4 *)value);
			else
				m_Float4Props[prop].set((vec4 *)value);
			break;
		case Enums::INT:
			m_IntProps[prop] = *(int *)value;
			break;
		case Enums::ENUM:
			assert(prop < COUNT_ENUMPROPERTY);
			int m = *(int *)value;
			setProp((EnumProperty)prop,m);
			break;
	}
}		
