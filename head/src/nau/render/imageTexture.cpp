#include <nau/render/imageTexture.h>

#if NAU_OPENGL_VERSION >= 420


#ifdef NAU_OPENGL
#include <nau/render/opengl/glimagetexture.h>
#endif

using namespace nau::render;
using namespace nau;


bool
ImageTexture::Init() {

	// UINT
	Attribs.add(Attribute(TEX_ID, "TEX_ID", Enums::DataType::UINT, false, new int(0)));
	Attribs.add(Attribute(LEVEL, "LEVEL", Enums::DataType::UINT, false, new int(0)));
	// ENUM
	Attribs.add(Attribute(ACCESS, "ACCESS", Enums::DataType::ENUM, false));
	// BOOL
	Attribs.add(Attribute(CLEAR, "CLEAR", Enums::DataType::BOOL, false, new bool(false)));
	return true;
}


AttribSet ImageTexture::Attribs;
bool ImageTexture::Inited = Init();




ImageTexture*
ImageTexture::Create (std::string label, unsigned int texID, unsigned int level, unsigned int access)
{
#ifdef NAU_OPENGL
	return new GLImageTexture (label, texID, level, access);
#elif NAU_DIRECTX
	//Meter função para DirectX
#endif
}


ImageTexture*
ImageTexture::Create (std::string label, unsigned int texID)
{
#ifdef NAU_OPENGL
	return new GLImageTexture (label, texID);
#elif NAU_DIRECTX
	//Meter função para DirectX
#endif
}


void
ImageTexture::initArrays() {

	Attribs.initAttribInstanceUIntArray(m_UIntProps);
	Attribs.initAttribInstanceEnumArray(m_EnumProps);
	Attribs.initAttribInstanceBoolArray(m_BoolProps);
}


void *
ImageTexture::getProp(int prop, Enums::DataType type) {

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
	case Enums::UINT:
		assert(m_UIntProps.count(prop) > 0);
		return(&(m_UIntProps[prop]));
		break;
	case Enums::BOOL:
		assert(m_BoolProps.count(prop) > 0);
		return(&(m_BoolProps[prop]));
		break;

		
	}
	return NULL;
}


void 
ImageTexture::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {

		case Enums::FLOAT:
			m_FloatProps[prop] = *(float *)value;
			break;
		case Enums::VEC4:
			m_Float4Props[prop].set((vec4 *)value);
			break;
		case Enums::INT:
			m_IntProps[prop] = *(int *)value;
			break;
		case Enums::UINT:
			m_UIntProps[prop] = *(unsigned int *)value;
			break;
		case Enums::ENUM:
			m_EnumProps[prop] = *(int *)value;
			break;
		case Enums::BOOL:
			m_BoolProps[prop] = *(bool *)value;
			break;

	}
}		


int 
ImageTexture::getPropi(IntProperty prop)
{
	assert(m_IntProps.find(prop) != m_IntProps.end());
	return m_IntProps[prop];
}



unsigned int
ImageTexture::getPropui(UIntProperty prop) 
{
	assert(m_UIntProps.find(prop) != m_UIntProps.end());
	return(m_UIntProps[prop]);
}


std::string&
ImageTexture::getLabel (void)
{
	return m_Label;
}


void
ImageTexture::setLabel (std::string label)
{
	m_Label = label;
}


#endif