#include <nau/render/buffer.h>

#ifdef NAU_OPENGL
#include <nau/render/opengl/glbuffer.h>
#endif

using namespace nau::render;
using namespace nau;


bool
Buffer::Init() {

	// UINT
	Attribs.add(Attribute(SIZE, "SIZE", Enums::DataType::UINT, true, new unsigned int(0));
	// INT
	Attribs.add(Attribute(ID, "ID", Enums::DataType::INT, true, new int(-1)));
	Attribs.add(Attribute(BINDING_POINT, "BINDING_POINT", Enums::DataType::INT, true, new int(-1)));

	return true;
}


AttribSet Buffer::Attribs;
bool Buffer::Inited = Init();


// For loaded images
Buffer*
Buffer::Create(std::string label, int size)
{
#ifdef NAU_OPENGL
	return new GLBuffer (label, size);
#elif NAU_DIRECTX
	//Meter função para DirectX
#endif
}


void *
Buffer::getProp(int prop, Enums::DataType type) {

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
		
	}
	return NULL;
}


void 
Buffer::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {

		case Enums::FLOAT:
			m_FloatProps[prop] = *(float *)value;
			break;
		case Enums::VEC4:
			m_Float4Props[prop].set((vec4 *)value);
			break;
		case Enums::INT:
			if (prop >= COUNT_INTPROPERTY)
				m_IntProps[prop] = *(int *)value;
			break;
	}
}		

