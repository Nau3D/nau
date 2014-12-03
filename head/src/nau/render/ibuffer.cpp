#include <nau/render/ibuffer.h>

#if NAU_OPENGL_VERSION >= 430

#ifdef NAU_OPENGL
#include <nau/render/opengl/glbuffer.h>
#endif

using namespace nau::render;
using namespace nau;


bool
IBuffer::Init() {

	// BOOL
	//Attribs.add(Attribute(BOUND, "BOUND", Enums::DataType::BOOL, true, new bool(false)));
	//Attribs.add(Attribute(CLEAR, "CLEAR", Enums::DataType::BOOL, false, new bool(false)));
	// UINT
	Attribs.add(Attribute(SIZE, "SIZE", Enums::DataType::UINT, false, new unsigned int(0)));
	// INT
	Attribs.add(Attribute(ID, "ID", Enums::DataType::INT, true, new int(-1)));
	//Attribs.add(Attribute(BINDING_POINT, "BINDING_POINT", Enums::DataType::INT, false, new int(0)));
	// ENUM
	//Attribs.add(Attribute(TYPE, "TYPE", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(CLEAR, "CLEAR", Enums::DataType::ENUM, false, new int(NEVER)));
	Attribs.setDefault("CLEAR", new int(NEVER));
	Attribs.listAdd("CLEAR", "NEVER", NEVER);
	Attribs.listAdd("CLEAR", "BY_FRAME", BY_FRAME);


	return true;
}


AttribSet IBuffer::Attribs;
bool IBuffer::Inited = Init();


IBuffer*
IBuffer::Create(std::string label)
{
#ifdef NAU_OPENGL
	return new GLBuffer (label);
#elif NAU_DIRECTX
	//Meter função para DirectX
#endif
}


std::string &
IBuffer::getLabel() {

	return m_Label;
}

#endif
	

