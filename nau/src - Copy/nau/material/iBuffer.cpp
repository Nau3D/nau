#include "nau/material/iBuffer.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glBuffer.h"
#endif

#include "nau.h"

using namespace nau::render;
using namespace nau;


bool
IBuffer::Init() {

	// UINT
	Attribs.add(Attribute(SIZE, "SIZE", Enums::DataType::UINT, false, new NauUInt(0)));
	Attribs.add(Attribute(STRUCT_SIZE, "STRUCT_SIZE", Enums::DataType::UINT, true, new NauUInt(0)));
	// INT
	Attribs.add(Attribute(ID, "ID", Enums::DataType::INT, true, new NauInt(-1)));
	// ENUM
	Attribs.add(Attribute(CLEAR, "CLEAR", Enums::DataType::ENUM, false, new NauInt(NEVER)));
	Attribs.setDefault("CLEAR", NauInt(NEVER));
	Attribs.listAdd("CLEAR", "NEVER", NEVER);
	Attribs.listAdd("CLEAR", "BY_FRAME", BY_FRAME);
	// UIVEC3
	Attribs.add(Attribute(DIM, "DIM", Enums::DataType::UIVEC3, false, new uivec3(1)));

#ifndef _WINDLL
	// for direct access
	NAU->registerAttributes("BUFFER", &Attribs);
	// for access through material
	NAU->registerAttributes("BUFFER_MATERIAL", &Attribs);
#endif
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


void
IBuffer::setStructure(std::vector<Enums::DataType> s) {

	m_Structure = s;
	int sts = 0;
	for (auto t : m_Structure) {
		sts += Enums::getSize(t);
	}
	setPropui(STRUCT_SIZE, sts);

}


std::vector<Enums::DataType> &
IBuffer::getStructure() {

	return m_Structure;
}


void
IBuffer::appendItemToStruct(Enums::DataType dt) {

	int card = Enums::getCardinality(dt);
	Enums::DataType bdt = Enums::getBasicType(dt);

	for (int i = 0; i < card; ++i)
		m_Structure.push_back(bdt);
}