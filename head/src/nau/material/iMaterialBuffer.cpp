#include "nau/material/iMaterialBuffer.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glMaterialBuffer.h"
#endif

#include "nau.h"

using namespace nau::material;

bool
IMaterialBuffer::Init() {

	// INT
	Attribs.add(Attribute(BINDING_POINT, "BINDING_POINT", Enums::DataType::INT, false, new NauInt(-1)));
	// ENUM
	Attribs.add(Attribute(TYPE, "TYPE", Enums::DataType::ENUM, false));

#ifndef _WINDLL
	NAU->registerAttributes("BUFFER_BINDING", &Attribs);
#endif

	return true;
}


AttribSet IMaterialBuffer::Attribs;
bool IMaterialBuffer::Inited = Init();


IMaterialBuffer*
IMaterialBuffer::Create(nau::material::IBuffer *b)
{

#ifdef NAU_OPENGL
	IMaterialBuffer *imb = (IMaterialBuffer *)(new nau::render::GLMaterialBuffer());
#endif

	imb->setBuffer(b);

	return imb;
}


void
IMaterialBuffer::setBuffer(nau::material::IBuffer *b) {

	m_Buffer = b;
}


IBuffer *
IMaterialBuffer::getBuffer() {

	return m_Buffer;
}