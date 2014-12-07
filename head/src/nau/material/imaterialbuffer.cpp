#include <nau/material/imaterialbuffer.h>

#ifdef NAU_OPENGL
#include <nau/render/opengl/glmaterialbuffer.h>
#endif

using namespace nau::material;

bool
IMaterialBuffer::Init() {

	// INT
	Attribs.add(Attribute(BINDING_POINT, "BINDING_POINT", Enums::DataType::INT, false, new int(-1)));
	// ENUM
	Attribs.add(Attribute(TYPE, "TYPE", Enums::DataType::ENUM, false));

	return true;
}


AttribSet IMaterialBuffer::Attribs;
bool IMaterialBuffer::Inited = Init();


IMaterialBuffer*
IMaterialBuffer::Create(nau::render::IBuffer *b)
{

#ifdef NAU_OPENGL
	IMaterialBuffer *imb = (IMaterialBuffer *)(new nau::render::GLMaterialBuffer());
#endif

	imb->setBuffer(b);

	return imb;
}


void
IMaterialBuffer::setBuffer(nau::render::IBuffer *b) {

	m_Buffer = b;
}
