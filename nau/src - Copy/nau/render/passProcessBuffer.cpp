#include "nau/render/passProcessBuffer.h"

#include "nau.h"

using namespace nau::render;

bool
PassProcessBuffer::Init() {

	// BOOL
	Attribs.add(Attribute(CLEAR, "CLEAR", Enums::DataType::BOOL, false, new NauInt(false)));


#ifndef _WINDLL
	NAU->registerAttributes("PASS_PRE_PROCESS_BUFFER", &Attribs);
	NAU->registerAttributes("PASS_POST_PROCESS_BUFFER", &Attribs);
#endif
	return true;
}

AttribSet PassProcessBuffer::Attribs;
bool PassProcessBuffer::Inited = Init();


PassProcessBuffer::PassProcessBuffer() : m_Buffer(NULL), PassProcessItem() {

	registerAndInitArrays(Attribs);
}


void 
PassProcessBuffer::setItem(IBuffer *buf) {

	m_Buffer = buf;
}


void 
PassProcessBuffer::process() {

	if (!m_Buffer)
		return;

	if (m_BoolProps[CLEAR])
		m_Buffer->clear();
}