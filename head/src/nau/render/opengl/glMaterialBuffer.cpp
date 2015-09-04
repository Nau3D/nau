#include "nau/render/opengl/glMaterialBuffer.h"

#include "nau/render/iAPISupport.h"

#include <GL/glew.h>

using namespace nau::render;
using namespace nau::material;

bool
GLMaterialBuffer::Init() {

	//Attribs.setDefault("TYPE", new int(GL_ATOMIC_COUNTER_BUFFER));

	Attribs.listAdd("TYPE", "ATOMIC_COUNTER", GL_ATOMIC_COUNTER_BUFFER, IAPISupport::BUFFER_ATOMICS);
	Attribs.listAdd("TYPE", "SHADER_STORAGE", GL_SHADER_STORAGE_BUFFER, IAPISupport::BUFFER_SHADER_STORAGE);

	return true;
}

bool GLMaterialBuffer::Inited = Init();

GLMaterialBuffer::GLMaterialBuffer(): IMaterialBuffer() {

}


GLMaterialBuffer::~GLMaterialBuffer() {

}

void
GLMaterialBuffer::bind() {

	int id = m_Buffer->getPropi(IBuffer::ID);
	//if (m_BoolProps[CLEAR] == true) {
	//	m_Buffer->clear();
	//}
	//glBindBuffer(m_EnumProps[TYPE], id);
	glBindBufferBase(m_EnumProps[TYPE], m_IntProps[BINDING_POINT], id);


}


void
GLMaterialBuffer::unbind() {

	int id = m_Buffer->getPropi(IBuffer::ID);
	glBindBufferBase(m_EnumProps[TYPE], m_IntProps[BINDING_POINT], 0);
	//glBindBuffer(m_EnumProps[TYPE], 0);
}
