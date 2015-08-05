#include "nau/render/opengl/glMaterialBuffer.h"

#include <GL/glew.h>

using namespace nau::render;
using namespace nau::material;

bool
GLMaterialBuffer::Init() {

	Attribs.setDefault("TYPE", new int(GL_SHADER_STORAGE_BUFFER));
#if NAU_OPENGL_VERSION >= 420
	Attribs.listAdd("TYPE", "ATOMIC_COUNTER", GL_ATOMIC_COUNTER_BUFFER);
#endif
#if NAU_OPENGL_VERSION >= 430
	Attribs.listAdd("TYPE", "SHADER_STORAGE", GL_SHADER_STORAGE_BUFFER);
#endif	
//#if NAU_OPENGL_VERSION >= 400
//	Attribs.listAdd("TYPE", "DRAW_INDIRECT", GL_DRAW_INDIRECT_BUFFER);
//#endif	
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
