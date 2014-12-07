#include <nau/render/opengl/glmaterialbuffer.h>

#include <GL/glew.h>

using namespace nau::render;

bool
GLMaterialBuffer::Init() {

	Attribs.setDefault("TYPE", new int(GL_ARRAY_BUFFER));
#if NAU_OPENGL_VERSION >= 420
	Attribs.listAdd("TYPE", "ATOMIC_COUNTER", GL_ATOMIC_COUNTER_BUFFER);
#endif
#if NAU_OPENGL_VERSION >= 430
	Attribs.listAdd("TYPE", "SHADER_STORAGE", GL_SHADER_STORAGE_BUFFER);
#endif	
	Attribs.listAdd("TYPE", "ARRAY", GL_ARRAY_BUFFER);
	return true;
}

bool GLMaterialBuffer::Inited = Init();

GLMaterialBuffer::GLMaterialBuffer() {

	initArrays(Attribs);
}


GLMaterialBuffer::~GLMaterialBuffer() {

}

void
GLMaterialBuffer::bind() {

	int id = m_Buffer->getPropi(IBuffer::ID);
	glBindBufferBase(m_EnumProps[TYPE], m_IntProps[BINDING_POINT], id);
	glBindBuffer(m_EnumProps[TYPE], id);
}


void
GLMaterialBuffer::unbind() {

	int id = m_Buffer->getPropi(IBuffer::ID);
	glBindBufferBase(m_EnumProps[TYPE], m_IntProps[BINDING_POINT], -1);
	glBindBuffer(m_EnumProps[TYPE], 0);
}
