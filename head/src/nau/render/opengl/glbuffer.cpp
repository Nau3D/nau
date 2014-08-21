#include <GL/glew.h>
#include <nau/render/opengl/glbuffer.h>

#if NAU_OPENGL_VERSION >= 420

using namespace nau::render;

bool
GLBuffer::Init() {

	Attribs.setDefault("TYPE", new int(GL_SHADER_STORAGE_BUFFER));
	Attribs.listAdd("TYPE", "SHADER_STORAGE", GL_SHADER_STORAGE_BUFFER);
	return true;
}


bool GLBuffer::Inited = Init();


GLBuffer::GLBuffer(std::string label, int size) {

	initArrays(Attribs);

	m_Label = label;
	m_UIntProps[SIZE] = size;

	glGenBuffers(1, (GLuint *)&m_IntProps[ID]);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_IntProps[ID]);
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, m_UIntProps[SIZE], NULL, GL_MAP_READ_BIT);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}


GLBuffer::~GLBuffer() {

	glDeleteBuffers(1, (GLuint*)&m_IntProps[ID]);
}


void
GLBuffer::bind() {

	m_BoolProps[BIND] = true;
	glBindBuffer(m_EnumProps[TYPE], m_IntProps[ID]);
}


void
GLBuffer::unbind() {

	m_BoolProps[BIND] = false;
	glBindBuffer(m_EnumProps[TYPE], 0);
}


void
GLBuffer::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {

		case Enums::FLOAT:
			m_FloatProps[prop] = *(float *)value;
			break;
		case Enums::VEC4:
			m_Float4Props[prop].set((vec4 *)value);
			break;
		case Enums::INT:
			m_IntProps[prop] = *(int *)value;
			if (prop == BINDING_POINT)
				glBindBufferBase(m_EnumProps[TYPE], m_IntProps[BINDING_POINT], m_IntProps[ID]);
			break;
		case Enums::UINT:
			m_UIntProps[prop] = *(unsigned int *)value;
			break;
		case Enums::ENUM:
			m_EnumProps[prop] = *(int *)value;
			break;
	}
}

#endif