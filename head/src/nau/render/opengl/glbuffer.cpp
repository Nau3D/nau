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
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, m_UIntProps[SIZE], NULL, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}


GLBuffer::~GLBuffer() {

	glDeleteBuffers(1, (GLuint*)&m_IntProps[ID]);
}


IBuffer *
GLBuffer::clone() {

	GLBuffer *b = new GLBuffer();
	b->m_Label = this->m_Label;
	AttributeValues::copy(b);
	return (IBuffer *)b;
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
GLBuffer::clear() {
	unsigned char c = 0;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_IntProps[ID]);
	glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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
			if (prop == BINDING_POINT) {
				int i = *(int *)value;
				if (i == -1)
					glBindBufferBase(m_EnumProps[TYPE], m_IntProps[BINDING_POINT], 0);
				else
					glBindBufferBase(m_EnumProps[TYPE], i, m_IntProps[ID]);
			}
			m_IntProps[prop] = *(int *)value;
			break;
		case Enums::UINT:
			m_UIntProps[prop] = *(unsigned int *)value;
			break;
		case Enums::ENUM:
			m_EnumProps[prop] = *(int *)value;
			break;
		case Enums::BOOL:
			m_BoolProps[prop] = *(bool *)value;
			break;
	}
}

#endif